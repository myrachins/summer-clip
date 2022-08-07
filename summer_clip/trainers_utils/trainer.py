import os
import copy
import torch
import torch.utils.data
import torch.nn.functional
import losses
import optims
import datasets
import models
import utils
import log_utils
import omegaconf
from pytorch_gan_metrics import get_inception_score_and_fid


trainers = utils.ClassRegistry()


@trainers.add_to_registry(name="base")
class BaseTrainer:
    def __init__(self, config):
        # initialized in setup_logger()
        self.logger = None
        self.disc_logger = None
        self.run_dir = None

        # initialized in setup_dataset()
        self.dataset = None

        # initialized in setup_loaders()
        self.loaders = None

        # initialized in setup_networks()
        self.gen = None
        self.discs = None
        self.nets = None
        self.checkpoint_dir = None

        # initialized in setup_optimizers()
        self.optim_disc = None
        self.optim_gen = None
        self.optims = None

        # initialized in setup_losses()
        self.combined_gen_loss = None
        self.combined_disc_loss = None

        # initialized in setup_num_iters()
        self.num_iters = None
        self.num_val_iters = None

        # initialized in setup_metrics()
        self.metrics = None

        self.config = config

    def setup_logger(self):
        config_for_logger = omegaconf.OmegaConf.to_container(self.config)
        config_for_logger["PID"] = os.getpid()
        exp_logger = log_utils.WandbLogger(
            project=self.config.exp.project,
            name=self.config.exp.name,
            dir=self.config.exp.root,
            tags=tuple(self.config.exp.tags) if self.config.exp.tags else None,
            notes=self.config.exp.notes,
            config=config_for_logger,
        )
        self.run_dir = exp_logger.run_dir
        console_logger = log_utils.ConsoleLogger(self.config.exp.name)
        self.logger = log_utils.LoggingManager(exp_logger, console_logger)
        self.disc_logger = log_utils.DiscriminatorLogging(self.logger)

    def setup_prior(self):
        dist_name = self.config.prior.dist
        self.prior = datasets.priors[dist_name](**self.config.dist_prior[dist_name])

    def setup_dataset(self):
        self.dataset = dict()
        for data_type in ["train", "test"]:
            dataset = datasets.datasets[self.config.data.name](
                **self.config.dataset[self.config.data.name][data_type]
            )
            self.dataset[data_type] = dataset

    def setup_loaders(self):
        self.loaders = dict()
        loader_type = self.config.data.loader
        loader_args = self.config.loader[loader_type]
        self.loaders["prior"] = datasets.loaders[loader_type](
            self.prior,
            **loader_args["prior"],
        )
        for data_type in ["train", "test"]:
            self.loaders[data_type] = datasets.loaders[loader_type](
                self.dataset[data_type],
                **loader_args[data_type],
            )

    def setup_networks(self):
        print("\nGenerator:")
        z = next(self.loaders["prior"])
        self.gen = models.generators[self.config.gen.model](
            **self.config.gennets[self.config.gen.model],
        )
        log_utils.print_network(
            self.gen,
            f"gen_{self.config.gen.model}",
            z,
            self.logger,
            self.config.log.log_complexity,
        )

        print("\nx Discriminator")
        x, _ = next(self.loaders["train"])
        self.disc = models.discriminators[self.config.disc.model](
            **self.config.discnets[self.config.disc.model]
        )
        log_utils.print_network(
            self.disc,
            f"disc_{self.config.disc.model}",
            x,
            self.logger,
            self.config.log.log_complexity,
        )

        self.nets = dict(gen=self.gen, disc=self.disc)
        self.to(self.config.training.device)

        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)

    def to(self, device):
        for net in self.nets.values():
            net.to(device)

    def train_mode(self):
        for net in self.nets.values():
            net.train()

    def eval_mode(self):
        for net in self.nets.values():
            net.eval()

    def setup_optimizers(self):
        self.optim_gen = optims.optims[self.config.gen.optim](
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            **self.config.gen_optim[self.config.gen.optim],
        )
        self.optim_disc = optims.optims[self.config.disc.optim](
            filter(lambda p: p.requires_grad, self.disc.parameters()),
            **self.config.disc_optim[self.config.disc.optim],
        )
        self.optims = dict(optim_gen=self.optim_gen, optim_disc=self.optim_disc)

    def setup_schedulers(self):
        self.scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_gen, gamma=self.config.gen.lr_decay, last_epoch=self.start_epoch
        )
        self.scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_disc,
            gamma=self.config.disc.lr_decay,
            last_epoch=self.start_epoch,
        )
        self.schedulers = dict(
            scheduler_gen=self.scheduler_gen, scheduler_disc=self.scheduler_disc
        )

    def setup_losses(self):
        self.combined_gen_loss = losses.CombinedGenLoss(
            self.config.gen.loss_funcs, self.config.gen.loss_coefs
        )
        self.combined_disc_loss = losses.CombinedDiscLoss(
            self.config.disc.loss_funcs, self.config.disc.loss_coefs
        )

    def setup_num_iters(self):
        num_batches = len(self.dataset["train"]) // self.config.training.batch_size
        self.num_iters = (
            self.config.training.num_iters
            if self.config.training.num_iters
            else num_batches
        )
        num_val_batches = (
            len(self.dataset["test"]) // self.config.training.val_batch_size
        )
        self.num_val_iters = (
            self.config.training.num_val_iters
            if self.config.training.num_val_iters
            else num_val_batches
        )

    def setup(self):
        self.setup_logger()
        self.setup_prior()
        self.setup_dataset()
        self.setup_loaders()
        self.setup_networks()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_num_iters()

    def compute_metrics(self, epoch_num, epoch_info):
        val_size = (
            self.config.log.small_val_size
            if epoch_num % self.config.log.calculate_every or epoch_num == 0
            else self.config.log.val_size
        )
        gen_imgs = self.gen.generate(
            val_size, self.config.training.batch_size, self.loaders["prior"]
        )
        gen_imgs = (gen_imgs + 1) / 2
        IS, FID = get_inception_score_and_fid(
            gen_imgs, self.config.metrics.fid_cache, verbose=True, use_torch=False
        )
        epoch_info["metrics/IS"] = IS[0]
        epoch_info["metrics_std/IS"] = IS[1]
        epoch_info["metrics/FID"] = FID

    def disc_loss(self, x_real, z, iter_info, no_grad=False, log_event="train"):
        log_event += "_disc"

        with torch.no_grad() if no_grad else torch.enable_grad():
            x_fake = self.gen(z)
            logits_real = self.disc(x_real)
            logits_fake = self.disc(x_fake)
        losses = self.combined_disc_loss(logits_real, logits_fake)

        if not self.config.log.disc_logging_off:
            self.disc_logger.log_disc(
                iter_info, logits_real, logits_fake, x_real, z, log_event
            )
        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def gen_loss(self, z, iter_info, no_grad=False, log_event="train"):
        log_event += "_gen"

        with torch.no_grad() if no_grad else torch.enable_grad():
            x_fake = self.gen(z)
            logits_fake = self.disc(x_fake)
        losses = self.combined_gen_loss(logits_fake)

        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def train_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingMeans()

        epoch_time_log = log_utils.TimeLog(self.logger, self.num_iters, event="epoch")

        for iter_num in range(self.num_iters):
            with log_utils.Timer(iter_info, "iter_train"):
                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc.zero_grad()
                    loss = self.disc_loss(x_real, z, iter_info)
                    loss.backward()
                    self.optim_disc.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen.zero_grad()
                    loss = self.gen_loss(z, iter_info)
                    loss.backward()
                    self.optim_gen.step()

            if iter_num % self.config.log.log_every == 0:
                self.logger.log_iter(iter_num, epoch_num, self.num_iters, iter_info)
                epoch_info.update(iter_info)
                iter_info.clear()

                epoch_time_log.now(iter_num)
        epoch_time_log.end()
        return epoch_info

    def validation_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingMeans()

        epoch_time_log = log_utils.TimeLog(
            self.logger, self.num_val_iters, event="validation"
        )

        for iter_num in range(self.num_val_iters):
            with log_utils.Timer(iter_info, "iter_val"):
                x_real, _ = next(self.loaders["test"])
                z = next(self.loaders["prior"])

                self.disc_loss(x_real, z, iter_info, no_grad=True, log_event="val")
                self.gen_loss(z, iter_info, no_grad=True, log_event="val")

            if iter_num % self.config.log.log_every == 0:
                self.logger.log_iter(
                    iter_num,
                    epoch_num,
                    self.num_val_iters,
                    iter_info,
                    event="val",
                )
                epoch_info.update(iter_info)
                iter_info.clear()

                epoch_time_log.now(iter_num)
        epoch_time_log.end()
        return epoch_info

    def generate_images(self, epoch_num):
        images = dict()
        with torch.no_grad():
            images["samples"] = self.gen(next(self.loaders["prior"]))
        if epoch_num == 0:
            images["real-samples"] = next(self.loaders["test"])
            if isinstance(images["real-samples"], list):
                images["real-samples"] = images["real-samples"][0]
        return images

    def train_loop(self):
        training_time_log = log_utils.TimeLog(
            self.logger, self.config.training.num_epochs + 1, event="training"
        )
        self.start_epoch = log_utils.restore_checkpoint(
            self.run_dir,
            self.config.exp.name,
            self.nets,
            self.optims,
            self.config.training.device,
            self.logger.console_logger.logger,
        )
        self.setup_schedulers()
        for epoch_num in range(
            self.start_epoch + 1, self.config.training.num_epochs + 1
        ):
            epoch_info = log_utils.StreamingMeans()
            self.train_mode()
            self.setup_loaders()
            with log_utils.Timer(epoch_info, "epoch_train"):
                epoch_info = self.train_epoch(epoch_num, epoch_info)

            for scheduler in self.schedulers.values():
                scheduler.step()

            self.eval_mode()
            with log_utils.Timer(epoch_info, "epoch_val"):
                epoch_info = self.validation_epoch(epoch_num, epoch_info)

            if epoch_num % self.config.log.calculate_every == 0:
                self.compute_metrics(epoch_num, epoch_info)

            images = self.generate_images(epoch_num)
            self.logger.log_epoch(epoch_num, epoch_info, images)

            if not self.config.log.disc_logging_off:
                self.disc_logger.log_tough_samples(epoch_num)

            if not self.config.checkpoint.checkpointing_off:
                checkpoint_dir = os.path.join(
                    self.run_dir, self.config.checkpoint.checkpoint_dir
                )
                log_utils.save_checkpoint(
                    checkpoint_dir,
                    self.config.exp.name,
                    epoch_num,
                    self.nets,
                    self.optims,
                    self.config.training.device,
                    save_full=True,
                )
                if (
                    epoch_num - 1
                ) % self.config.checkpoint.save_every != 0 or epoch_num - 1 == 0:
                    log_utils.remove_checkpoint(
                        checkpoint_dir,
                        epoch_num - 1,
                        self.nets,
                        self.optims,
                        remove_full=True,
                    )
                elif (epoch_num - 1) % self.config.checkpoint.save_full_every != 0:
                    log_utils.remove_checkpoint(
                        checkpoint_dir,
                        epoch_num - 1,
                        self.nets,
                        self.optims,
                        remove_full=False,
                    )

            training_time_log.now(epoch_num)
        training_time_log.end()


@trainers.add_to_registry(name="avg_gan")
class AvgGAN(BaseTrainer):
    def setup_networks(self):
        print("\nGenerator:")
        z = next(self.loaders["prior"])
        self.gen = models.generators[self.config.gen.model](
            **self.config.gennets[self.config.gen.model],
        )
        log_utils.print_network(
            self.gen,
            f"gen_{self.config.gen.model}",
            z,
            self.logger,
            self.config.log.log_complexity,
        )
        self.gens = dict()
        for key in ["avg", "ema"]:
            self.gens[key] = copy.deepcopy(self.gen)

        print("\nx Discriminator")
        x, _ = next(self.loaders["train"])
        self.disc = models.discriminators[self.config.disc.model](
            **self.config.discnets[self.config.disc.model]
        )
        log_utils.print_network(
            self.disc,
            f"disc_{self.config.disc.model}",
            x,
            self.logger,
            self.config.log.log_complexity,
        )

        self.nets = dict(gen=self.gen, disc=self.disc, **self.gens)
        self.to(self.config.training.device)

        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)

    def train_mode(self):
        for name, net in self.nets.items():
            if name not in self.gens:
                net.train()

    def eval_mode(self):
        for name, net in self.nets.items():
            if name not in self.gens:
                net.eval()

    def compute_metrics(self, epoch_num, epoch_info):
        val_size = (
            self.config.log.small_val_size
            if epoch_num % self.config.log.calculate_every or epoch_num == 0
            else self.config.log.val_size
        )
        gen_imgs = self.gen.generate(
            val_size, self.config.training.batch_size, self.loaders["prior"]
        )
        gen_imgs = (gen_imgs + 1) / 2
        IS, FID = get_inception_score_and_fid(
            gen_imgs, self.config.metrics.fid_cache, verbose=True, use_torch=False
        )
        epoch_info["metrics/IS"] = IS[0]
        epoch_info["metrics_std/IS"] = IS[1]
        epoch_info["metrics/FID"] = FID

        for name, gen in self.gens.items():
            gen_imgs = gen.generate(
                val_size, self.config.training.batch_size, self.loaders["prior"]
            )
            gen_imgs = (gen_imgs + 1) / 2
            IS, FID = get_inception_score_and_fid(
                gen_imgs, self.config.metrics.fid_cache, verbose=True, use_torch=False
            )
            epoch_info[f"metrics/IS_{name}"] = IS[0]
            epoch_info[f"metrics_std/IS_{name}"] = IS[1]
            epoch_info[f"metrics/FID_{name}"] = FID

    def update_gens(self, step, beta_ema=0.9999):
        l_param = list(self.gen.parameters())
        l_avg_param = list(self.gens["avg"].parameters())

        for i in range(len(l_param)):
            with torch.no_grad():
                l_avg_param[i].data.copy_(
                    l_avg_param[i]
                    .data.mul(step)
                    .div(step + 1.0)
                    .add(l_param[i].data.div(step + 1.0))
                )

        l_param = list(self.gen.parameters())
        l_ema_param = list(self.gens["ema"].parameters())

        for i in range(len(l_param)):
            with torch.no_grad():
                l_ema_param[i].data.copy_(
                    l_ema_param[i]
                    .data.mul(beta_ema)
                    .add(l_param[i].data.mul(1 - beta_ema))
                )

    def train_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingMeans()

        epoch_time_log = log_utils.TimeLog(self.logger, self.num_iters, event="epoch")

        for iter_num in range(self.num_iters):
            with log_utils.Timer(iter_info, "iter_train"):
                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc.zero_grad()
                    loss = self.disc_loss(x_real, z, iter_info)
                    loss.backward(create_graph=True)
                    self.optim_disc.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen.zero_grad()
                    loss = self.gen_loss(z, iter_info)
                    loss.backward(create_graph=True)
                    self.optim_gen.step()

                self.update_gens(epoch_num * self.config.training.num_iters + iter_num)

            if iter_num % self.config.log.log_every == 0:
                self.logger.log_iter(iter_num, epoch_num, self.num_iters, iter_info)
                epoch_info.update(iter_info)
                iter_info.clear()

                epoch_time_log.now(iter_num)
        epoch_time_log.end()
        return epoch_info


@trainers.add_to_registry(name="extra_gan")
class ExtraGAN(AvgGAN):
    def setup_networks(self):
        super().setup_networks()

        self.gen_extra = copy.deepcopy(self.gen)
        self.disc_extra = copy.deepcopy(self.disc)
        self.nets.update(dict(gen_extra=self.gen_extra, disc_extra=self.disc_extra))
        self.to(self.config.training.device)

    def setup_optimizers(self):
        super().setup_optimizers()
        self.optim_gen_extra = optims.optims[self.config.gen.optim](
            filter(lambda p: p.requires_grad, self.gen_extra.parameters()),
            **self.config.gen_optim[self.config.gen.optim],
        )
        self.optim_disc_extra = optims.optims[self.config.disc.optim](
            filter(lambda p: p.requires_grad, self.disc_extra.parameters()),
            **self.config.disc_optim[self.config.disc.optim],
        )
        self.optims.update(
            dict(
                optim_gen_extra=self.optim_gen_extra,
                optim_disc_extra=self.optim_disc_extra,
            )
        )

    def _sync_extra_nets(self):
        self.gen_extra.load_state_dict(self.gen.state_dict())
        self.disc_extra.load_state_dict(self.disc.state_dict())

    def disc_loss(self, x_real, z, iter_info, no_grad=False, log_event="train"):
        log_event += "_disc"

        with torch.no_grad() if no_grad else torch.enable_grad():
            x_fake = self.gen_extra(z)
            logits_real = self.disc(x_real)
            logits_fake = self.disc(x_fake)
        losses = self.combined_disc_loss(logits_real, logits_fake)

        if not self.config.log.disc_logging_off:
            self.disc_logger.log_disc(
                iter_info, logits_real, logits_fake, x_real, z, log_event
            )
        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def disc_extra_loss(self, x_real, z, iter_info, no_grad=False, log_event="train"):
        log_event += "_disc"

        with torch.no_grad() if no_grad else torch.enable_grad():
            x_fake = self.gen(z)
            logits_real = self.disc_extra(x_real)
            logits_fake = self.disc_extra(x_fake)
        losses = self.combined_disc_loss(logits_real, logits_fake)

        if not self.config.log.disc_logging_off:
            self.disc_logger.log_disc(
                iter_info, logits_real, logits_fake, x_real, z, log_event
            )
        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def gen_loss(self, z, iter_info, no_grad=False, log_event="train"):
        log_event += "_gen"

        with torch.no_grad() if no_grad else torch.enable_grad():
            x_fake = self.gen(z)
            logits_fake = self.disc_extra(x_fake)
        losses = self.combined_gen_loss(logits_fake)

        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def gen_extra_loss(self, z, iter_info, no_grad=False, log_event="train"):
        log_event += "_gen"

        with torch.no_grad() if no_grad else torch.enable_grad():
            x_fake = self.gen_extra(z)
            logits_fake = self.disc(x_fake)
        losses = self.combined_gen_loss(logits_fake)

        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def train_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingMeans()

        epoch_time_log = log_utils.TimeLog(self.logger, self.num_iters, event="epoch")

        for iter_num in range(self.num_iters):
            self._sync_extra_nets()
            with log_utils.Timer(iter_info, "iter_train"):
                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc_extra.zero_grad()
                    loss = self.disc_extra_loss(x_real, z, iter_info)
                    loss.backward()
                    self.optim_disc_extra.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen_extra.zero_grad()
                    loss = self.gen_extra_loss(z, iter_info)
                    loss.backward()
                    self.optim_gen_extra.step()

                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc.zero_grad()
                    loss = self.disc_loss(x_real, z, iter_info)
                    loss.backward()
                    self.optim_disc.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen.zero_grad()
                    loss = self.gen_loss(z, iter_info)
                    loss.backward()
                    self.optim_gen.step()

                self.update_gens(epoch_num * self.config.training.num_iters + iter_num)

            if iter_num % self.config.log.log_every == 0:
                self.logger.log_iter(iter_num, epoch_num, self.num_iters, iter_info)
                epoch_info.update(iter_info)
                iter_info.clear()

                epoch_time_log.now(iter_num)
        epoch_time_log.end()
        return epoch_info


@trainers.add_to_registry(name="extra_oasis")
class ExtraOASIS(ExtraGAN):
    def setup_optimizers(self):
        assert self.config.gen.optim in ("oasis", "oasis2", "adahess")
        assert self.config.disc.optim in ("oasis", "oasis2", "adahess")
        super().setup_optimizers()

    def train_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingMeans()

        epoch_time_log = log_utils.TimeLog(self.logger, self.num_iters, event="epoch")

        for iter_num in range(self.num_iters):
            self._sync_extra_nets()
            with log_utils.Timer(iter_info, "iter_train"):
                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc_extra.zero_grad()
                    loss = self.disc_extra_loss(x_real, z, iter_info)
                    loss.backward(create_graph=True)
                    self.optim_disc_extra.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen_extra.zero_grad()
                    loss = self.gen_extra_loss(z, iter_info)
                    loss.backward(create_graph=True)
                    self.optim_gen_extra.step()

                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc.zero_grad()
                    loss = self.disc_loss(x_real, z, iter_info)
                    loss.backward(create_graph=True)
                    self.optim_disc.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen.zero_grad()
                    loss = self.gen_loss(z, iter_info)
                    loss.backward(create_graph=True)
                    self.optim_gen.step()

                self.update_gens(epoch_num * self.config.training.num_iters + iter_num)

            if iter_num % self.config.log.log_every == 0:
                self.logger.log_iter(iter_num, epoch_num, self.num_iters, iter_info)
                epoch_info.update(iter_info)
                iter_info.clear()

                epoch_time_log.now(iter_num)
        epoch_time_log.end()
        return epoch_info


@trainers.add_to_registry(name="la-gan")
class LAGAN(AvgGAN):
    def setup_networks(self):
        print("\nGenerator:")
        z = next(self.loaders["prior"])
        self.gen = models.generators[self.config.gen.model](
            **self.config.gennets[self.config.gen.model],
        )
        log_utils.print_network(
            self.gen,
            f"gen_{self.config.gen.model}",
            z,
            self.logger,
            self.config.log.log_complexity,
        )
        self.gens = dict()
        for key in ["avg", "ema", "avg_slow", "ema_slow"]:
            self.gens[key] = copy.deepcopy(self.gen)

        print("\nx Discriminator")
        x, _ = next(self.loaders["train"])
        self.disc = models.discriminators[self.config.disc.model](
            **self.config.discnets[self.config.disc.model]
        )
        log_utils.print_network(
            self.disc,
            f"disc_{self.config.disc.model}",
            x,
            self.logger,
            self.config.log.log_complexity,
        )

        self.nets = dict(gen=self.gen, disc=self.disc)
        self.to(self.config.training.device)

        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)

    def setup_optimizers(self):
        self.optim_gen = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            lr=self.config.gen.lr,
            betas=tuple(self.config.gen.betas),
        )
        self.optim_disc = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.disc.parameters()),
            lr=self.config.disc.lr,
            betas=tuple(self.config.disc.betas),
        )
        optim_name = self.config.gen.optim
        optim_args = self.config.gen_optim[optim_name]
        self.optim_gen = optims.optims[optim_name](self.optim_gen, **optim_args)
        self.la_k = optim_args["k"]
        self.la_super_slow_k = optim_args["super_slow_k"]

        optim_name = self.config.disc.optim
        optim_args = self.config.disc_optim[optim_name]
        self.optim_disc = optims.optims[optim_name](self.optim_disc, **optim_args)

        self.optims = dict(optim_gen=self.optim_gen, optim_disc=self.optim_disc)

    def update_slow_gens(self, step, beta_ema=0.9999):
        l_param = list(self.gen.parameters())
        l_avg_param = list(self.gens["avg_slow"].parameters())

        for i in range(len(l_param)):
            with torch.no_grad():
                l_avg_param[i].data.copy_(
                    l_avg_param[i]
                    .data.mul(step)
                    .div(step + 1.0)
                    .add(l_param[i].data.div(step + 1.0))
                )

        l_param = list(self.gen.parameters())
        l_ema_param = list(self.gens["ema_slow"].parameters())

        for i in range(len(l_param)):
            with torch.no_grad():
                l_ema_param[i].data.copy_(
                    l_ema_param[i]
                    .data.mul(beta_ema)
                    .add(l_param[i].data.mul(1 - beta_ema))
                )

    def train_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingMeans()

        epoch_time_log = log_utils.TimeLog(self.logger, self.num_iters, event="epoch")

        for iter_num in range(self.num_iters):
            with log_utils.Timer(iter_info, "iter_train"):
                for _ in range(self.config.disc.num_steps):
                    x_real, _ = next(self.loaders["train"])
                    z = next(self.loaders["prior"])

                    self.optim_disc.zero_grad()
                    loss = self.disc_loss(x_real, z, iter_info)
                    loss.backward()
                    self.optim_disc.step()
                for _ in range(self.config.gen.num_steps):
                    z = next(self.loaders["prior"])

                    self.optim_gen.zero_grad()
                    loss = self.gen_loss(z, iter_info)
                    loss.backward()
                    self.optim_gen.step()

                if (iter_num + 1) % self.la_k == 0:
                    self.optim_disc.update_lookahead()
                    self.optim_gen.update_lookahead()
                    if (
                        self.la_super_slow_k > 0
                        and (epoch_num * self.config.training.num_iters + iter_num + 1)
                        % self.la_super_slow_k
                        == 0
                    ):
                        self.update_slow_gens(
                            epoch_num * self.config.training.num_iters + iter_num,
                            beta_ema=0.9,
                        )
                    else:
                        self.update_slow_gens(
                            epoch_num * self.config.training.num_iters + iter_num
                        )
                if (
                    self.la_super_slow_k > 0
                    and (epoch_num * self.config.training.num_iters + iter_num + 1)
                    % self.la_super_slow_k
                    == 0
                ):
                    self.optim_disc.update_lookahead_super_slow()
                    self.optim_gen.update_lookahead_super_slow()

                self.update_gens(epoch_num * self.config.training.num_iters + iter_num)

            if iter_num % self.config.log.log_every == 0:
                self.logger.log_iter(iter_num, epoch_num, self.num_iters, iter_info)
                epoch_info.update(iter_info)
                iter_info.clear()

                epoch_time_log.now(iter_num)
        epoch_time_log.end()
        return epoch_info
