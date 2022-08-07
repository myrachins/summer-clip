import torch
import torch.nn.functional as F
import utils
import collections
import datasets


gen_losses = utils.ClassRegistry()
disc_losses = utils.ClassRegistry()


@disc_losses.add_to_registry("bce")
def binary_cross_entopy(logits_real, logits_fake):
    labels_real = torch.ones_like(logits_real)
    loss = F.binary_cross_entropy_with_logits(logits_real, labels_real)
    labels_fake = torch.zeros_like(logits_fake)
    loss += F.binary_cross_entropy_with_logits(logits_fake, labels_fake)
    return loss


@disc_losses.add_to_registry("ce")
def cross_entopy(logits_real, logits_fake, y_real, y_fake, n_classes=10):
    loss = F.cross_entropy(logits_real, y_real)
    loss += F.cross_entropy(logits_fake, y_fake + n_classes)
    return loss


@gen_losses.add_to_registry("ce")
def cross_entopy(logits_fake, y_fake):
    loss = F.cross_entropy(logits_fake, y_fake)
    return loss


@gen_losses.add_to_registry("st_bce")
def saturating_bce_loss(logits_fake):
    zeros_label = torch.zeros_like(logits_fake)
    loss = -F.binary_cross_entropy_with_logits(logits_fake, zeros_label)
    return loss


@gen_losses.add_to_registry("nonst_bce")
def nonsaturating_bce_loss(logits_fake):
    ones_label = torch.ones_like(logits_fake)
    loss = F.binary_cross_entropy_with_logits(logits_fake, ones_label)
    return loss


@gen_losses.add_to_registry("kl")
def kl_gan_cross_entropy(logits_fake):
    ones_label = torch.ones_like(logits_fake)
    zeros_label = torch.zeros_like(logits_fake)
    loss = F.binary_cross_entropy_with_logits(logits_fake, ones_label)
    loss -= F.binary_cross_entropy_with_logits(logits_fake, zeros_label)
    return loss


@disc_losses.add_to_registry("hinge")
def disc_hinge_loss(logits_real, logits_fake):
    loss = F.relu(1.0 - logits_real).mean()
    loss += F.relu(1.0 + logits_fake).mean()
    return loss


@gen_losses.add_to_registry("hinge")
def gen_hinge_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


@disc_losses.add_to_registry("wgan")
def disc_wgan_loss(logits_real, logits_fake):
    loss = -logits_real.mean() + logits_fake.mean()
    return loss


@gen_losses.add_to_registry("wgan")
def gen_wgan_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


@disc_losses.add_to_registry("lsgan")
def disc_lsgan_loss(logits_real, logits_fake):
    loss = (logits_real - 1).square().mean()
    loss += logits_fake.square().mean()
    return loss


@gen_losses.add_to_registry("lsgan")
def gen_lsgan_loss(logits_fake):
    loss = (logits_fake - 1).square().mean()
    return loss


@gen_losses.add_to_registry("feature")
def feature_loss(act1, act2):
    loss = 0.0
    for a1, a2 in zip(act1, act2):
        loss += F.l1_loss(a1, a2).mean()
    return loss


@gen_losses.add_to_registry("l1raw")
def l1raw_loss(x_real, x_fake):
    if x_real.shape != x_fake.shape:
        x_real = x_real.squeeze(1)
    return F.l1_loss(x_real, x_fake).mean()


@gen_losses.add_to_registry("l1spec")
def l1spec_loss(x_real, x_fake):
    mel_real = datasets.mel_spectrogram(
        x_real.squeeze(1),
        1024,
        80,
        22050,
        256,
        1024,
        0,
        None,
    )
    mel_fake = datasets.mel_spectrogram(
        x_fake.squeeze(1),
        1024,
        80,
        22050,
        256,
        1024,
        0,
        None,
    )
    return F.l1_loss(mel_real, mel_fake).mean()

def stft_loss(x_real, x_fake, n_fft, hop_size, win_len):
    stft_real = torch.stft(x_real.squeeze(), n_fft, hop_size, win_len)
    stft_fake = torch.stft(x_fake.squeeze(), n_fft, hop_size, win_len)

    sc_loss = ((stft_real.abs() - stft_fake.abs()) ** 2).sum() / (stft_real.abs() ** 2).sum()
    mag_loss = F.l1_loss(torch.log(stft_real.abs() + 1e-6), torch.log(stft_fake.abs() + 1e-6))
    return sc_loss + mag_loss

@gen_losses.add_to_registry('mstft')
def mstft_loss(x_real, x_fake, n_ffts=[512, 1024, 2048], hop_sizes=[50, 120, 240], win_lens=[240, 600, 1200]):
    assert len(n_ffts) == len(hop_sizes) == len(win_lens)
    loss = 0
    for i in range(len(n_ffts)):
        loss += stft_loss(x_real, x_fake, n_ffts[i], hop_sizes[i], win_lens[i])
    loss /= len(n_ffts)
    return loss

@gen_losses.add_to_registry('mse_mag_spec')
def mse_mag_spec(x_real, x_fake):
    real_spec = torch.stft(x_real.squeeze(), 256)[:, 128:, :, 0] # Only real upper part of the spectrogram because we predict only this one
    fake_spec = torch.stft(x_fake.squeeze(), 256)[:, 128:, :, 0]
    return F.mse_loss(real_spec, fake_spec)

@gen_losses.add_to_registry("feat_comp_l1spec")
def feat_comp_l1spec_loss(spec_real, spec_fake):
    return F.l1_loss(spec_real, spec_fake).mean()


class CombinedLoss:
    def __init__(self, loss_funcs, loss_coefs):
        self.loss_funcs = loss_funcs
        num_losses = len(self.loss_funcs)
        self.loss_coefs = loss_coefs
        if not loss_coefs:
            self.loss_coefs = [1] * num_losses
        assert len(self.loss_coefs) == num_losses


class CombinedGenLoss(CombinedLoss):
    adv_losses = ("nonst_bce", "kl", "hinge", "lsgan")

    def __call__(
        self, logits_fake
    ):
        losses = collections.defaultdict(float)
        for func, coef in zip(self.loss_funcs, self.loss_coefs):
            if func in self.adv_losses:
                if isinstance(logits_fake, dict):
                    for key in logits_fake.keys():
                        loss_key = f"{func}_{key}"
                        losses[loss_key] = gen_losses[func](logits_fake[key])
                        losses[func] += losses[loss_key]
                else:
                    losses[func] = gen_losses[func](logits_fake)
                losses["total"] += coef * losses[func]
            else:
                raise NotImplementedError(f"{func} loss is not implemented!")

        for func in self.loss_funcs + ["total"]:
            losses[f"agg/{func}"] = losses[func]

        return losses


class CombinedDiscLoss(CombinedLoss):
    def __call__(self, logits_real, logits_fake, **kwargs):
        losses = collections.defaultdict(float)
        for func, coef in zip(self.loss_funcs, self.loss_coefs):
            if isinstance(logits_real, dict):
                for key in logits_real.keys():
                    loss_key = f"{func}_{key}"
                    losses[loss_key] = disc_losses[func](
                        logits_real[key], logits_fake[key]
                    )
                    losses[func] += losses[loss_key]
            else:
                losses[func] = disc_losses[func](logits_real, logits_fake)
            losses["total"] += coef * losses[func]

        for func in self.loss_funcs + ["total"]:
            losses[f"agg/{func}"] = losses[func]

        return losses
