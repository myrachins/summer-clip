import time
import datetime
import collections
import logging
import os
import glob

import torch
import wandb


def strf_time_delta(td):
    td_str = ""
    if td.days > 0:
        td_str += f"{td.days} days, " if td.days > 1 else f"{td.days} day, "
    hours = td.seconds // 3600
    if hours > 0:
        td_str += f"{hours}h "
    minutes = (td.seconds // 60) % 60
    if minutes > 0:
        td_str += f"{minutes}m "
    seconds = td.seconds % 60 + td.microseconds * 1e-6
    td_str += f"{seconds:.1f}s"
    return td_str


class LoggingManager:
    def __init__(self, exp_logger, console_logger):
        self.exp_logger = exp_logger
        self.console_logger = console_logger

    def log_iter(self, iter_num, epoch_num, num_iters, iter_info, **kwargs):
        self.console_logger.log_iter(
            epoch_num, iter_num, num_iters, iter_info, **kwargs
        )

    def log_epoch(self, epoch_num, epoch_info):
        self.exp_logger.log(dict(epoch=epoch_num, **epoch_info.to_dict()))
        self.console_logger.log_epoch(epoch_num, epoch_info)

    def log_info(self, output_info):
        self.console_logger.logger.info(output_info)

    def log_wandb(self, output_info):
        self.exp_logger.log(output_info)

    def log_info_wandb(self, output_info):
        self.log_info(output_info)
        self.log_wandb(output_info)


class WandbLogger:
    def __init__(self, **kwargs):
        wandb.init(**kwargs)
        self.run_dir = wandb.run.dir  # type: ignore
        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob("**/*.py", recursive=True):
            if not path.startswith("wandb"):
                if os.path.basename(path) != path:
                    code.add_dir(
                        os.path.dirname(path), name=os.path.dirname(path)
                    )
                else:
                    code.add_file(os.path.basename(path), name=path)
        wandb.run.log_artifact(code)  # type: ignore

    def log(self, info, **kwargs):
        wandb.log(info, **kwargs)

    def log_images(self, name, imgs, epoch_num):
        wandb.log({name: wandb.Image(imgs, caption=f"epoch = {epoch_num}")})


class ConsoleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    @staticmethod
    def format_info(info):
        if not info:
            return str(info)
        log_groups = collections.defaultdict(dict)
        for k, v in info.to_dict().items():
            prefix, suffix = k.split("/", 1)
            log_groups[prefix][suffix] = (
                f"{v:.3f}" if isinstance(v, float) else str(v)
            )
        formatted_info = ""
        max_group_size = len(max(log_groups, key=len)) + 2
        max_k_size = (
            max([len(max(g, key=len)) for g in log_groups.values()]) + 1
        )
        max_v_size = (
            max([len(max(g.values(), key=len)) for g in log_groups.values()]) + 1
        )
        for group, group_info in log_groups.items():
            group_str = [
                f"{k:<{max_k_size}}={v:>{max_v_size}}"
                for k, v in group_info.items()
            ]
            max_g_size = len(max(group_str, key=len)) + 2
            group_str = "".join([f"{g:>{max_g_size}}" for g in group_str])
            formatted_info += f"\n{group + ':':<{max_group_size}}{group_str}"
        return formatted_info

    def log_iter(
        self, epoch_num, iter_num, num_iters, iter_info, event="epoch"
    ):
        output_info = (
            f"{event.upper()} {epoch_num}, ITER {iter_num}/{num_iters}:"
        )
        output_info += self.format_info(iter_info)
        self.logger.info(output_info)

    def log_epoch(self, epoch_num, epoch_info):
        output_info = f"EPOCH {epoch_num}:"
        output_info += self.format_info(epoch_info)
        self.logger.info(output_info)


class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event
        self.is_cuda = torch.cuda.is_available()

    def __enter__(self):
        if not self.is_cuda:
            return self
        self.start = torch.cuda.Event(enable_timing=True)  # type: ignore
        self.end = torch.cuda.Event(enable_timing=True)  # type: ignore
        self.start.record()  # type: ignore
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_cuda:
            return
        self.end.record()  # type: ignore
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info:
            self.info[f"duration/{self.log_event}"] = self.duration


class TimeLog:
    def __init__(self, logger, total_num, event):
        self.logger = logger
        self.total_num = total_num
        self.event = event.upper()
        self.start = time.time()

    def now(self, current_num):
        elapsed = time.time() - self.start
        left = self.total_num * elapsed / (current_num + 1) - elapsed
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        left = strf_time_delta(datetime.timedelta(seconds=left))
        self.logger.log_info(
            f"TIME ELAPSED SINCE {self.event} START: {elapsed}"
        )
        self.logger.log_info(f"TIME LEFT UNTIL {self.event} END: {left}")

    def end(self):
        elapsed = time.time() - self.start
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        self.logger.log_info(
            f"TIME ELAPSED SINCE {self.event} START: {elapsed}"
        )
        self.logger.log_info(f"{self.event} ENDS")


class _StreamingMean:
    def __init__(self, val=None, counts=None):
        if val is None:
            self.mean = 0.0
            self.counts = 0
        else:
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            self.mean = val
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1

    def update(self, mean, counts=1):
        if isinstance(mean, torch.Tensor):
            mean = mean.data.cpu().numpy()
        elif isinstance(mean, _StreamingMean):
            mean, counts = mean.mean, mean.counts * counts
        assert counts >= 0
        if counts == 0:
            return
        total = self.counts + counts
        self.mean = self.counts / total * self.mean + counts / total * mean
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.mean, self.counts)
        if isinstance(other, _StreamingMean):
            if other.counts == 0:
                return new
            else:
                new.update(other.mean, other.counts)
        else:
            new.update(other)
        return new


class StreamingMeans(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMean)

    def __setitem__(self, key, value):
        if isinstance(value, _StreamingMean):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, _StreamingMean(value))

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            self[k].update(v)

    def to_dict(self, prefix=""):
        return dict((prefix + k, v.mean) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])
