from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_val(self) -> float:
        pass

    def get_val_step(self) -> float:
        val = self.get_val()
        self.step()
        return val


class ConstantScheduler(Scheduler):
    def __init__(self, val, **kwargs):
        self.val = val

    def step(self):
        pass

    def get_val(self):
        return self.val


class LinearScheduler(Scheduler):
    def __init__(self, start_val, end_val, change_iters, **kwargs):
        self.start_val = start_val
        self.end_val = end_val
        self.change_iters = change_iters
        self.curr_iter = 0
        self.delta = (end_val - start_val) / change_iters

    def step(self):
        self.curr_iter += 1

    def get_val(self):
        if self.curr_iter > self.change_iters:
            return self.end_val
        val = self.start_val + self.delta * self.curr_iter
        return val
