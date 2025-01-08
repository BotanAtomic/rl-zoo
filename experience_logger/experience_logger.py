import os
from datetime import datetime
from functools import wraps

import torch.nn
from torch.utils.tensorboard import SummaryWriter


def enabled_only(method):
    """Decorator to ensure the method only runs if logging is enabled."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.enable:
            return method(self, *args, **kwargs)

    return wrapper


class ExperienceLogger:
    def __init__(self, algorithm: str, enable: bool = True):
        self.algorithm = algorithm
        self.enable = enable

        self.log_dir: str | None = None
        self.summary_writer: SummaryWriter | None = None

        if enable:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = f"runs/{algorithm}/{current_time}"
            os.makedirs(self.log_dir + "/models", exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)

    @enabled_only
    def log_scalar(self, tag: str, value: float, step: int):
        self.summary_writer.add_scalar(tag, value, step)

    @enabled_only
    def log_model(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        self.summary_writer.add_graph(model, input_tensor)

    @enabled_only
    def save_model(self, model: torch.nn.Module, step: int):
        model_path = f"{self.log_dir}/models/{self.algorithm}_{step}.pt"
        torch.save(model.state_dict(), model_path)
