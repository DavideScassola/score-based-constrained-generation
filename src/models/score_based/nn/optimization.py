from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Optimization:
    epochs: int
    batch_size: int
    optimizer_class: Callable
    optimizer_hyperparameters: dict
    gradient_clipping: float | None = None
    parameters_momentum: float | None = None

    def build_optimizer(self, params) -> torch.optim.Optimizer:
        return self.optimizer_class(params, **self.optimizer_hyperparameters)
