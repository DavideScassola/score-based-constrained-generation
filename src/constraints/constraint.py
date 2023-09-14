from dataclasses import dataclass
from typing import Any, Callable

import torch

from src.constraints.gradient_mixer import GradientMixer, linear_interpolation


@dataclass
class Constraint:
    f: Callable[[torch.Tensor], torch.Tensor]
    gradient_mixer: GradientMixer = linear_interpolation
    strength: float = 1.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.strength * self.f(x)
