from abc import ABC, abstractmethod

import numpy as np
import torch

from src.constants import MINIMUM_SAMPLING_T
from src.models.score_based.sdes.sde import Sde
from src.util import gradient


def fix_bounds(g):
    """
    Rescales a function g(t) such that g(0)=1 and g(1)=0
    """
    if g(0.0) == 1 and g(1.0) == 0.0:
        return g
    h = -g(1.0) * (g(0.0) - g(1.0))
    m = 1.0 / (g(0.0) - g(1.0))
    return lambda t: g(t) * m + h


class GradientMixer(ABC):
    @abstractmethod
    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        pass


class NoConstraint(GradientMixer):
    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        return score


class TimeWeighting(GradientMixer):
    def __init__(self, g, rescale: bool = True) -> None:
        self.g = fix_bounds(g) if rescale else g

    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        return constraint_gradient * self.g(t) + score


class LogisticTimeWeighting(TimeWeighting):
    def __init__(self, *, k: float = 10.0, x0: float = 0.5) -> None:
        g = lambda t: 1 - 1 / (1 + np.exp(-(t - x0) * k))
        super().__init__(g, rescale=True)


def expected_score(*, n: int, t: float, sde: Sde, X: torch.Tensor, constraint):
    mean, std = sde.transition_kernel_mean_std(
        X=X,
        t=torch.full(
            size=(len(X), 1), fill_value=t, dtype=torch.float32, device=X.device
        ),
    )
    shape = [n] + list(X.shape)
    samples = torch.rand(size=shape, device=X.device) * std + mean
    # TODO: check reshaping
    return constraint.strength * gradient(
        f=constraint.f, X=samples.reshape([-1] + shape[2:])
    ).reshape(shape).mean(dim=0)


class NoisedConstraint(GradientMixer):
    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        X: torch.Tensor,
        constraint,
        **kwargs
    ) -> torch.Tensor:
        return (
            expected_score(constraint=constraint, t=t, n=self.n, X=X, sde=sde) + score
        )


class GradientClipping(GradientMixer):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = abs(k)

    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        _, std = sde.transition_kernel_mean_std(X=torch.tensor(0.0), t=torch.tensor(t))
        cap = (self.k / (std**2)).to(device=constraint_gradient.device)
        clipped_constraint_grad = torch.clip(constraint_gradient, min=-cap, max=cap)
        return score + clipped_constraint_grad


class NoiseWeighting(GradientMixer):
    def __init__(self, min_t: float = MINIMUM_SAMPLING_T) -> None:
        super().__init__()
        self.min_t = min_t
        self.min_sigma = None

    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        if not self.min_sigma:
            self.min_sigma = sde.transition_kernel_mean_std(
                X=torch.tensor(0.0), t=torch.tensor(self.min_t)
            )[1]

        sigma = sde.transition_kernel_mean_std(X=torch.tensor(0.0), t=torch.tensor(t))[
            1
        ]
        return score + constraint_gradient * self.min_sigma / sigma


class ExpNoiseWeighting(GradientMixer):
    def __init__(self, min_t: float = MINIMUM_SAMPLING_T) -> None:
        super().__init__()
        self.min_sigma = None

    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        if not self.min_sigma:
            self.min_sigma = sde.transition_kernel_mean_std(
                X=torch.tensor(0.0), t=torch.tensor(0)
            )[1]

        sigma = sde.transition_kernel_mean_std(X=torch.tensor(0.0), t=torch.tensor(t))[
            1
        ]
        return score + constraint_gradient * torch.exp(-sigma + self.min_sigma)


class SNRWeighting(GradientMixer):
    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        return score + constraint_gradient * sde.snr(t)


class SNRC(GradientMixer):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = abs(k)

    def __call__(
        self,
        *,
        constraint_gradient: torch.Tensor,
        score: torch.Tensor,
        t: float,
        sde: Sde,
        **kwargs
    ) -> torch.Tensor:
        _, std = sde.transition_kernel_mean_std(X=torch.tensor(0.0), t=torch.tensor(t))
        cap = (self.k / (std**2)).to(device=constraint_gradient.device)
        weighted_constraint = constraint_gradient * sde.snr(t)
        clipped_constraint_grad = torch.clip(weighted_constraint, min=-cap, max=cap)
        return score + clipped_constraint_grad


linear_interpolation = TimeWeighting(g=lambda t: 1 - t)
exponential_interpolation = TimeWeighting(g=lambda t: np.exp(-5.0 * t))
logistic_interpolation = LogisticTimeWeighting(k=10.0)
constant = TimeWeighting(g=lambda t: 1.0, rescale=False)
noise_weighting = NoiseWeighting()
exp_noise_weighting = ExpNoiseWeighting()
snr = SNRWeighting()
