from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def expand_like(input: Tensor, *, like: Tensor):
    return (
        input.reshape(tuple(input.shape) + (1,) * (len(like.shape) - len(input.shape)))
        .expand_as(like)
        .to(like.device)
    )


def dw(dt: float, like: Tensor) -> Tensor:
    return torch.randn_like(like) * (dt**0.5)


class Sde(ABC):
    """
    dx = f(x,t)dt + g(t)dw
    dw = eps * dt**0.5
    eps ~ N(0,1)
    """

    # TODO: in the paper T is a property of the SDE. T is probably such that
    # X_T ~ prior, but why not fixing T=1 and defining the sde such that x_t=1
    # ~ pior ?

    @abstractmethod
    def f(self, *, X: Tensor, t: float | Tensor) -> Tensor:
        """
        drift coefficient f(x,t)
        """
        pass

    @abstractmethod
    def g(self, *, t: float | Tensor) -> float | Tensor:
        """
        diffusion coefficient g(t)
        """
        pass

    @abstractmethod
    def prior_sampling(self, *, shape: tuple, n_samples: int) -> Tensor:
        pass

    @abstractmethod
    def snr(self, t: float) -> Tensor:
        pass

    @abstractmethod
    def denoising(self, *, t: Tensor, score: Tensor, X: Tensor) -> Tensor:
        """
        remember this:
            score(xt, t) = (mean[sde(t, x0)] - xt) / std[sde(t, x0)]**2

        then:
            mean[sde(t, x0)] = score(xt, t) * std[sde(t, x0)]**2 + xt
        """
        pass

    def score_from_denoising(self, *, t: Tensor, X: Tensor, x0: Tensor) -> Tensor:
        """
        remember this:
            score(xt, t) = (mean[sde(t, x0)] - xt) / std[sde(t, x0)]**2
        """
        mean_xt, std_xt = self.transition_kernel_mean_std(X=x0, t=t)
        return (mean_xt - X) / (std_xt**2)

    def transition_kernel_mean_std(
        self, *, X: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "This sde has not an explicit gaussian transition kernel"
        )

    def sde_step(self, *, X: Tensor, t: float, dt: float) -> Tensor:
        """
        Samples from the transition kernel of the SDE: q(X_t+1 | X_t)
        """
        dx = self.f(X=X, t=t) * dt + self.g(t=t) * dw(dt, like=X)
        return X + dx

    def reverse_f(self, *, X: Tensor, t: float, score: Tensor) -> Tensor:
        return self.f(X=X, t=t) - ((self.g(t=t)) ** 2) * score

    @abstractmethod
    def prior_score(self, *, X: Tensor, t: Tensor) -> Tensor:
        pass


class VE(Sde):
    def __init__(self, *, sigma_min=0.01, sigma_max=50.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def f(self, *, X: Tensor, t: float | Tensor) -> Tensor:
        return torch.zeros_like(X)

    def g(self, *, t: float | Tensor) -> float | Tensor:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma * np.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))

    def transition_kernel_mean_std(
        self, *, X: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return X, expand_like(
            self.sigma_min * (self.sigma_max / self.sigma_min) ** t, like=X
        )

    def prior_sampling(self, shape: tuple, n_samples: int) -> Tensor:
        return torch.randn(size=[n_samples] + list(shape)) * self.sigma_max

    def snr(self, t: float) -> Tensor:
        std = self.transition_kernel_mean_std(X=torch.tensor(0.0), t=torch.tensor(t))[1]
        return (1 + std**2) ** (-0.5)

    def denoising(self, *, t: Tensor, score: Tensor, X: Tensor) -> Tensor:
        std = self.transition_kernel_mean_std(X=X, t=t)[1]
        return X + score * (std**2)

    def prior_score(self, *, X: Tensor, t: Tensor) -> Tensor:
        return -X / (1 + self.sigma_max**2)


class subVP(Sde):
    def __init__(self, *, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def f(self, *, X: Tensor, t: float | Tensor) -> Tensor:
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -0.5 * beta_t * X

    def g(self, *, t: float | Tensor) -> float | Tensor:
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        if isinstance(t, Tensor):
            discount = 1.0 - torch.exp(
                -2.0 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
            )
            return torch.sqrt(beta_t * discount)
        else:
            discount = 1.0 - np.exp(
                -2.0 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
            )
            return np.sqrt(beta_t * discount)

    def transition_kernel_mean_std(
        self, *, X: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        log_mean_coeff = expand_like(
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0, like=X
        )
        mean = torch.exp(log_mean_coeff) * X
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape: tuple, n_samples: int) -> Tensor:
        return torch.randn(size=[n_samples] + list(shape))

    def snr(self, t: float) -> Tensor:
        log_mean_coeff = torch.tensor(
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        k = torch.exp(2 * log_mean_coeff)
        std = 1 - k
        return k / (k**2 + std**2) ** 0.5

    def denoising(self, *, t: Tensor, score: Tensor, X: Tensor) -> Tensor:
        log_mean_coeff = expand_like(
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0, like=X
        )
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return (X + score * (std**2)) / torch.exp(log_mean_coeff)

    def prior_score(self, *, X: Tensor, t: Tensor) -> Tensor:
        return -X
