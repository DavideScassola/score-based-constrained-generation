from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from src.constants import MINIMUM_SAMPLING_T

from ..score_function import ScoreFunction
from .sde import Sde

# TODO: try https://github.com/rtqichen/torchdiffeq or https://github.com/google-research/torchsde
# TODO: try scipy ode solvers
# TODO: try https://www.wikiwand.com/en/Stochastic_differential_equation

DEFAULT_TARGET_SNR = 0.16  # TODO: another hyperparameter that should be settable
DEFAULT_CLEANING_SNR = 0.16


def langevin_update(
    *, x: torch.Tensor, score: torch.Tensor, target_snr=DEFAULT_TARGET_SNR
):
    # TODO: it doesn't work properly
    alpha = 1  # TODO: this works only for VE sde
    noise = torch.randn_like(x)
    grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
    noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
    eps = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
    x += eps * score + noise * torch.sqrt(2.0 * eps)


def langevin_dynamics(
    *,
    x: torch.Tensor,
    t: float,
    score_function: ScoreFunction,
    steps: int,
    snr=DEFAULT_CLEANING_SNR,
):
    it = range(steps) if steps < 1000 else tqdm(range(steps), desc="Langevin cleaning")
    for _ in it:
        langevin_update(x=x, score=score_function(X=x, t=t))


class SdeSolver:
    def dx(
        self, *, x: torch.Tensor, sde: Sde, score: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor | None:
        NotImplementedError("This solver doesn't explicitly define a formula for dx")

    def update(
        self, *, x: torch.Tensor, sde: Sde, score: torch.Tensor, t: float, dt: float
    ):
        x += self.dx(sde=sde, score=score, t=t, x=x, dt=dt)

    def noise_reinjection(
        self, *, x: torch.Tensor, sde: Sde, t: float, dt: float
    ) -> None:
        x = sde.sde_step(X=x, t=t, dt=dt)

    def solve(
        self,
        *,
        sde: Sde,
        score_function: ScoreFunction,
        X_0: torch.Tensor,
        steps: int = 1000,
        T: float = 1.0,
        corrector_steps: int = 0,
        final_corrector_steps: int = 0,
        final_corrector_snr: float = DEFAULT_CLEANING_SNR,
        per_step_self_recurrence_steps: int = 0,
    ) -> torch.Tensor:
        x = X_0.clone()
        dt = T / steps
        for t in tqdm(
            np.linspace(T, MINIMUM_SAMPLING_T, steps), desc="Reversing the SDE"
        ):
            langevin_dynamics(
                x=x, t=t, score_function=score_function, steps=corrector_steps
            )
            self.update(x=x, sde=sde, score=score_function(X=x, t=t), t=t, dt=dt)
            if t - dt > MINIMUM_SAMPLING_T:
                for _ in range(per_step_self_recurrence_steps):
                    self.noise_reinjection(x=x, sde=sde, t=t - dt, dt=dt)
                    self.update(
                        x=x, sde=sde, score=score_function(X=x, t=t), t=t, dt=dt
                    )

        if final_corrector_steps:
            langevin_dynamics(
                x=x,
                t=MINIMUM_SAMPLING_T,
                score_function=score_function,
                steps=final_corrector_steps,
                snr=final_corrector_snr,
            )
        return x

    def sample(
        self,
        *,
        sde: Sde,
        score_function: ScoreFunction,
        steps: int = 1000,
        T: float = 1.0,
        n_samples: int,
        corrector_steps: int = 0,
        final_corrector_steps: int = 0,
        final_corrector_snr: float = DEFAULT_CLEANING_SNR,
        per_step_self_recurrence_steps: int = 0,
    ) -> torch.Tensor:
        X_0 = sde.prior_sampling(
            shape=score_function.get_shape(), n_samples=n_samples
        ).to(score_function.device)
        return self.solve(
            X_0=X_0,
            score_function=score_function,
            sde=sde,
            T=T,
            steps=steps,
            corrector_steps=corrector_steps,
            final_corrector_steps=final_corrector_steps,
            per_step_self_recurrence_steps=per_step_self_recurrence_steps,
            final_corrector_snr=final_corrector_snr,
        )


def dw(*, dt: float, shape: tuple, device):
    return torch.randn(size=shape, dtype=torch.float32, device=device) * (dt**0.5)


class EulerMethod(SdeSolver):
    def dx(
        self, *, x: torch.Tensor, sde: Sde, score: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        return -sde.reverse_f(X=x, t=t, score=score) * dt + sde.g(t=t) * dw(
            dt=dt, shape=x.shape, device=x.device
        )


class TimeAdaptiveEuler(EulerMethod):
    def __init__(self, *, dx_max: float = 0.01, dt_max: float = 0.01) -> None:
        super().__init__()
        self.dx_max = dx_max
        self.dt_max = dt_max

    def adaptive_dt(self, *, gradient: torch.Tensor) -> float:
        return self.dx_max / torch.max(gradient).item()

    def solve(
        self,
        *,
        sde: Sde,
        score_function: ScoreFunction,
        X_0: torch.Tensor,
        T: float = 1.0,
        corrector_steps: int = 0,
    ) -> torch.Tensor:
        x = X_0.clone()

        print("Reversing the SDE...")

        t = float(T)
        pbar = tqdm(total=T)
        while t > MINIMUM_SAMPLING_T:
            # print(f"{t/T:.1%}")
            score = score_function(X=x, t=t)
            dt = min(self.dt_max, self.adaptive_dt(gradient=score))

            if t - dt < MINIMUM_SAMPLING_T:
                dt = t - MINIMUM_SAMPLING_T

            for _ in range(corrector_steps):
                langevin_update(x=x, score=score)
            self.update(x=x, sde=sde, score=score, t=t, dt=dt)

            t -= dt
            pbar.update(dt)

        pbar.close()

        return x

    def sample(
        self,
        *,
        sde: Sde,
        score_function: ScoreFunction,
        dx_max: float = 0.01,
        dt_max: float = 0.01,
        T: float = 1.0,
        n_samples: int,
        corrector_steps: int = 0,
    ) -> torch.Tensor:
        X_0 = sde.prior_sampling(
            shape=score_function.get_shape(), n_samples=n_samples
        ).to(score_function.device)
        return self.solve(
            X_0=X_0,
            score_function=score_function,
            sde=sde,
            T=T,
            corrector_steps=corrector_steps,
        )
