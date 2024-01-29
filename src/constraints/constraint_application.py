from typing import List

import torch
from torch.optim import RAdam

from src.constants import MINIMUM_SAMPLING_T
from src.constraints.constraint import Constraint
from src.models.score_based.score_based_sde import ScoreBasedSde
from src.models.score_based.score_function import ScoreFunction
from src.models.score_based.sdes.sampler import langevin_dynamics
from src.models.score_based.sdes.sde import Sde
from src.preprocessors.preprocessor import (TensorPreprocessor,
                                            composed_inverse_transform)
from src.util import gradient, requires_grad

GRADIENT_CLIPPING = None  # TODO: is clipping OK?
BG_LR = 1e-2
"""
def opt_update(*, x: torch.Tensor, grad: torch.Tensor):
    rate = 0.1  # TODO: another hyperparameter that should be settable
    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
    eps = rate / grad_norm
    x += eps * grad


def gradient_ascent(*, x: torch.Tensor, grad_function, steps: int):
    for _ in range(steps):
        opt_update(x=x, grad=grad_function(X=x))
"""


def gradient_descent(*, x: torch.Tensor, loss_function, steps: int):
    with requires_grad(x, reset_grad=True):
        optimizer = RAdam(params=[x], lr=BG_LR)
        optimizer.zero_grad()
        for _ in range(steps):
            _ = loss_function(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()


def backward_guidance_score(
    *,
    x0_start: torch.Tensor,
    sde: Sde,
    t: torch.Tensor,
    X: torch.Tensor,
    score_function,
    constraint: Constraint
):
    """
    EXPERIMENTAL
    """

    """
    def constrained_score_function_at0(*, X, t: torch.Tensor):
        return (
            score_function(t=t, X=X)
            + gradient(f=constraint.f, X=X) * constraint.strength
        )

    def constraint_grad(*, X):
        return gradient(f=constraint.f, X=X) * constraint.strength

    if False:
        langevin_dynamics(
            x=x0_start,
            t=0.0,
            score_function=constrained_score_function_at0,
            steps=BACKWARD_GUIDANCE_STEPS,
        )
    else:
    """

    if constraint.backward_guidance_steps:
        gradient_descent(
            x=x0_start,
            loss_function=lambda x: -constraint(x),
            steps=constraint.backward_guidance_steps,
        )

    return sde.score_from_denoising(t=t, X=X, x0=x0_start)


def forward_guidance_score(
    *, sde: Sde, t: float, X: torch.Tensor, score_function, constraint: Constraint
) -> torch.Tensor:
    t_tensor = torch.full(
        size=(len(X), 1),
        fill_value=t,
        dtype=torch.float32,
        device=X.device,
    )

    with requires_grad(X, reset_grad=True):
        score = score_function(X=X, t=t_tensor, grad=True)

        predicted_X0 = sde.denoising(t=t_tensor, X=X, score=score)

        value = constraint(predicted_X0).sum()
        value.backward()

        constrained_score = constraint.gradient_mixer(
            constraint_gradient=X.grad,
            score=score.detach(),
            t=t,
            sde=sde,
            constraint=constraint,
        )

    return constrained_score


def universal_guidance_score(
    *, sde: Sde, t: float, X: torch.Tensor, score_function, constraint: Constraint
) -> torch.Tensor:
    t_tensor = torch.full(
        size=(len(X), 1),
        fill_value=t,
        dtype=torch.float32,
        device=X.device,
    )

    modified_score = forward_guidance_score(
        sde=sde, t=t, X=X, score_function=score_function, constraint=constraint
    )

    if constraint.backward_guidance_steps:
        modified_score = backward_guidance_score(
            sde=sde,
            t=t_tensor,
            X=X,
            x0_start=sde.denoising(t=t_tensor, X=X, score=modified_score),
            score_function=score_function,
            constraint=constraint,
        )

    return modified_score


def theoretical_score_limit(*, X: torch.Tensor, sde: Sde, t: float):
    """
    Let's consider inpainting for one dimension x, where the value h is imposed.
    x_t ~ N(mu(h, t), sigma(h, t)) where mu() and sigma() depend on the sde.
    Then the score of x_t is s_inp(x_t) = ( mu(h, t) - x_t ) / sigma(h, t)**2.
    |s_inp(x_t)| can be considered the limit in the value of the score, since inpainting
    is the strongest constraint. h can be an extreme value but in the gaussian range (for example +3 or -3)
    """
    # EXPERIMENTAL!
    EXTREME = 3.0
    t_tensor = torch.full((len(X),), t)
    mean1, std1 = sde.transition_kernel_mean_std(
        X=torch.full_like(X, EXTREME), t=t_tensor
    )
    mean2, std2 = sde.transition_kernel_mean_std(
        X=torch.full_like(X, -EXTREME), t=t_tensor
    )
    score1 = (mean1 - X) / std1**2
    score2 = (mean2 - X) / std2**2
    limit = torch.maximum(score1.abs(), score2.abs())
    return limit


def get_constrained_score(
    score_function: ScoreFunction, constraint: Constraint
) -> ScoreFunction:
    class ConstrainedScoreFunction(ScoreFunction):
        def __init__(
            self, *, original_score_function: ScoreFunction, constraint: Constraint
        ) -> None:
            self.original_score_function = original_score_function
            super().__init__(
                sde=self.original_score_function.sde,
                shape=self.original_score_function.shape,
                device=self.original_score_function.device,
            )
            self.constraint = constraint

            # TODO: check if appropriate
            for param in self.parameters():
                param.requires_grad = False

            self.train(False)
            self.model.eval()

        def forward(self, *, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.original_score_function.forward(X=X, t=t)

        def build_model(self, **hyperparameters) -> torch.nn.Module:
            return self.original_score_function.model

        def noised_constraint_score_estimate(
            self, *, X: torch.Tensor, t: float, n: int
        ):
            mean, std = self.sde.transition_kernel_mean_std(
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

        def __call__(self, *, X: torch.Tensor, t: float) -> torch.Tensor:
            """
            p_constrained(x, t) = p(x, t)*exp(lambda*f(x, t))

            score(p_constrained(x, t)) =
            grad{log[p(x, t)*exp(lambda*f(x, t))]} =
            grad{logp(x, t) + lambda*f(x, t)]} =
            grad{logp(x, t) + lambda*f(x, t)]} =
            grad{logp(x, t)} + lambda*grad{f(x, t)} =
            s(x, t) + lambda*grad{f(x, t)}
            """

            # TODO: osserva come score dei dati cambia col tempo
            # TODO: applica riscalo dello score dei dati al gradiente del vincolo
            if self.constraint is None:
                return super().__call__(X=X, t=t)

            if constraint.forward_guidance and t > MINIMUM_SAMPLING_T:
                return universal_guidance_score(
                    sde=self.sde,
                    t=t,
                    X=X,
                    score_function=super().__call__,
                    constraint=self.constraint,
                )

            else:
                constraint_gradient = (
                    gradient(f=self.constraint.f, X=X) * self.constraint.strength
                )
                if GRADIENT_CLIPPING:
                    constraint_gradient = torch.clip(
                        gradient(f=self.constraint.f, X=X) * self.constraint.strength,
                        min=-GRADIENT_CLIPPING,
                        max=GRADIENT_CLIPPING,
                    )

                return self.constraint.gradient_mixer(
                    score=super().__call__(X=X, t=t),
                    constraint_gradient=constraint_gradient,
                    t=t,
                    sde=self.sde,
                    X=X,
                    constraint=self.constraint,
                )

    return ConstrainedScoreFunction(
        original_score_function=score_function, constraint=constraint
    )


def include_inverse_preprocessing(
    *, constraint: Constraint, preprocessors: List[TensorPreprocessor]
) -> Constraint:
    return Constraint(
        f=lambda x: constraint.f(
            composed_inverse_transform(x, preprocessors=preprocessors)
        ),
        strength=constraint.strength,
        gradient_mixer=constraint.gradient_mixer,
        backward_guidance_steps=constraint.backward_guidance_steps,
        forward_guidance=constraint.forward_guidance,
        per_step_self_recurrence_steps=constraint.per_step_self_recurrence_steps,
    )


def apply_constraint_(*, model: ScoreBasedSde, constraint: Constraint) -> ScoreBasedSde:
    model.score_model = get_constrained_score(
        score_function=model.score_model,
        constraint=include_inverse_preprocessing(
            constraint=constraint, preprocessors=model.tensor_preprocessors
        ),
    )
    return model
