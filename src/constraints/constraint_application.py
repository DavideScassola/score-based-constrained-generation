from typing import Callable, List

import torch

from src.constraints.constraint import Constraint
from src.models.score_based.score_based_sde import ScoreBasedSde
from src.models.score_based.score_function import ScoreFunction
from src.models.score_based.sdes.sde import VE, Sde
from src.preprocessors.preprocessor import (TensorPreprocessor,
                                            composed_inverse_transform)
from src.util import gradient

GRADIENT_CLIPPING = None  # TODO: is clipping OK?
CONSTRAINT_PREDICTION = False  # TODO: experimental


def prediction_based_constrained_score(
    *, sde: Sde, t: float, X: torch.Tensor, score_function, constraint: Constraint
) -> torch.Tensor:
    # EXPERIMENTAL!

    t_tensor = torch.full(
        size=(len(X), 1),
        fill_value=t,
        dtype=torch.float32,
        device=X.device,
    )

    X.requires_grad = True
    X.grad = torch.zeros_like(X)

    score = score_function(
        X=X, t=t_tensor
    ).detach()  # TODO: maybe use the approximated constrained score instead?

    predicted_X0 = sde.denoising(t=t_tensor, X=X, score=score)

    predicted_constraint_value = constraint(predicted_X0)
    predicted_constraint_value.sum().backward()

    # TODO: all gradients are the same!
    constrained_score = constraint.gradient_mixer(
        constraint_gradient=X.grad, score=score.detach(), t=t, sde=sde
    )

    X.grad = None
    X.requires_grad = False

    return constrained_score


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

            if CONSTRAINT_PREDICTION:
                return prediction_based_constrained_score(
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
    )


def apply_constraint_(*, model: ScoreBasedSde, constraint: Constraint) -> ScoreBasedSde:
    model.score_model = get_constrained_score(
        score_function=model.score_model,
        constraint=include_inverse_preprocessing(
            constraint=constraint, preprocessors=model.tensor_preprocessors
        ),
    )
    return model
