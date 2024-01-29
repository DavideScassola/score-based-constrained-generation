import fnmatch
import os

import torch
from torch import Tensor

from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod

DEFAULT_MODELS_FOLDER = "artifacts/models"
SCORE_APPROXIMATOR = snr
LOGIC_GRAD = 20.0
SEED = 42

# Universal guidance
universal_guidance = dict(
    forward_guidance=True, backward_guidance_steps=0, per_step_self_recurrence_steps=0
)


def mnist_model_path() -> str:
    """
    Picks the last stored model
    """
    return f"{DEFAULT_MODELS_FOLDER}/{fnmatch.filter(sorted(os.listdir(DEFAULT_MODELS_FOLDER)),f'*mnist*')[-1]}"


def f(x: torch.Tensor) -> Tensor:
    return Logic.equal(x, torch.flip(x, (1,)), grad=LOGIC_GRAD).flatten(1).sum(1)


constraint = Constraint(
    f=f, strength=1.0, gradient_mixer=SCORE_APPROXIMATOR, **universal_guidance  # type: ignore
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=64,
    steps=1000,
    corrector_steps=0,
    final_corrector_steps=0,
)

CONFIG = ConstrainedGenerationConfig(
    name="mnist_horizontal_symmetry",
    constraint=constraint,
    model_path=mnist_model_path(),
    generation_options=generation_options,
    seed=SEED,
)
