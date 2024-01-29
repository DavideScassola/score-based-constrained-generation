import fnmatch
import os

import torch

from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import noise_weighting, snr
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod

DEFAULT_MODELS_FOLDER = "artifacts/models"
GAP = 0
FILL_DIFFERENCE = 1
LOGIC_K = 80
STRENGTH = 1
GRADIENT_APPROXIMATION = snr
SEED = 1234


# Universal guidance
universal_guidance = dict(
    forward_guidance=False, backward_guidance_steps=0, per_step_self_recurrence_steps=0
)


def mnist_model_path():
    """
    Picks the last stored model
    """
    return f"{DEFAULT_MODELS_FOLDER}/{fnmatch.filter(sorted(os.listdir(DEFAULT_MODELS_FOLDER)), '*mnist*')[-1]}"


""""
def expand_like(input: torch.Tensor, *, like: torch.Tensor):
    return input.reshape(
        tuple(input.shape) + (1,) * (len(like.shape) - len(input.shape))
    ).expand_as(like)

def to01(x: torch.Tensor):
    x = x - expand_like(x.amin(dim=(1, 2)), like=x)
    return x / expand_like(x.amax(dim=(1, 2)), like=x)

def f(x: torch.Tensor):
    return Logic.greater(to01(x).mean(dim=(1,2)), 1.0)
"""


def relative_filling_down(x: torch.Tensor):
    return Logic.greater(
        x[:, : 14 - GAP, :].mean(dim=(1, 2)) - FILL_DIFFERENCE,
        x[:, 14 + GAP :, :].mean(dim=(1, 2)),
    )


constraint = Constraint(
    f=relative_filling_down,
    strength=STRENGTH,
    gradient_mixer=GRADIENT_APPROXIMATION,
    **universal_guidance,
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=64,
    steps=1000,
    corrector_steps=2,
    final_corrector_steps=0,
)

CONFIG = ConstrainedGenerationConfig(
    name="mnist_relative_filling_up",
    constraint=constraint,
    model_path=mnist_model_path(),
    generation_options=generation_options,
    seed=SEED,
)
