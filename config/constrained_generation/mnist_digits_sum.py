import fnmatch
import os

import torch
from torch import Tensor

from mnist_classifier.get_module import mnist_classifier
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod

DEFAULT_MODELS_FOLDER = "artifacts/models"
SCORE_APPROXIMATOR = snr
LOGIC_GRAD = 30.0
SEED = 1234


def mnist_model_path() -> str:
    """
    Picks the last stored model
    """
    return f"{DEFAULT_MODELS_FOLDER}/{fnmatch.filter(sorted(os.listdir(DEFAULT_MODELS_FOLDER)),f'*mnist*')[-1]}"


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    min_values = images.view(images.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
    max_values = images.view(images.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    return (images - min_values) / (max_values - min_values)


classifier = mnist_classifier()

universal_guidance = dict(
    forward_guidance=True, backward_guidance_steps=20, per_step_self_recurrence_steps=2
)


def is_a(logprob: torch.Tensor, label: int) -> Tensor:
    return Logic.equal(logprob[:, label], 0.0, LOGIC_GRAD)


def sum10_prob(logprob1: torch.Tensor, logprob2: torch.Tensor) -> Tensor:
    joint_logprob = logprob1.unsqueeze(-1) + logprob2.unsqueeze(-2)
    prob = torch.exp(joint_logprob)
    return (
        prob[:, 1, 9]
        + prob[:, 2, 8]
        + prob[:, 3, 7]
        + prob[:, 4, 6]
        + prob[:, 5, 5]
        + prob[:, 6, 4]
        + prob[:, 7, 3]
        + prob[:, 8, 2]
        + prob[:, 9, 1]
    )


def f(x: torch.Tensor) -> Tensor:
    logprob = classifier(normalize_images(x.unsqueeze(1)))
    i = torch.arange(0, len(x), 2)
    a = logprob[i]
    b = logprob[i + 1]

    # return Logic.equal(sum10_prob(a,b), 1.0, LOGIC_GRAD)

    return Logic.or_(
        Logic.and_(is_a(a, 1), is_a(b, 9)),
        Logic.and_(is_a(a, 2), is_a(b, 8)),
        Logic.and_(is_a(a, 3), is_a(b, 7)),
        Logic.and_(is_a(a, 4), is_a(b, 6)),
        Logic.and_(is_a(a, 5), is_a(b, 5)),
        Logic.and_(is_a(a, 6), is_a(b, 4)),
        Logic.and_(is_a(a, 7), is_a(b, 3)),
        Logic.and_(is_a(a, 8), is_a(b, 2)),
        Logic.and_(is_a(a, 9), is_a(b, 1)),
    )


constraint = Constraint(f=f, strength=1.0, gradient_mixer=SCORE_APPROXIMATOR, **universal_guidance)  # type: ignore

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=50,
    steps=1000,
    corrector_steps=5,
    final_corrector_steps=10,
)
name = "digits_sum10"
if universal_guidance["forward_guidance"]:
    name = (
        name
        + f"_bw{universal_guidance['backward_guidance_steps']}_rs{universal_guidance['per_step_self_recurrence_steps']}"
    )

CONFIG = ConstrainedGenerationConfig(
    name=name,
    constraint=constraint,
    model_path=mnist_model_path(),
    generation_options=generation_options,
    seed=SEED,
)
