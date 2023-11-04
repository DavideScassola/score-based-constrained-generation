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
LOGIC_GRAD = 5.0
SEED = 42
MODEL = "2000"


def mnist_model_path() -> str:
    """
    Picks the last stored model
    """
    return f"{DEFAULT_MODELS_FOLDER}/{fnmatch.filter(sorted(os.listdir(DEFAULT_MODELS_FOLDER)),f'*mnist*{MODEL}*')[-1]}"


classifier = mnist_classifier()


def f(x: torch.Tensor) -> Tensor:
    logits = classifier(x.unsqueeze(1))
    probs = torch.softmax(logits, dim=1)
    probs = probs.reshape((probs.shape[0], 2, -1))
    a = probs[:, 0]
    b = probs[:, 1]

    return Logic.or_(
        Logic.and_(Logic.equal(a[1 - 1], 1), Logic.equal(b[9 - 1], 1)),
        Logic.and_(Logic.equal(a[2 - 1], 1), Logic.equal(b[8 - 1], 1)),
        Logic.and_(Logic.equal(a[3 - 1], 1), Logic.equal(b[7 - 1], 1)),
        Logic.and_(Logic.equal(a[4 - 1], 1), Logic.equal(b[6 - 1], 1)),
        Logic.and_(Logic.equal(a[5 - 1], 1), Logic.equal(b[5 - 1], 1)),
        Logic.and_(Logic.equal(a[6 - 1], 1), Logic.equal(b[4 - 1], 1)),
        Logic.and_(Logic.equal(a[7 - 1], 1), Logic.equal(b[3 - 1], 1)),
        Logic.and_(Logic.equal(a[8 - 1], 1), Logic.equal(b[2 - 1], 1)),
        Logic.and_(Logic.equal(a[9 - 1], 1), Logic.equal(b[1 - 1], 1)),
    )


def f(x: torch.Tensor) -> Tensor:
    # batch = x.reshape((-1, 2, 28, 28))
    torch.softmax(classifier(x.unsqueeze(1)), dim=1)
    return Logic.equal(x, torch.flip(x, (1,)), grad=LOGIC_GRAD).flatten(1).sum(1)


constraint = Constraint(f=f, strength=1.0, gradient_mixer=SCORE_APPROXIMATOR)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=64,
    steps=1000,
    corrector_steps=1,
    final_corrector_steps=0,
)

CONFIG = ConstrainedGenerationConfig(
    name="mnist_horizontal_symmetry",
    constraint=constraint,
    model_path=mnist_model_path(),
    generation_options=generation_options,
    seed=SEED,
)
