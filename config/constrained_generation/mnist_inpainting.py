import fnmatch
import os

import numpy as np
import torch
from matplotlib import image

from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import *
from src.constraints.real_logic import Product as Logic
from src.data import get_MNIST
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import upscale

DEFAULT_MODELS_FOLDER = "artifacts/models"
GRADIENT_APPROXIMATION = linear_interpolation
SEED = 1234
MODEL = "2000"


def mnist_model_path():
    return f"{DEFAULT_MODELS_FOLDER}/{fnmatch.filter(sorted(os.listdir(DEFAULT_MODELS_FOLDER)), f'*mnist*{MODEL}*')[-1]}"


def expand_like(input: torch.Tensor, *, like: torch.Tensor):
    return input.reshape(
        tuple(input.shape) + (1,) * (len(like.shape) - len(input.shape))
    ).expand_as(like)


def to01(x: torch.Tensor):
    x = x - expand_like(x.amin(dim=(1, 2)), like=x)
    return x / expand_like(x.amax(dim=(1, 2)), like=x)


def get_inpainting_constraint_function(
    *,
    image_index: int,
    start_row: int | None = None,
    end_row: int | None = None,
    start_col: int | None = None,
    end_col: int | None = None,
):
    original = get_MNIST(train=False)[image_index]
    rows = slice(start_row, end_row)
    cols = slice(start_col, end_col)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_masked = original.detach().clone().unsqueeze(0)
    original_masked_cuda = original.to(device)
    mask = torch.ones_like(original_masked)
    mask[:, rows, cols] = 0.0
    mask_cuda = torch.ones_like(original_masked_cuda)
    mask_cuda[rows, cols] = 0.0
    original_masked *= mask
    original_masked_cuda *= mask_cuda

    image.imsave(f"inpaint_original.png", upscale(original.numpy()), cmap="gray")
    image.imsave(
        f"inpaint_masked.png",
        upscale(np.where(mask[0], original.numpy(), 2.5)),
        cmap="gray",
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        m = original_masked if x.get_device() else original_masked_cuda
        msk = mask if x.get_device() else mask_cuda
        return Logic.equal((m - x) * msk, 0, grad=10).sum(dim=(1, 2))
        # return Logic.and_(Logic.equal((m - x) * msk, 0, grad=1000))

    return f


constraint = Constraint(
    f=get_inpainting_constraint_function(
        image_index=1, start_row=7, end_row=20, start_col=7, end_col=None
    ),
    strength=1,
    gradient_mixer=GRADIENT_APPROXIMATION,
)


generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=64,
    steps=1000,
    corrector_steps=0,
    final_corrector_steps=0,
)

CONFIG = ConstrainedGenerationConfig(
    name="mnist_inpainting",
    constraint=constraint,
    model_path=mnist_model_path(),
    generation_options=generation_options,
    seed=SEED,
)
