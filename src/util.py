import fnmatch
import importlib.util
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable, Tuple

import numpy as np
import torch


def load_json(file: str | Path) -> dict:
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def store_json(d: dict, *, file: str | Path):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def file_name(file: str | Path) -> str:
    return str(file).split("/")[-1]


def load_module(path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("module", path)
    if spec == None:
        raise ValueError(f"{path} is not a valid module path")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def execute_python_script(path: str | Path):
    os.system(f"python {path}")


def create_experiment_folder(*, path: Path, postfix: str | None = None) -> Path:
    postfix = f"_{postfix}" if postfix else ""
    folder_name = Path(datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f") + postfix)
    experiment_folder = path / folder_name
    os.makedirs(experiment_folder)
    return experiment_folder


def gradient(
    *, f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor
) -> torch.Tensor:
    # TODO: Check if all of this boilerplate is necessary
    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    X.requires_grad = True
    X.grad = None

    value = f(X).sum()
    value.backward()
    assert torch.isfinite(value), f(X)
    grad = X.grad

    X.grad = None
    X.requires_grad = False
    torch.set_grad_enabled(was_grad_enabled)
    return grad  # type: ignore


def upscale(image, n=20):
    new_data = np.zeros(np.array(image.shape) * n)

    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            new_data[j * n : (j + 1) * n, k * n : (k + 1) * n] = image[j, k]

    return new_data


def find(folder: str, *, pattern: str = "*", index: int = -1):
    """
    Picks the last mnist stored model
    """
    matches = fnmatch.filter(sorted(os.listdir(folder)), pattern)
    return f"{folder}/{matches[index]}" if matches else None


def D_kl(x: np.ndarray, y: np.ndarray) -> float:
    bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    bins = min(len(x), len(y)) // 20

    def density_estimator(k):
        b = np.histogram(k, density=False, range=bounds, bins=bins)[0] + 1
        return b / np.sum(b)

    px = density_estimator(x)
    py = density_estimator(y)
    return np.sum(np.where(px != 0, px * (np.log(px) - np.log(py)), 0))


def max_symmetric_D_kl(x: np.ndarray, y: np.ndarray) -> float:
    bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    bins = min(len(x), len(y)) // 20

    def density_estimator(k):
        b = (
            np.histogram(k, density=False, range=bounds, bins=bins)[0] + 1
        )  # +1 in order to have numerical stability
        return b / np.sum(b)

    px = density_estimator(x)
    py = density_estimator(y)

    dkl = lambda p1, p2: np.sum(np.where(p1 != 0, p1 * (np.log(p1) - np.log(p2)), 0))
    return max(dkl(px, py), dkl(py, px))


def l1_divergence(x: np.ndarray, y: np.ndarray) -> float:
    bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    bins = int(np.sqrt(min(len(x), len(y))))

    def density_estimator(k):
        b = np.histogram(k, density=False, range=bounds, bins=bins)[0]
        return b / np.sum(b)

    px = density_estimator(x)
    py = density_estimator(y)

    return np.sum(np.abs(px - py)) / 2


def rejection_sampling(
    x: np.ndarray, acceptance_function, max_acceptance: float | None = 1.0
) -> np.ndarray:
    """acceptance_function is a probability density function"""
    acceptance = acceptance_function(x)
    observed_max_acceptance = np.max(acceptance)
    if not max_acceptance or observed_max_acceptance > max_acceptance:
        max_acceptance = observed_max_acceptance
    acceptance /= max_acceptance
    selection = np.random.binomial(n=1, p=acceptance) == 1
    return x[selection]


def reciprocal_distribution_sampler(
    *, low: float, up: float, size: tuple, dtype=None, device=None
) -> torch.Tensor:
    return torch.exp(
        torch.rand(size=size, dtype=dtype, device=device)
        * (math.log(up) - math.log(low))
        + math.log(low)
    )


def uniform_sampler(
    *, low: float, up: float, size: tuple, dtype=None, device=None
) -> torch.Tensor:
    return torch.rand(size=size, dtype=dtype, device=device) * (up - low) + low


def names_index_map(file) -> dict[str, int]:
    with open(file, "r") as f:
        names = (f.readline()[:-1]).split(",")
    return {n: i for (i, n) in enumerate(names)}


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
