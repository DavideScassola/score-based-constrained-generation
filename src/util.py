import _thread
import fnmatch
import importlib.util
import json
import math
import os
import pickle
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, NamedTuple, Tuple

import GPUtil
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import normalized_mutual_info_score
from torch import Tensor
from torch.optim import SGD, RAdam

INT_IS_NUMERICAL_THRESHOLD = 20
DEFAULT_TARGET_SNR = 0.16  # TODO: another hyperparameter that should be settable


def load_json(file: str | Path) -> dict:
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def store_json(d: dict, *, file: str | Path):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


class edit_json:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if os.path.exists(self.filename):
            self.file = open(self.filename, "r+")
            self.data = json.load(self.file)
            # Move the pointer to the beginning of the file
            self.file.seek(0)
        else:
            self.file = open(self.filename, "w")
            self.data = {}
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        json.dump(self.data, self.file, indent=4)
        self.file.close()


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


@contextmanager
def requires_grad(X, *, reset_grad: bool):
    original_state = X.requires_grad
    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    X.requires_grad = True
    if reset_grad:
        X.grad = None
    try:
        yield X
    finally:
        X.requires_grad = original_state
        torch.set_grad_enabled(was_grad_enabled)
        if reset_grad:
            X.grad = None


def gradient(
    *, f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor
) -> torch.Tensor:
    with requires_grad(X, reset_grad=True):
        value = f(X).sum()
        value.backward()
        # TODO: check if gradient flows only with respect to X!
        assert torch.isfinite(value), f(X)
        grad = X.grad
    return grad  # type: ignore


def upscale(image, n=20):
    new_data = np.zeros(np.array(image.shape) * n)

    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            new_data[j * n : (j + 1) * n, k * n : (k + 1) * n] = image[j, k]

    return new_data


def downscale(image, n=20):
    shape = np.array(image.shape) // n
    new_data = np.zeros(shape)

    for j in range(shape[0]):
        for k in range(shape[1]):
            new_data[j, k] = np.mean(image[j * n : (j + 1) * n, k * n : (k + 1) * n])

    return new_data


def find(folder: str, *, pattern: str = "*", index: int = -1):
    """
    Picks the last stored model
    """
    matches = fnmatch.filter(sorted(os.listdir(folder)), pattern)
    if matches:
        return f"{folder}/{matches[index]}"
    else:
        raise ValueError(f"No model found matching {pattern}")


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
    x: np.ndarray | pd.DataFrame,
    acceptance_function,
    max_acceptance: float | None = 1.0,
) -> np.ndarray:
    """acceptance_function is a probability density function"""
    acceptance = acceptance_function(x)
    observed_max_acceptance = np.max(acceptance)
    if not max_acceptance or observed_max_acceptance > max_acceptance:
        max_acceptance = observed_max_acceptance
    acceptance /= max_acceptance
    selection = np.random.binomial(n=1, p=acceptance) == 1
    return x[selection] if isinstance(x, np.ndarray) else x.iloc[selection]


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


def store_array(x: np.ndarray | Tensor, path: Path, name: str) -> None:
    if isinstance(x, Tensor):
        x = x.cpu().numpy()
    np.save(f"{path}/{name}", x)


def logits_sample(
    logits: torch.Tensor, dim: int = -1, temperature: float = 1
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=dim)
    return torch.distributions.Categorical(logits=logits / temperature).sample().long()


def input_listener():
    def input_thread(a_list):
        input()
        a_list.append(True)

    a_list = []
    _thread.start_new_thread(input_thread, (a_list,))
    return a_list


def normalize(x: Tensor, dim):
    return (x - x.mean(dim=dim)) / x.std(dim=dim)


def pickle_load(file_name: str):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def pickle_store(object, *, file: str):
    with open(file, "wb") as f:
        return pickle.dump(object, f)


def bernoulli(p: float) -> bool:
    return random.random() < p


class NamedFunction(NamedTuple):
    f: Callable
    name: str

    def __call__(self) -> Any:
        return self.f()


class BatchSampler:
    def __init__(
        self, iterable, *, batch_size: int, shuffle: bool = True, drop_last: bool = True
    ) -> None:
        self.iterable = iterable
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.i = -1
        self.size = len(iterable)
        self.start = np.arange(0, self.size, self.batch_size)
        self.end = self.start + self.batch_size
        self.end[-1] = min(self.end[-1], self.size)
        if self.drop_last and self.end[-1] - self.start[-1] != self.batch_size:
            self.end = self.end[:-1]
            self.start = self.start[:-1]
        self.num_batches = len(self.start)

    def _shuffle(self):
        if isinstance(self.iterable, Tensor):
            self.iterable = self.iterable[torch.randperm(self.size)]
        elif isinstance(self.iterable, np.ndarray):
            self.iterable = self.iterable[np.random.permutation(self.size)]
        else:
            random.shuffle(self.iterable)

    def __next__(self):
        if self.i == 0 and self.shuffle:
            self._shuffle()
        self.i = (self.i + 1) % self.num_batches
        return self.iterable[self.start[self.i] : self.end[self.i]]

    def __call__(self) -> Any:
        return self.__next__()

    def __getitem__(self, index: int) -> Any:
        return self.iterable[self.start[index] : self.end[index]]

    def reset(self) -> None:
        self.i = -1


def get_available_device(mem_required: float = 0.05, verbose: bool = False):
    if not torch.cuda.is_available():
        return "cpu"

    try:
        devices = GPUtil.getGPUs()

        device_usages = [
            (device.id, device.memoryUsed / device.memoryTotal) for device in devices
        ]

        device_usages.sort(key=lambda x: x[1])

        if device_usages[0][1] > 1 - mem_required:
            return "cpu"

        out = "cuda:" + str(device_usages[0][0])
        if verbose:
            print("\033[92m" + f"Using {out} ... let's g{'o'*69}" + "\033[0m")
        return out
    except ValueError as e:
        print(e)
        return "cpu"


def cross_entropy(*, logits: Tensor, targets: Tensor) -> Tensor:
    """
    Cross entropy loss where the logits dimension is the last one, everything before is batched (differently from base working of torch cross entropy)
    """

    return torch.nn.functional.cross_entropy(
        logits.permute(0, -1, 1) if len(logits.shape) > 2 else logits,
        targets,
        reduction="none",
    )


def batch_function(
    f: Callable,
    *,
    X: Tensor,
    max_batch_size: int = 0,
    verbose_lim: int | None = None,
    verbose_text: str | None = None,
) -> Tensor:
    """
    Apply function f to X in batches of size batch_max_size, still returning a tensor of the expected shape
    """
    n = len(X)
    if max_batch_size <= 0 or n <= max_batch_size:
        return f(X)
    first_output = f(X[0:max_batch_size])
    out = torch.zeros([n] + list(first_output.shape[1:]), dtype=first_output.dtype)
    out[0:max_batch_size] = first_output

    it = range(max_batch_size, n, max_batch_size)
    if verbose_lim and len(it) > verbose_lim:
        it = tqdm.tqdm(it, desc=verbose_text)
    for i in it:
        out[i : i + max_batch_size] = f(X[i : i + max_batch_size])
    return out


def categorical_l1_histogram_distance(x, y) -> float:
    px = pd.Series(x).value_counts(normalize=True)
    py = pd.Series(y).value_counts(normalize=True)
    return abs(px.subtract(py, fill_value=0)).sum() / 2


def normalized_mutual_information_matrix(df: pd.DataFrame):
    # TODO: could be vectorized
    n = len(df.columns)
    nmi_matrix = np.ones((n, n))
    for i in tqdm.tqdm(range(n), desc="Computing NMI matrix"):
        for j in range(i):
            nmi_matrix[i, j] = normalized_mutual_info_score(
                df.iloc[:, i], df.iloc[:, j]
            )
            nmi_matrix[j, i] = nmi_matrix[i, j]
    return pd.DataFrame(nmi_matrix, columns=df.columns, index=df.columns)


def is_int(c: pd.Series) -> bool:
    """
    This function is used to determine if an int column is really numerical or used as categorical.
    Obviously it's not perfect, but it's a good heuristic.
    """
    if c.dtype != int:
        return False
    unique_values = c.nunique()
    range_values = c.max() - c.min() + 1
    # TODO: this is probably not possible and wrong, could cause issues
    return (
        unique_values < range_values or range_values > INT_IS_NUMERICAL_THRESHOLD
    ) and (unique_values > INT_IS_NUMERICAL_THRESHOLD)


def is_numerical(c: pd.Series) -> bool:
    return ("float" in str(c.dtype)) or is_int(c)


def is_categorical(s: pd.Series) -> bool:
    if s.dtype == "object":
        return True
    return not is_numerical(s)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(filter(lambda c: is_categorical(df[c]), df.columns))


def numerical_columns(df: pd.DataFrame) -> list[str]:
    return list(filter(lambda c: is_numerical(df[c]), df.columns))


def only_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[categorical_columns(df)]


def only_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[numerical_columns(df)]


def nan_as_category(
    df: pd.DataFrame, name: str = "MISSING", columns: list[str] | None = None
):
    # df.loc[:,categorical_columns(df)].fillna(name, inplace=True)
    cat_columns = categorical_columns(df) if columns is None else columns
    for col in cat_columns:
        df[col].fillna(name, inplace=True)


class timer:
    def __init__(self, message=None):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = round(self.end - self.start, 2)
        print(
            f"{self.message} - Elapsed time: {self.interval} seconds"
            if self.message
            else f"Elapsed time: {self.interval} seconds"
        )


def langevin_update(
    *,
    x: torch.Tensor,
    score: torch.Tensor,
    target_snr=DEFAULT_TARGET_SNR,
    step_size=None,
):
    noise = torch.randn_like(x)
    if step_size is None:
        alpha = 1
        grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        eps = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x += eps * score + noise * torch.sqrt(2.0 * eps)
    else:
        x += step_size * score + noise * (2.0 * step_size) ** 0.5


def gradient_descent(
    *,
    x: torch.Tensor,
    loss_function,
    steps: int,
    lr: float = 0.1,
    optimizer: str = "RAdam",
):
    with requires_grad(x, reset_grad=True):
        _optimizer = eval(optimizer)(params=[x], lr=lr)
        _optimizer.zero_grad()
        for _ in range(steps):
            _ = loss_function(x).sum().backward()
            _optimizer.step()
            _optimizer.zero_grad()


def normalize_logits(x, maxl: float = 3.0):
    x_centered = x - x.mean(-1, keepdim=True)
    max_logit = abs(x_centered).max(-1, keepdim=True)[0]
    return torch.where(
        (max_logit > maxl).any(-1, keepdim=True),
        maxl * x_centered / max_logit,
        x_centered,
    )


def output_and_grad(*, f, x):
    with requires_grad(x, reset_grad=True):
        output = f(x)
        (grad,) = torch.autograd.grad(output.sum(), x, create_graph=True)
    return output.detach(), grad.detach()
