from dataclasses import dataclass
from pathlib import Path

from src.util import get_available_device, load_module

from .data import Dataset
from .models.model import Model


@dataclass
class TrainConfig:
    dataset: Dataset
    model: Model
    generation_options: dict
    name: str | None = None
    device: str = get_available_device()
    seed: int | None = None


def getConfig(path: str | Path) -> TrainConfig:
    return load_module(str(path)).CONFIG
