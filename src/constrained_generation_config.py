from dataclasses import dataclass

from src.constraints.constraint import Constraint
from src.util import get_available_device, load_module


@dataclass
class ConstrainedGenerationConfig:
    constraint: Constraint
    model_path: str
    generation_options: dict
    name: str | None = None
    seed: int | None = None
    device: str = get_available_device()


def getConfig(path: str) -> ConstrainedGenerationConfig:
    return load_module(str(path)).CONFIG
