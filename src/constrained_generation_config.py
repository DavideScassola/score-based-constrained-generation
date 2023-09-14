from dataclasses import dataclass

from src.constraints.constraint import Constraint
from src.util import load_module


@dataclass
class ConstrainedGenerationConfig:
    constraint: Constraint
    model_path: str
    generation_options: dict
    name: str | None = None
    seed: int | None = None


def getConfig(path: str) -> ConstrainedGenerationConfig:
    return load_module(str(path)).CONFIG
