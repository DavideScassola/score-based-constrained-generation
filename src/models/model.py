import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
import torch

from src.data import Dataset
from src.preprocessors.preprocessor import Preprocessor


class Model(ABC):
    MODEL_FOLDER_NAME = Path("model")
    PARAMS_FILE = Path("params.json")

    def __init__(self, *, preprocessors: List[Preprocessor] = []) -> None:
        super().__init__()
        self.preprocessors = preprocessors

    def store(self, experiment_path: str | Path):
        model_path = self.get_model_folder(experiment_path)
        os.makedirs(model_path)
        self.store_preprocessors(model_path)
        self._store(model_path)

    def load_(self, experiment_path: str | Path):
        model_path = self.get_model_folder(experiment_path)
        self.load_preprocessors(model_path)
        self._load_(model_path)

    def get_model_folder(self, experiment_path: str | Path) -> Path:
        return Path(experiment_path) / self.MODEL_FOLDER_NAME

    def params_file(self, model_path: str | Path):
        return Path(model_path) / self.PARAMS_FILE

    def store_preprocessors(self, model_path: str | Path):
        for p in self.preprocessors:
            p.store(model_path)

    def load_preprocessors(self, model_path: str | Path):
        for p in self.preprocessors:
            p.load_(model_path)

    def specific_report_plots(self, path: Path):
        pass

    @abstractmethod
    def train(self, dataset: Dataset):
        pass

    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def _train(self, X: torch.Tensor):
        pass

    @abstractmethod
    def _generate(self, n_samples: int) -> torch.Tensor:
        pass

    @abstractmethod
    def _store(self, model_path: str | Path):
        pass

    @abstractmethod
    def _load_(self, model_path: str | Path):
        pass

    @abstractmethod
    def generate_report(
        self, *, path: str | Path, dataset: Dataset, generation_options: dict, **kwargs
    ):
        pass
