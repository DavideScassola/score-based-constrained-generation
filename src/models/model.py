import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch

from src.constraints.constraint import Constraint
from src.data import Dataset
from src.preprocessors.preprocessor import (Preprocessor,
                                            composed_inverse_transform,
                                            composed_transform)
from src.util import get_available_device


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

    def get_preprocessed_data(
        self,
        *,
        dataset: Dataset | None = None,
        train: bool = True,
        fit: bool | None = None,
        device: str | torch.device,
    ) -> torch.Tensor:
        if fit is None:
            fit = train
        if not hasattr(self, "dataset"):
            self.dataset = dataset
        return composed_transform(
            self.dataset.get(train=train), preprocessors=self.preprocessors, fit=fit  # type: ignore
        ).to(device)

    def train(self, dataset: Dataset, *, device):
        self._train(
            self.get_preprocessed_data(
                dataset=dataset, train=True, fit=True, device=device
            )
        )

    def generate(self, n_samples: int, **kwargs) -> torch.Tensor:
        samples = self._generate(n_samples, **kwargs)
        return composed_inverse_transform(samples, preprocessors=self.preprocessors)

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
        self,
        *,
        path: str | Path,
        dataset: Dataset,
        generation_options: dict,
        constraint: Constraint | None = None,
        **kwargs,
    ):
        pass
