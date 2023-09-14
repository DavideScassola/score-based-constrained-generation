import os
from pathlib import Path
from typing import List

import torch

from src.data import Dataset
from src.preprocessors.preprocessor import (TensorPreprocessor,
                                            composed_fit_transform,
                                            composed_inverse_transform)
from src.report import *

from .model import Model


class TimeSeriesModel(Model):
    def __init__(self, *, tensor_preprocessors: List[TensorPreprocessor] = []) -> None:
        self.tensor_preprocessors = tensor_preprocessors
        self.preprocessors = self.tensor_preprocessors

    def train(self, dataset: Dataset):
        X = dataset.get(train=True)
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"dataset {dataset} should be a tensor")
        X = composed_fit_transform(X, preprocessors=self.tensor_preprocessors)
        self._train(X)

    def generate(self, n_samples: int, **kwargs) -> torch.Tensor:
        samples = self._generate(n_samples, **kwargs)
        return composed_inverse_transform(
            samples, preprocessors=self.tensor_preprocessors
        )

    def generate_report(
        self, *, path: str | Path, dataset: Dataset, generation_options: dict
    ) -> None:
        raise NotImplementedError()
