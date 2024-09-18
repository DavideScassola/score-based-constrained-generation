import os
from pathlib import Path
from typing import List

import torch

from src.data import Dataset
from src.preprocessors.preprocessor import (TensorPreprocessor,
                                            composed_inverse_transform,
                                            composed_transform)
from src.report import *

from .model import Model


class ImageModel(Model):
    def __init__(self, *, tensor_preprocessors: List[TensorPreprocessor] = []) -> None:
        self.tensor_preprocessors = tensor_preprocessors
        self.preprocessors = self.tensor_preprocessors

    def train(self, dataset: Dataset):
        X = dataset.get(train=True)
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"dataset {dataset} should be a tensor")
        X = composed_transform(X, fit=True, preprocessors=self.tensor_preprocessors)
        self._train(X)

    def generate_report(
        self, *, path: str | Path, dataset: Dataset, generation_options: dict
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        samples = self.generate(**generation_options).cpu()
        images_folder = str(path) + "/samples"
        os.makedirs(images_folder)

        store_images(samples, folder=images_folder)

        plt.imshow(samples[0])
        plt.colorbar()
        plt.savefig(f"image.{IMAGE_FORMAT}")
        plt.close()
