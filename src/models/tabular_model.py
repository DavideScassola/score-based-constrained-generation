import os
from pathlib import Path
from typing import List

import pandas as pd
import torch

from src.data import Dataset
from src.preprocessors.preprocessor import (TensorPreprocessor,
                                            composed_inverse_transform)
from src.preprocessors.to_tensor import ToTensor
from src.report import *

from .model import Model


class TabularModel(Model):
    def __init__(
        self,
        *,
        tensor_preprocessors: List[TensorPreprocessor] = [],
        df2tensor=ToTensor()
    ) -> None:
        super().__init__()
        self.tensor_preprocessors = tensor_preprocessors
        self.df2tensor = df2tensor
        self.preprocessors = [self.df2tensor] + self.tensor_preprocessors

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        samples = self._generate(n_samples, **kwargs)
        return self.df2tensor.reverse_transform(
            composed_inverse_transform(samples, preprocessors=self.tensor_preprocessors)
        )

    def generate_report(
        self,
        *,
        path: str | Path,
        dataset: Dataset,
        generation_options: dict,
        constraint: Constraint | None = None
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        samples = self.generate(**generation_options)

        df_train = dataset.get(train=True)

        store_samples(df_generated=samples, path=report_folder)

        print("plotting...")

        """
        for comparison in (
            histograms_comparison,
            statistics_comparison,
            correlations_comparison,
        ):
            comparison(
                df_generated=samples,
                df_train=df_train,
                path=report_folder,
                model=self,
            )
        """
        if len(samples.shape) == 2 and samples.shape[1] == 2:
            coordinates_comparison(
                df_generated=samples,
                df_train=df_train,
                path=report_folder,
                model=self,
            )

        self.specific_report_plots(report_folder)
