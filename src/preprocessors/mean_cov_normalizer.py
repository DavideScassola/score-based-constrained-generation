import numpy as np
import torch

from .preprocessor import TensorPreprocessor


class MeanCovNormalizer(TensorPreprocessor):
    def fit(self, x: torch.Tensor):
        self.parameters["mean"] = np.mean(x, axis=0)
        self.parameters["costd"] = np.linalg.cholesky(np.cov(x.T)).T

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.parameters["mean"]).dot(
            np.linalg.inv(self.parameters["costd"])
        )

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.dot(self.parameters["costd"]) + self.parameters["mean"]

    def serialize(self, p: dict) -> dict:
        # TODO
        return {k: v.tolist() for k, v in p.items()}

    def deserialize(self, p: dict) -> dict:
        # TODO
        return {k: torch.tensor(v) for k, v in p.items()}
