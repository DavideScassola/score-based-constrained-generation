import torch

from .preprocessor import TensorPreprocessor

EPS = 1e-6


class Log(TensorPreprocessor):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + EPS)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - EPS
