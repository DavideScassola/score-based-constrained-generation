import torch

from .preprocessor import TensorPreprocessor


class Discretizer(TensorPreprocessor):
    # TODO: check gradient behavior

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)
