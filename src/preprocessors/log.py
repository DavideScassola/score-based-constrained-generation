import torch

from .preprocessor import TensorPreprocessor


class Log(TensorPreprocessor):
    def __init__(self, target_features, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset
        self.target_features = target_features

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, self.target_features] = torch.log(
            x[:, self.target_features] + self.offset
        )
        return out

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, self.target_features] = (
            torch.exp(x[:, self.target_features]) - self.offset
        )
        return out
