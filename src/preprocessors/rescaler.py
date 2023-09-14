import torch

from .preprocessor import TensorPreprocessor


class Rescaler(TensorPreprocessor):
    def __init__(
        self,
        *,
        subtract: float = 0.0,
        add: float = 0.0,
        divide_by: float = 1.0,
        multiply_by: float = 1.0
    ):
        super().__init__()
        self.parameters["subtract"] = subtract - add
        self.parameters["divide_by"] = divide_by
        self.parameters["multiply_by"] = multiply_by

    def fit(self, x: torch.Tensor):
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.parameters["multiply_by"]
            * (x - self.parameters["subtract"])
            / self.parameters["divide_by"]
        )

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x * self.parameters["divide_by"] / self.parameters["multiply_by"]
            + self.parameters["subtract"]
        )

    def serialize(self, p: dict) -> dict:
        return {k: v for k, v in p.items()}

    def deserialize(self, p: dict) -> dict:
        return {k: v for k, v in p.items()}
