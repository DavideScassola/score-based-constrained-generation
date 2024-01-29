import torch

from .preprocessor import TensorPreprocessor


class HypercubeNormalizer(TensorPreprocessor):
    def fit(self, x: torch.Tensor):
        self.parameters["min"] = x.amin(axis=0)  # type: ignore
        self.parameters["max"] = x.amax(axis=0)  # type: ignore

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self.parameters_to_device(x)
        return (x - self.parameters["min"]) / (
            self.parameters["max"] - self.parameters["min"]
        )

    def parameters_to_device(self, x: torch.Tensor):
        if x.device != self.parameters["min"].device:
            self.parameters["min"] = self.parameters["min"].to(x.device)
            self.parameters["max"] = self.parameters["max"].to(x.device)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.parameters_to_device(x)
        return (
            x * (self.parameters["max"] - self.parameters["min"])
            + self.parameters["min"]
        )

    def serialize(self, p: dict) -> dict:
        return {k: v.tolist() for k, v in p.items()}

    def deserialize(self, p: dict) -> dict:
        return {k: torch.tensor(v) for k, v in p.items()}
