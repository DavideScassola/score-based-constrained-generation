from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import Tensor

from src.constants import WEIGHTS_FILE
from src.util import get_available_device

from .nn.time_residual_mlp import TimeResidualMLP
from .nn.unet import MyUNet
from .sdes.sde import VE, Sde


class ScoreFunction(ABC):
    def __init__(
        self,
        *,
        sde: Sde,
        shape: tuple,
        device: str,
        rescale_factor_limit: float | None = 1000.0,
        add_prior_score_bias: bool = False,
        **hyperparameters,
    ) -> None:
        self.shape = shape
        self.sde = sde
        self.device = device
        self.rescale_factor_limit = rescale_factor_limit
        self.add_prior_score_bias = add_prior_score_bias
        self.model = self.build_model(**hyperparameters)
        # print(self.model)

    @abstractmethod
    def build_model(self, **hyperparameters) -> torch.nn.Module:
        pass

    @abstractmethod
    def forward(self, *, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(
        self, *, X: torch.Tensor, t: torch.Tensor | float, grad: bool = False
    ) -> torch.Tensor:
        self.train(False)
        self.model.eval()
        if isinstance(t, float):
            t = torch.full(
                size=(len(X), 1), fill_value=t, dtype=torch.float32, device=X.device
            )
        if grad:
            return self.forward(X=X, t=t)
        else:
            with torch.no_grad():
                out = self.forward(X=X, t=t)
            return out

    def get_shape(self) -> tuple:
        return self.shape

    def parameters(self):
        return self.model.parameters()

    def train(self, train: bool = True):
        self.model.train(train)

    def input_scaler(self, *, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if isinstance(self.sde, VE):
            _, std = self.sde.transition_kernel_mean_std(X=X, t=t)
            return X / torch.sqrt(1.0 + torch.square(std))
        return X

    def store(self, path: str):
        # TODO: using safe tensors would be better
        torch.save(self.model.state_dict(), f"{path}/{WEIGHTS_FILE}.pth")

    def load_(self, path: str):
        # TODO: using safe tensors would be better
        self.model.load_state_dict(
            torch.load(f"{path}/{WEIGHTS_FILE}.pth", map_location=self.device)
        )

    def weighted_score_coefficient(self, *, X, t: torch.Tensor) -> Tensor | float:
        # experimental
        """
        w(t) is defined such that:
        gaussian_score(t=0) * w(t) + (1-w(t)) * gaussian_score(t=1) = gaussian_score(t)

        so that at t = 1, w(t) = 0 and the score starts from the prior
        """
        if isinstance(self.sde, VE):
            mv = self.sde.sigma_max**2
            v = self.sde.transition_kernel_mean_std(X=X, t=t)[1] ** 2
            return (1 - v / mv) / (1 + v)
        else:
            return 1.0

    def prior_score_weighting(self, *, X, t: Tensor, score: Tensor):
        # experimental
        w = self.weighted_score_coefficient(X=X, t=t)
        return w * score + (1 - w) * self.prior_normal_score(X=X)

    def output_scaler(
        self, *, Y: torch.Tensor, X: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if not self.rescale_factor_limit:
            return Y
        _, std = self.sde.transition_kernel_mean_std(X=X, t=t)
        return Y * torch.min(
            1.0 / std,
            torch.full_like(input=std, fill_value=self.rescale_factor_limit),
        )


class MLP(ScoreFunction):
    def build_model(self, **hyperparameters) -> torch.nn.Module:
        numel = int(np.prod(np.array(self.get_shape())))
        return TimeResidualMLP(
            in_channels=numel, out_channels=numel, **hyperparameters
        ).to(self.device)

    def forward(self, *, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scaled_X = self.input_scaler(X=X.flatten(1), t=t)
        output = (self.model.forward(X=scaled_X, t=t)).reshape([-1] + list(self.shape))
        nn_score = self.output_scaler(X=X, Y=output, t=t)
        if self.add_prior_score_bias:
            return nn_score + self.sde.prior_score(X=X, t=t)
        return nn_score


class GineMLP(ScoreFunction):
    def build_model(self, **hyperparameters) -> torch.nn.Module:
        numel = int(np.prod(np.array(self.get_shape())))
        return GMLP(n_features=numel, **hyperparameters).to(self.device)

    def forward(self, *, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scaled_X = self.input_scaler(X=X.flatten(1), t=t)
        output = (self.model.forward(x=scaled_X, t=t)).reshape([-1] + list(self.shape))
        return output  # self.output_scaler(X=X, Y=output, t=t)


class Unet(ScoreFunction):
    def build_model(self, **hyperparameters) -> torch.nn.Module:
        self.device = get_available_device()
        return MyUNet(**hyperparameters).to(self.device)

    def forward(self, *, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scaled_X = self.input_scaler(X=X, t=t)
        output = self.model.forward(scaled_X, t)  # - scaled_X
        return self.output_scaler(X=X, Y=output, t=t)
