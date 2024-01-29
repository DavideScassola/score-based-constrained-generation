from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.model import Model
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import ScoreFunction
from src.models.score_based.score_matching import denoising_score_matching_loss
from src.report import score_plot
from src.util import get_available_device, load_json, store_json

from .score_matching import loss_plot, score_matching
from .sdes.sampler import SdeSolver
from .sdes.sde import Sde


class ScoreBasedSde(Model):
    def __init__(
        self,
        *,
        sde: Sde,
        score_function_class,
        score_function_hyperparameters: dict,
        score_matching_loss: Callable = denoising_score_matching_loss,
        optimization: Optimization,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.sde = sde
        self.score_function_class = score_function_class
        self.score_function_hyperparameters = score_function_hyperparameters
        self.optimization = optimization
        self.score_loss_function = score_matching_loss
        self.train_losses = None

    def _train(self, X: torch.Tensor) -> None:
        self.shape = X.data[0].shape
        self.device = get_available_device()

        self.score_model = self.score_function_class(
            shape=self.shape,
            sde=self.sde,
            device=self.device,
            **self.score_function_hyperparameters,
        )

        self.train_losses = score_matching(
            train_set=X,
            score_model=self.score_model,
            optimization=self.optimization,
            sm_loss=self.score_loss_function,
        )

    def _generate(
        self, n_samples: int, sde_solver: SdeSolver, **kwargs
    ) -> torch.Tensor:
        constraint = getattr(self.score_model, "constraint", None)
        per_step_self_recurrence_steps = (
            0 if not constraint else constraint.per_step_self_recurrence_steps
        )

        x = sde_solver.sample(
            sde=self.sde,
            score_function=self.score_model,
            n_samples=n_samples,
            per_step_self_recurrence_steps=per_step_self_recurrence_steps,
            **kwargs,
        )

        return x

    def _store(self, model_path: str) -> None:
        store_json(
            {"shape": self.shape, "device": str(self.device)},
            file=self.params_file(model_path),
        )
        self.score_model.store(model_path)

    def _load_(self, model_path: str) -> None:
        params = load_json(self.params_file(model_path))
        self.shape = params["shape"]
        # self.device = torch.device(params["device"])
        self.device = get_available_device()
        self.score_model = self.score_function_class(
            shape=self.shape,
            sde=self.sde,
            device=self.device,
            **self.score_function_hyperparameters,
        )
        self.score_model.load_(model_path)

    def specific_report_plots(self, path: Path) -> None:
        if self.shape[0] == 1:
            score_plot(score_function=self.score_model, path=path)
        if self.train_losses:
            loss_plot(losses=self.train_losses, path=str(object=path))
