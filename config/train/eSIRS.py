from functools import partial

import torch

from src.data import Dataset
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import MLP
from src.models.score_based.score_matching import \
    accumulated_denoising_score_matching_loss
from src.models.score_based.sdes.sampler import EulerMethod
from src.models.score_based.sdes.sde import VE, subVP
from src.models.score_based.time_series_score_based_sde import \
    TimeSeriesScoreBasedSde
from src.preprocessors.mean_std_normalizer import MeanStdNormalizer
from src.train_config import TrainConfig

dataset = Dataset(path="data/eSIRS.npy", train_proportion=1.0)

model = TimeSeriesScoreBasedSde(
    tensor_preprocessors=[MeanStdNormalizer()],
    sde=subVP(),
    score_function_class=MLP,
    score_function_hyperparameters={
        "hidden_channels": (200, 100, 100),
        "activation_layer": torch.nn.SiLU,
        "add_prior_score_bias": True,
        "rescale_factor_limit": 1,
        "batch_norm": True,
    },
    optimization=Optimization(
        epochs=2000,
        batch_size=1000,
        optimizer_class=torch.optim.RAdam,
        optimizer_hyperparameters={"lr": 5e-3},
    ),
    score_matching_loss=partial(accumulated_denoising_score_matching_loss, n=10),
    names=["S", "I"],
)

generation_options = dict(
    sde_solver=EulerMethod(), steps=1000, corrector_steps=0, n_samples=100
)

CONFIG = TrainConfig(
    name="eSIRS", dataset=dataset, model=model, generation_options=generation_options
)
