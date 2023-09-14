from functools import partial

import torch

from src.data import Dataset
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import MLP
from src.models.score_based.score_matching import (
    accumulated_denoising_score_matching_loss, denoising_score_matching_loss,
    hybrid_ssm_dsm_loss, sliced_score_matching_loss)
from src.models.score_based.sdes.sampler import EulerMethod
from src.models.score_based.sdes.sde import VE, subVP
from src.models.score_based.tabular_score_based_sde import TabularScoreBasedSde
from src.preprocessors.mean_std_normalizer import MeanStdNormalizer
from src.train_config import TrainConfig

dataset = Dataset(path="data/airbnb.csv", train_proportion=1.0)

model = TabularScoreBasedSde(
    tensor_preprocessors=[
        MeanStdNormalizer(),
    ],
    sde=subVP(),
    score_function_class=MLP,
    score_function_hyperparameters={
        "hidden_channels": (100, 50),
        "activation_layer": torch.nn.SiLU,
        "add_prior_score_bias": True,
        "rescale_factor_limit": 4,
    },
    optimization=Optimization(
        epochs=100,
        batch_size=1000,
        optimizer_class=torch.optim.RAdam,
        optimizer_hyperparameters={"lr": 1e-3},
        # parameters_momentum=0.999,
    ),
    score_matching_loss=partial(hybrid_ssm_dsm_loss, w=0.1),
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=3000,
    steps=1000,
    corrector_steps=1,
    final_corrector_steps=100,
)

CONFIG = TrainConfig(
    name="airbnb",
    dataset=dataset,
    model=model,
    generation_options=generation_options,
)
