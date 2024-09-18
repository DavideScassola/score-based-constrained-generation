from functools import partial

import torch

from src.data import Dataset
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import MLP, GineMLP
from src.models.score_based.score_matching import (
    accumulated_denoising_score_matching_loss, hybrid_ssm_dsm_loss,
    sliced_score_matching_loss)
from src.models.score_based.sdes.sampler import EulerMethod, TimeAdaptiveEuler
from src.models.score_based.sdes.sde import VE, subVP
from src.models.score_based.tabular_score_based_sde import TabularScoreBasedSde
from src.preprocessors.mean_std_normalizer import MeanStdNormalizer
from src.train_config import TrainConfig

dataset = Dataset(path="data/mono_mixture.csv", train_proportion=0.8)

model = TabularScoreBasedSde(
    tensor_preprocessors=[
        MeanStdNormalizer(),
    ],
    sde=subVP(),
    score_function_class=MLP,
    score_function_hyperparameters={
        "hidden_channels": (64, 64),
        "activation_layer": torch.nn.SiLU,
        "add_prior_score_bias": True,
        "rescale_factor_limit": 5,
        "batch_norm": True,
    },
    optimization=Optimization(
        epochs=600,
        batch_size=500,
        optimizer_class=torch.optim.AdamW,
        optimizer_hyperparameters={"lr": 1e-2},
        # parameters_momentum=0.99,
    ),
    score_matching_loss=partial(
        hybrid_ssm_dsm_loss,
        w=0.01,
    ),
)

generation_options = dict(
    sde_solver=EulerMethod(),
    steps=1000,
    n_samples=2000,
    corrector_steps=1,
    final_corrector_steps=200,
)

CONFIG = TrainConfig(
    name="mixture", dataset=dataset, model=model, generation_options=generation_options
)
