from functools import partial

import torch

from src.data import Dataset
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import MLP
from src.models.score_based.score_matching import \
    accumulated_denoising_score_matching_loss
from src.models.score_based.sdes.sampler import EulerMethod, TimeAdaptiveEuler
from src.models.score_based.sdes.sde import VE, subVP
from src.models.score_based.tabular_score_based_sde import TabularScoreBasedSde
from src.preprocessors.mean_std_normalizer import MeanStdNormalizer
from src.train_config import TrainConfig

dataset = Dataset(path="data/mono_normal.csv", train_proportion=0.8)

model = TabularScoreBasedSde(
    tensor_preprocessors=[
        MeanStdNormalizer(),
    ],
    sde=VE(),
    score_function_class=MLP,
    score_function_hyperparameters={
        "hidden_channels": (32,),
        "activation_layer": torch.nn.SiLU,
        "add_prior_score_bias": True,
        "rescale_factor_limit": 1,
    },
    optimization=Optimization(
        epochs=300,
        batch_size=200,
        optimizer_class=torch.optim.RAdam,
        optimizer_hyperparameters={"lr": 2e-2},
    ),
    score_matching_loss=partial(
        accumulated_denoising_score_matching_loss, reciprocal_distribution_t=True, n=5
    ),
)

generation_options = dict(
    sde_solver=EulerMethod(),
    steps=1000,
    n_samples=1000,
    corrector_steps=0,
    final_corrector_steps=1000,
)

CONFIG = TrainConfig(
    name="normal", dataset=dataset, model=model, generation_options=generation_options
)
