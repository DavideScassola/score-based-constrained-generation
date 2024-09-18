from functools import partial

import torch

from src.data import Dataset
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import MLP
from src.models.score_based.score_matching import accumulated_denoising_score_matching_loss
from src.models.score_based.sdes.sampler import EulerMethod
from src.models.score_based.sdes.sde import VE, subVP
from src.models.score_based.tabular_score_based_sde import TabularScoreBasedSde
from src.preprocessors.everything_to_float import EverythingToFloat
from src.preprocessors.mean_std_normalizer import MeanStdNormalizer
from src.train_config import TrainConfig

SEED = 1234
NAME = "adult"

dataset = Dataset(path="data/adult.csv", train_proportion=0.8)

model = TabularScoreBasedSde(
    df2tensor=EverythingToFloat(),
    tensor_preprocessors=[
        MeanStdNormalizer(),
    ],
    sde=subVP(beta_min=0.01),
    score_function_class=MLP,
    score_function_hyperparameters={
        "hidden_channels": (500, 500, 300),
        "activation_layer": torch.nn.ELU,
        "add_prior_score_bias": True,
        "rescale_factor_limit": 5,
    },
    optimization=Optimization(
        epochs=10000,
        batch_size=2000,
        optimizer_class=torch.optim.RAdam,
        optimizer_hyperparameters={"lr": 1e-3},
        # parameters_momentum=0.999,
    ),
    score_matching_loss=partial(
        accumulated_denoising_score_matching_loss, n=2, reciprocal_distribution_t=False
    ),
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=1000,
    steps=1000,
    corrector_steps=1,
    final_corrector_steps=20,
)

CONFIG = TrainConfig(
    name=NAME,
    dataset=dataset,
    model=model,
    generation_options=generation_options,
    seed=SEED,
)
