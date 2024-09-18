from functools import partial

import torch

from src.data import Dataset
from src.models.diffusion.absorbing_diffusion_model import \
    AbsorbingDiscreteDiffusion
from src.models.diffusion.absorbing_diffusion_process import \
    AbsorbingDiffusionProcess
from src.models.diffusion.masked_predictor import MaskedMLP
from src.models.score_based.nn.optimization import Optimization
from src.preprocessors.hypercube_normalizer import HypercubeNormalizer
from src.preprocessors.quantizer import Quantizer
from src.train_config import TrainConfig

dataset = Dataset(path="data/airbnb.csv", train_proportion=1.0)


model = AbsorbingDiscreteDiffusion(
    tensor_preprocessors=[
        HypercubeNormalizer(),
        Quantizer(base=10, base10digits=3),
    ],
    masked_predictor_class=MaskedMLP,
    masked_predictor_params={
        "hidden_channels": (100, 100, 100),
        "activation_layer": torch.nn.SiLU,
        "time_embedding": None,
    },
    absorbing_diffusion_process=AbsorbingDiffusionProcess(
        num_timesteps=1000, mask_schedule="random"
    ),
    optimization=Optimization(
        epochs=350,
        batch_size=1000,
        optimizer_class=torch.optim.RAdam,
        optimizer_hyperparameters={"lr": 1e-3},
    ),
)

generation_options = dict(
    n_samples=4000,
    sample_steps=28,
)

CONFIG = TrainConfig(
    name="airbnb_discrete",
    dataset=dataset,
    model=model,
    generation_options=generation_options,
)
