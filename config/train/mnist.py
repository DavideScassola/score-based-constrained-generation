import torch

from src.data import Dataset
from src.models.score_based.image_score_based_sde import ImageScoreBasedSde
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_function import MLP, Unet
from src.models.score_based.sdes.sampler import EulerMethod
from src.models.score_based.sdes.sde import VE, subVP
from src.preprocessors.mean_std_normalizer import MeanStdNormalizer
from src.preprocessors.rescaler import Rescaler
from src.train_config import TrainConfig

dataset = Dataset(path="MNIST", train_proportion=1.0)

model = ImageScoreBasedSde(
    tensor_preprocessors=[],
    sde=VE(),
    score_function_class=Unet,
    score_function_hyperparameters={
        "add_prior_score_bias": True,
        "rescale_factor_limit": 10,
    },
    optimization=Optimization(
        epochs=1000,
        batch_size=512,
        optimizer_class=torch.optim.RAdam,
        optimizer_hyperparameters={"lr": 1e-3},
    ),
)

generation_options = dict(
    sde_solver=EulerMethod(), n_samples=20, steps=1000, corrector_steps=0
)

CONFIG = TrainConfig(
    name=f"mnist_{model.optimization.epochs}",
    dataset=dataset,
    model=model,
    generation_options=generation_options,
)
