from torch import Tensor

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find

NAME = "eSIRS"

S = 0
I = 1

GRAD = 25.0
SEED = 1234

positive_populations = lambda x: Logic.and_(Logic.greater(x, 0.0, grad=1))
limited_population = lambda x: Logic.and_(Logic.smaller(x.sum(dim=2), 100.0, grad=1))
limited_I_from_0 = lambda x: Logic.and_(Logic.smaller(x[:, 0:, I], 20.0, grad=GRAD))


def predicate(x: Tensor) -> Tensor:
    return Logic.and_(
        positive_populations(x), limited_population(x), limited_I_from_0(x)
    )


constraint = Constraint(f=lambda x: predicate(x), strength=1.0, gradient_mixer=snr)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=100,
    steps=1000,
    corrector_steps=2,
    final_corrector_steps=2500,
)

CONFIG = ConstrainedGenerationConfig(
    name=NAME + "_inequality",
    constraint=constraint,
    model_path=find(str(MODELS_FOLDER), pattern=f"*{NAME}*"),
    generation_options=generation_options,
    seed=SEED,
)
