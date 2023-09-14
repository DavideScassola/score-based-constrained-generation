from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            logistic_interpolation, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find

SEED = 1234

model_path = find(str(MODELS_FOLDER), pattern="*mixture*")

constraint = Constraint(
    f=lambda x: Logic.greater(x, 0.0),
    strength=1.0,
    gradient_mixer=snr,
)

generation_options = dict(
    sde_solver=EulerMethod(),
    steps=1000,
    n_samples=5000,
    corrector_steps=1,
    final_corrector_steps=200,
)

CONFIG = ConstrainedGenerationConfig(
    name="mode_selection",
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
