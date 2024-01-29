import os

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

model_path = find(str(MODELS_FOLDER), pattern="*white_wine_dsm")

i = names_index_map("data/wine_quality_white.csv")

alcohol = i["alcohol"]
fixed_acidity = i["fixed acidity"]
residual_sugar = i["residual sugar"]
citric_acid = i["citric acid"]


GRAD = 30.0
SCORE_APPROXIMATOR = snr
SEED = 1234

universal_guidance = dict(
    forward_guidance=False, backward_guidance_steps=0, per_step_self_recurrence_steps=0
)

predicate = lambda x: Logic.and_(
    Logic.or_(
        Logic.in_(x[:, fixed_acidity], 5.0, 6.0, grad=GRAD),
        Logic.in_(x[:, fixed_acidity], 8.0, 9.0, grad=GRAD),
    ),
    Logic.greater(x[:, alcohol], x2=11.0, grad=GRAD),
    Logic.or_(
        Logic.greater(x[:, residual_sugar], 5.0, grad=GRAD),
        Logic.greater(x[:, citric_acid], 0.5, grad=GRAD),
    ),
)


constraint = Constraint(
    f=lambda x: predicate(x),
    strength=1.0,
    gradient_mixer=SCORE_APPROXIMATOR,
    **universal_guidance,
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=5000,
    steps=1000,
    corrector_steps=2,
    final_corrector_steps=2000,
)

name = "white_wine_dsm"
if universal_guidance["forward_guidance"]:
    name = (
        name
        + f"_bw{universal_guidance['backward_guidance_steps']}_rs{universal_guidance['per_step_self_recurrence_steps']}"
    )

CONFIG = ConstrainedGenerationConfig(
    name=name,
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
