import os

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

NAME = "adult"
model_path = find(str(MODELS_FOLDER), pattern=f"*{NAME}")

GRAD = 30.0
SCORE_APPROXIMATOR = snr
SEED = 1234

universal_guidance = dict(
    forward_guidance=False, backward_guidance_steps=0, per_step_self_recurrence_steps=0
)

age_i = 0
ed_master_i = 23  # (11, 27) + 12
race_white_i = 75  # (71, 76) + 4


predicate = lambda x: Logic.and_(
    Logic.greater(x[:, age_i], 39.6, grad=GRAD),
    Logic.or_(
        Logic.equal(x[:, race_white_i], 0.0, grad=GRAD),
        Logic.equal(x[:, ed_master_i], 1.0, grad=GRAD),
    ),
)

constraint = Constraint(
    f=lambda x: predicate(x),
    strength=1.0,
    gradient_mixer=SCORE_APPROXIMATOR,
    hard_f=lambda df: (df["age"] >= 40.0)
    & ((df["race"] != "White") | (df["education"] == "Masters")),
    **universal_guidance,
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=5000,
    steps=1500,
    corrector_steps=1,
    final_corrector_steps=10,
)

name = NAME
if universal_guidance["forward_guidance"]:
    name = (
        NAME
        + f"_bw{universal_guidance['backward_guidance_steps']}_rs{universal_guidance['per_step_self_recurrence_steps']}"
    )

CONFIG = ConstrainedGenerationConfig(
    name=name,
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
