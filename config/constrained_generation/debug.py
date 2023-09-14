from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import *
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find

constraint = Constraint(
    f=lambda x: Logic.or_(Logic.greater(x, 0.5), Logic.smaller(x, -0.5)),
    strength=1.0,
    gradient_mixer=linear_interpolation,
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=4000,
    steps=1500,
    corrector_steps=0,
    final_corrector_steps=0,
)

CONFIG = ConstrainedGenerationConfig(
    name="not test",
    constraint=constraint,
    model_path=find(str(MODELS_FOLDER), pattern="*normal*"),  # type: ignore
    generation_options=generation_options,
)
