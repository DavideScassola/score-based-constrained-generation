from torch import Tensor

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import *
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

NAME = "eSIRS"

S = 0
I = 1

GRAD = 7.0
SEED = 1234
SCORE_APPROXIMATOR = snr

universal_guidance = dict(
    forward_guidance=False, backward_guidance_steps=0, per_step_self_recurrence_steps=0
)

positive_populations = lambda x: Logic.and_(Logic.greater(x, 0.0, grad=1))
limited_population = lambda x: Logic.and_(Logic.smaller(x.sum(dim=-1), 100.0, grad=1))
fix_S_at_0 = lambda x: Logic.equal(x[:, 0, S], 80.0, grad=GRAD)
fix_I_at_0 = lambda x: Logic.equal(x[:, 0, I], 10.0, grad=GRAD)
fix_S_at_25 = lambda x: Logic.equal(x[:, 25, S], 30.0, grad=GRAD)


def predicate(x: Tensor) -> Tensor:
    return Logic.and_(
        positive_populations(x),
        limited_population(x),
        fix_S_at_0(x),
        fix_I_at_0(x),
        fix_S_at_25(x),
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

name = NAME + "_bridging"
if universal_guidance["forward_guidance"]:
    name = (
        name
        + f"_bw{universal_guidance['backward_guidance_steps']}_rs{universal_guidance['per_step_self_recurrence_steps']}"
    )


CONFIG = ConstrainedGenerationConfig(
    name=name,
    constraint=constraint,
    model_path=find(str(MODELS_FOLDER), pattern=f"*{NAME}*"),
    generation_options=generation_options,
    seed=SEED,
)
