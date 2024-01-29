import torch

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import snr
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

model_path = find(str(MODELS_FOLDER), pattern="*white_wine_sliced")

i = names_index_map("data/wine_quality_white.csv")

alcohol = i["alcohol"]
fixed_acidity = i["fixed acidity"]
residual_sugar = i["residual sugar"]
citric_acid = i["citric acid"]


GRAD = 50.0
SCORE_APPROXIMATOR = snr
SEED = 1234

universal_guidance = dict(
    forward_guidance=False, backward_guidance_steps=0, per_step_self_recurrence_steps=0
)


def predicate(x: torch.Tensor):
    i = torch.arange(0, len(x), 2)
    a = x[i]
    b = x[i + 1]
    out = Logic.greater(a[:, alcohol], b[:, alcohol] + 1, grad=GRAD).flatten()
    return out if len(x) % 2 == 0 else torch.cat([out, torch.tensor([0.0])])


def predicate(x: torch.Tensor):
    sub_size = 2

    y = x[len(x) % sub_size :]
    B = y.shape[0]

    y = y.view((B // sub_size, sub_size, -1))
    out = (
        Logic.greater(y[:, 0, alcohol], y[:, 1, alcohol] + 1, grad=GRAD)
        .reshape(-1, 1)
        .repeat((1, sub_size))
        .flatten()
    )
    return out if len(x) % sub_size == 0 else torch.cat([out, torch.tensor([-100.0])])


constraint = Constraint(
    f=predicate, strength=1.0, gradient_mixer=SCORE_APPROXIMATOR, **universal_guidance
)

generation_options = dict(
    sde_solver=EulerMethod(),
    n_samples=5000,
    steps=1000,
    corrector_steps=5,
    final_corrector_steps=4000,
)

CONFIG = ConstrainedGenerationConfig(
    name="white_wine_multi",
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
