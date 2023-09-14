import os
import shutil
from pathlib import Path

import src.train_config as train_config
from src.constants import (CONFIG_FILE_NAME, CONSTRAINED_GENERATIONS_FOLDER,
                           EXPERIMENTS_FOLDER)
from src.constrained_generation_config import (ConstrainedGenerationConfig,
                                               getConfig)
from src.constraints.constraint_application import apply_constraint_
from src.models.load_model import Model, load_model
from src.models.score_based.score_based_sde import ScoreBasedSde
from src.util import create_experiment_folder, set_seeds


def create_constrained_generations_folder() -> Path:
    models_folder = EXPERIMENTS_FOLDER / CONSTRAINED_GENERATIONS_FOLDER
    os.makedirs(models_folder, exist_ok=True)
    return models_folder


def store_results(
    *,
    config_file: str | Path,
    constrained_generation_config: ConstrainedGenerationConfig,
    model: Model,
):
    training_config = train_config.getConfig(
        f"{constrained_generation_config.model_path}/{CONFIG_FILE_NAME}"
    )
    training_config.generation_options = (
        constrained_generation_config.generation_options
    )
    folder: Path = Path(
        create_experiment_folder(
            path=create_constrained_generations_folder(),
            postfix=getConfig(str(config_file)).name,
        )
    )
    model.generate_report(
        path=folder,
        dataset=training_config.dataset,
        generation_options=training_config.generation_options,
        constraint=constrained_generation_config.constraint,
    )
    shutil.copyfile(config_file, folder / CONFIG_FILE_NAME)


def run_constrained_generation(config_file: str):
    constrained_generation_config = getConfig(config_file)
    if constrained_generation_config.seed:
        set_seeds(constrained_generation_config.seed)
    model = load_model(constrained_generation_config.model_path)

    if not isinstance(model, ScoreBasedSde):
        raise ValueError(
            "Currently constrained generation is allowed only for ScoreBasedSde models"
        )

    apply_constraint_(model=model, constraint=constrained_generation_config.constraint)
    store_results(
        config_file=config_file,
        constrained_generation_config=constrained_generation_config,
        model=model,
    )
