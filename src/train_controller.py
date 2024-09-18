import os
import shutil
from datetime import datetime
from pathlib import Path

from src.constants import CONFIG_FILE_NAME, MODELS_FOLDER
from src.util import create_experiment_folder, set_seeds

from .train_config import TrainConfig, getConfig


def create_models_folder() -> Path:
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    return MODELS_FOLDER


def create_model_folder() -> Path:
    models_folder = create_models_folder()
    folder_name = Path(datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f"))
    path = models_folder / folder_name
    os.makedirs(path)
    return path


def store_results(*, config_file: str | Path, config: TrainConfig, postfix: str | None):
    folder = Path(
        create_experiment_folder(path=create_models_folder(), postfix=postfix)
    )
    config.model.store(folder)
    shutil.copyfile(config_file, folder / CONFIG_FILE_NAME)
    config.model.generate_report(
        path=folder,
        dataset=config.dataset,
        generation_options=config.generation_options,
    )


def run_model_training(config_file: str):
    config = getConfig(config_file)
    if config.seed:
        set_seeds(config.seed)
    config.model.train(config.dataset, device=config.device)
    store_results(config_file=config_file, config=config, postfix=config.name)
