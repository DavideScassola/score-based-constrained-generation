from pathlib import Path

from src.constants import CONFIG_FILE_NAME
from src.models.model import Model
from src.train_config import getConfig


def load_model(model_path: str) -> Model:
    train_config = getConfig(Path(model_path) / CONFIG_FILE_NAME)
    train_config.model.load_(model_path)
    return train_config.model
