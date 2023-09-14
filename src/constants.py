from pathlib import Path

EXPERIMENTS_FOLDER = Path("artifacts")
MODELS_FOLDER_NAME = Path("models")
CONFIG_FILE_NAME = Path("config.py")
CONSTRAINED_GENERATIONS_FOLDER = Path("constrained_generation")
DATASET_SCRIPTS_FOLDER = Path("datasets_scripts")
MINIMUM_SAMPLING_T = 1e-4  # TODO: check this

MODELS_FOLDER = EXPERIMENTS_FOLDER / MODELS_FOLDER_NAME
LANGEVIN_CLEANING_PATIENCE = 100  # TODO: make configurable
ALLOW_LANGEVIN_CLEANING = False  # TODO: make configurable
