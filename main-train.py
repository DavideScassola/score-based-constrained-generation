import argparse

from src.train_controller import run_model_training

DEBUG_CONFIG = "config/train/debug.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs="?", default=DEBUG_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run_model_training(parse_args().config_file)
