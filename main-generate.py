import argparse

from src.constrained_generation_controller import run_constrained_generation

DEBUG_CONFIG = "config/constrained_generation/debug.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs="?", default=DEBUG_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run_constrained_generation(parse_args().config_file)
