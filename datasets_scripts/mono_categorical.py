import os
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = {"A": 0.1, "B": 0.45, "C": 0.3, "D": 0.15}


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def generate(size: int = 1000) -> pd.DataFrame:
    c = np.random.choice(list(CLASSES.keys()), size=size, p=list(CLASSES.values()))
    return pd.DataFrame({f"categorical({str(CLASSES)})": c})


def store(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    generate(1000).to_csv(csv_path(), index=False)


def main():
    store(generate())


if __name__ == "__main__":
    main()
