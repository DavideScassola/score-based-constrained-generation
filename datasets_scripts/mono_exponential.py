import os

import numpy as np
import pandas as pd


def generate(size: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"exponential(1)": np.random.exponential(1, size)})


def store(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    name = os.path.basename(__file__).split(".py")[0]
    df.to_csv(f"data/{name}.csv", index=False)


if __name__ == "__main__":
    store(generate())
