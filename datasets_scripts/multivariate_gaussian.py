import os

import numpy as np
import pandas as pd


def generate(size: int = 1000) -> pd.DataFrame:
    cov = [[2.0, 1.0, -1.0], [1.0, 1.0, 0.2], [-1, 0.2, 1.5]]
    mean = [0, -1, 2]
    x = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    return pd.DataFrame(x)


def store(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    name = os.path.basename(__file__).split(".py")[0]
    df.to_csv(f"data/{name}.csv", index=False)


if __name__ == "__main__":
    store(generate())
