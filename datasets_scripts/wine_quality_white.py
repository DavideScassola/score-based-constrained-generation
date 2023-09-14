import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(CSV_URL, csv_path())
    df = pd.read_csv(csv_path(), sep=";")
    df.drop("quality", axis=1, inplace=True)
    df.to_csv(csv_path(), index=False)


if __name__ == "__main__":
    main()
