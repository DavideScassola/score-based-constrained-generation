import os
import sys
import urllib.request
from pathlib import Path

sys.path.append(".")
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.util import nan_as_category

SHUFFLE_SEED = 510


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    # fetch dataset
    print("Fetching dataset...")
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    df = adult.data.original

    # find and replace every "."
    df["income"] = df["income"].str.replace(".", "")

    os.makedirs("data", exist_ok=True)
    df = df.dropna().to_csv(csv_path(), index=False)


if __name__ == "__main__":
    main()
