import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CSV_URL = "https://datahub.io/machine-learning/adult/r/adult.csv"


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(CSV_URL, csv_path())


if __name__ == "__main__":
    main()
