import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CSV_URL = "http://data.insideairbnb.com/united-states/ny/new-york-city/2023-12-04/visualisations/listings.csv"
COLUMNS_TO_DROP = ["id", "name", "host_id", "host_name", "last_review", "license"]


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(CSV_URL, csv_path())
    df = pd.read_csv(csv_path())
    df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)
    df.to_csv(csv_path(), index=False)


if __name__ == "__main__":
    main()
