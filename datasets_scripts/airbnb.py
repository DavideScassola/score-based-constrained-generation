import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZIP_URL = "https://www.dropbox.com/scl/fi/wyz56vpff026gou08s230/airbnb_ny.zip?rlkey=bix104azzv9uc9uqs9yskh3po&dl=1"
CSV_PATH = "data/AB_NYC_2019.csv"
ZIP_PATH = "data/airbnb_ny.zip"


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def main():
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        # zip_ref.extractall(directory_to_extract_to)
        zip_ref.extract("AB_NYC_2019.csv", "data")

    pd.read_csv(CSV_PATH)[["latitude", "longitude"]].to_csv(csv_path(), index=False)
    os.remove(ZIP_PATH)
    os.remove(CSV_PATH)


if __name__ == "__main__":
    main()
