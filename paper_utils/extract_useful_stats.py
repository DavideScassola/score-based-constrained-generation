import json
import os
import sys
from pathlib import Path

import pandas as pd


def load_json(file: str | Path) -> dict:
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def store_json(d: dict, *, file: str | Path):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


paths = sys.argv[1:]

print(paths)

if not isinstance(paths, list):
    paths = [paths]

for path in paths:
    p = path + "/report/stats.json"

    if os.path.isfile(p):
        stats = load_json(p)
        interesting_stats = {}

        del stats["columnwise_comparison"]["l1_divergence"]["constraint_value"]
        interesting_stats["l1_divergence"] = stats["columnwise_comparison"][
            "l1_divergence"
        ]
        interesting_stats["correlations_l1"] = stats["global_comparison"][
            "correlation matrix similarity"
        ]["average l1"]

        l1 = pd.Series(interesting_stats["l1_divergence"])

        interesting_stats["l1_divergence_median"] = l1.median()
        interesting_stats["l1_divergence_mean"] = l1.mean()
        interesting_stats["l1_divergence_max"] = l1.max()

        store_json(interesting_stats, file=path + "/interesting_stats.json")
