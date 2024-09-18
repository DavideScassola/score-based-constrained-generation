import sys

sys.path.append(".")

import pandas as pd
import tqdm

from src.util import edit_json, load_module

paths = sys.argv[1:]

if not isinstance(paths, list):
    paths = [paths]

for path in tqdm.tqdm(paths):
    # find config
    config_path = f"{path}/config.py"

    # find hard constraint from config
    module = load_module(config_path)
    hard_constraint = module.CONFIG.constraint.hard_f

    if hard_constraint is not None:
        # find guidance.csv and rejection_sampling.csv
        guidance_df = pd.read_csv(f"{path}/report/guidance.csv")
        rejection_sampling_df = pd.read_csv(f"{path}/report/rejection_sampling.csv")

        # compute hard constraint satisfaction for guidance.csv and rejection_sampling.csv
        guidance_sat = hard_constraint(guidance_df).mean()
        rejection_sampling_sat = hard_constraint(rejection_sampling_df).mean()

        # add in stats_summary.json the hard constraint satisfaction for both
        with edit_json(f"{path}/report/stats_summary.json") as stats_summary:
            stats_summary["guidance_sat"] = guidance_sat
            stats_summary["rejection_sampling_sat"] = rejection_sampling_sat
