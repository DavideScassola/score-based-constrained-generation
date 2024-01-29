import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(".")
from src.report import *


def get_dataframes(path):
    df_guidance = pd.read_csv(path + "/report/guidance.csv")
    df_no_guidance = pd.read_csv(path + "/report/no_guidance.csv")
    df_rejection_sampling = pd.read_csv(path + "/report/rejection_sampling.csv")
    return df_guidance, df_no_guidance, df_rejection_sampling


def split_dataframe(df: pd.DataFrame):
    n = len(df)
    i = np.arange(0, n, 2) + n % 2
    return df.iloc[i], df.iloc[i + 1]


def alc_experiment_rej_sampling(df: pd.DataFrame):
    i = np.arange(0, len(df) - 1)
    x = df["alcohol"].to_numpy()
    selection_a = np.where(x[i] >= (x[i + 1] + 1))[0]
    selection_b = selection_a + 1
    return df.iloc[selection_a], df.iloc[selection_b]


def report(
    df_guidance: pd.DataFrame, df_rejection_sampling: pd.DataFrame, name: str, path: str
):
    subpath = Path(path + f"/{name}")
    os.makedirs(subpath, exist_ok=True)
    store_samples(df_generated=df_guidance, path=subpath, name="guidance.csv")
    store_samples(
        df_generated=df_rejection_sampling, path=subpath, name="rejection_sampling.csv"
    )
    for comparison in (
        histograms_comparison,
        statistics_comparison,
        correlations_comparison,
    ):
        comparison(
            df_generated=df_guidance,
            name_generated=f"guidance_{name}",
            df_train=df_rejection_sampling,
            name_original=f"rejection sampling_{name}",
            path=subpath,
            constraint=None,
        )

    summary_report(path=Path(path) / Path(name))


path = sys.argv[1]
df_guidance, df_no_guidance, _ = get_dataframes(path)
df_guidance_a, df_guidance_b = split_dataframe(df_guidance)


df_rejection_sampling_a, df_rejection_sampling_b = alc_experiment_rej_sampling(
    df_no_guidance
)

report(df_guidance_a, df_rejection_sampling_a, "a", path)
report(df_guidance_b, df_rejection_sampling_b, "b", path)
