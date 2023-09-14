from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from matplotlib import image, rcParams
from scipy import stats
from scipy.stats import kstest
from sklearn.neighbors import KernelDensity
from torch import Tensor

from src.constants import MINIMUM_SAMPLING_T
from src.constraints.constraint import Constraint
from src.util import l1_divergence, max_symmetric_D_kl, store_json, upscale

SAMPLES_NAME = "samples.csv"
CORRELATIONS_NAME = "correlations"
HISTOGRAMS_NAME = "histograms"
STATS_NAME = "stats.json"
REPORT_FOLDER_NAME = "report"
IMAGE_FORMAT = "png"


def kstest_pvalue_and_bool(x, y):
    result = kstest(x, y)
    return (1 if result.pvalue > 0.1 else 0, result.pvalue)


def list_corr(df):
    m = df.corr().to_numpy()
    return m[np.triu_indices_from(m, 1)]


def correlation_matrix_similarities(df1: pd.DataFrame, df2: pd.DataFrame):
    if "constraint_value" in df1.columns:
        df1 = df1.drop("constraint_value", axis=1)
        df2 = df2.drop("constraint_value", axis=1)
    c1 = list_corr(df1)
    c2 = list_corr(df2)

    return {
        # "pearson corr": stats.pearsonr(c1, c2),
        # "spearman corr": stats.spearmanr(c1, c2),
        "average l1": np.mean(abs(c1 - c2)),
    }


NUMERICAL_STATS = {
    "mean": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0, ddof=1),
    "correlations": lambda df: df.corr(),
}

COLUMNWISE_NUMERICAL_COMPARISON_STATS = {
    "l1_divergence": l1_divergence,
    "max_symmetric_D_kl": max_symmetric_D_kl,
    "two-sample Kolmogorov-Smirnov test": kstest_pvalue_and_bool,
}

GLOBAL_NUMERICAL_COMPARISON_STATS = {
    "correlation matrix similarity": correlation_matrix_similarities
}

CATEGORICAL_STATS = {}

DEFAULT_KDE_ARGS = {}


def kde_plot(
    samples: pd.Series,
    *,
    ax=None,
    kde_args: dict = DEFAULT_KDE_ARGS,
    plt_args: dict = {},
    start=None,
    end=None,
    constraint: None | Constraint = None,
):
    ax = ax if ax else plt
    x = np.linspace(
        np.min(samples) if start is None else start,
        np.max(samples) if end is None else end,
        1000,
    )
    c = (
        constraint.f(torch.from_numpy(x)).numpy() * constraint.strength
        if constraint
        else 0.0
    )
    kde = KernelDensity(**kde_args).fit(samples.to_numpy().reshape(-1, 1))
    density = np.exp(kde.score_samples(x.reshape(-1, 1)) + c)
    density /= np.trapz(y=density, x=x)
    ax.plot(x, density, **plt_args)


def bins(s: pd.Series):
    return np.linspace(min(s), max(s), int(len(s) ** 0.5))


def constraint_plot(
    *, start: float, end: float, constraint: Constraint, max_y: float | None = None
):
    x = np.linspace(start, end, 1000)
    c = np.exp(constraint.f(torch.from_numpy(x)).numpy() * constraint.strength)
    if max_y:
        c -= np.max(c) * 1.05
        c *= 0.25 * max_y / abs(np.min(c))
    plt.plot(x, c, label="constraint", alpha=0.8, color="black", linestyle="--")


def histograms_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    constraint = kwargs["constraint"]

    n = len(df_generated.columns)
    fig, axes = plt.subplots(n, 1, sharex=False)
    if n == 1:
        axes = [axes]
    fig.set_size_inches(6, n + 4)
    fig.suptitle("Histogram Comparison")

    plt.tight_layout()

    for i, c in enumerate(df_generated.columns):
        is_categorical = df_generated[c].dtype in (str, "O")

        if is_categorical:
            common_args = dict(
                x=c,
                stat="proportion",
                alpha=0.5,
            )
        else:
            data = pd.Series(pd.concat((df_generated[c], df_train[c])))
            std = data.std()
            start = np.min(data.to_numpy()) - 0.5 * std
            end = np.max(data.to_numpy()) + 0.5 * std
            b = bins(pd.Series(pd.concat((df_generated[c], df_train[c]))))
            start = start
            end = end
            common_args = dict(x=c, stat="density", alpha=0.5, bins=b)

        ax_true = sns.histplot(
            ax=axes[i], data=df_train, label=name_original, **common_args
        )

        bandwidth = min(df_train[c].std(), df_generated[c].std()) / 10

        kde_plot(
            df_train[c],
            ax=axes[i],
            start=start,  # type: ignore
            end=end,  # type: ignore
            kde_args={"bandwidth": bandwidth},
            plt_args={"color": "blue", "alpha": 0.5, "linestyle": "-"},
        )

        ax_generated = sns.histplot(
            ax=axes[i], data=df_generated, label=name_generated, **common_args
        )
        kde_plot(
            df_generated[c],
            ax=axes[i],
            start=start,  # type: ignore
            end=end,  # type: ignore
            kde_args={"bandwidth": bandwidth},
            plt_args={"color": "orange", "alpha": 0.8, "linestyle": "-"},
        )

        if constraint and n == 1 and not is_categorical:
            constraint_plot(
                start=start,  # type: ignore
                end=end,  # type: ignore
                max_y=max(ax_true.get_ylim()[1], ax_generated.get_ylim()[1]),
                constraint=constraint,
            )

            kde_plot(
                df_train[c],
                start=start,  # type: ignore
                end=end,  # type: ignore
                kde_args={"bandwidth": bandwidth},
                plt_args={
                    "color": "blue",
                    "alpha": 0.8,
                    "linestyle": "--",
                    "label": "original constrained",
                },
                constraint=constraint,
            )

    plt.legend()
    plt.savefig(path / Path(f"{HISTOGRAMS_NAME}.{IMAGE_FORMAT}"))
    plt.close()


def store_samples(*, df_generated: pd.DataFrame, path: Path, name: str = SAMPLES_NAME):
    df_generated.to_csv(path / Path(name), index=False)


def get_stats(df: pd.DataFrame):
    df_numerical = df.select_dtypes(include="number")
    df_categorical = df.select_dtypes(include=["O"])
    numerical_stats = {
        k: stat(df_numerical).to_dict() for k, stat in NUMERICAL_STATS.items()
    }
    categorical_stats = {
        k: stat(df_categorical).to_dict() for k, stat in CATEGORICAL_STATS.items()
    }
    return {**numerical_stats, **categorical_stats}


def get_columnwise_comparison_stats(df_train: pd.DataFrame, df_generated: pd.DataFrame):
    df_numerical = df_train.select_dtypes(include="number")
    df_numerical_generated = df_generated.select_dtypes(include="number")

    numerical_stats = {
        k: {
            column: stat(df_numerical[column], df_numerical_generated[column])
            for column in df_train.columns
        }
        for k, stat in COLUMNWISE_NUMERICAL_COMPARISON_STATS.items()
    }

    return numerical_stats


def get_global_comparison_stats(df_train: pd.DataFrame, df_generated: pd.DataFrame):
    df_numerical = df_train.select_dtypes(include="number")
    df_numerical_generated = df_generated.select_dtypes(include="number")

    numerical_stats = {
        k: stat(df_numerical, df_numerical_generated)
        for k, stat in GLOBAL_NUMERICAL_COMPARISON_STATS.items()
    }

    return numerical_stats


def statistics_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    generated_stats = get_stats(df_generated)
    true_data_stats = get_stats(df_train)
    columnwise_comparison_stats = get_columnwise_comparison_stats(
        df_train=df_train, df_generated=df_generated
    )
    global_comparison_stats = (
        get_global_comparison_stats(df_train=df_train, df_generated=df_generated)
        if len(df_train.columns) > 1
        else None
    )
    stats = {
        name_generated: generated_stats,
        name_original: true_data_stats,
        "columnwise_comparison": columnwise_comparison_stats,
    }

    if global_comparison_stats:
        stats["global_comparison"] = global_comparison_stats
    store_json(stats, file=path / STATS_NAME)


def correlations_comparison(
    *,
    df_generated: pd.DataFrame,
    df_train: pd.DataFrame,
    path: Path,
    name_generated="generated",
    name_original="original",
    **kwargs,
):
    df_generated_numerical = df_generated.select_dtypes(include="number")
    df_train_numerical = df_train.select_dtypes(include="number")

    n = len(df_generated_numerical.columns)
    plt.figure().set_size_inches(n / 2 + 4, n / 2)

    if n < 2:
        return
    bi_corr = np.tril(df_generated_numerical.corr()) + np.triu(
        df_train_numerical.corr(), k=1
    )
    df = pd.DataFrame(
        bi_corr,
        index=df_generated_numerical.columns,
        columns=df_generated_numerical.columns,
    )
    sns.heatmap(
        df,
        annot=True,
        vmin=-1.0,
        vmax=1.0,
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        alpha=0.8,
    )

    plt.title(f"Correlations (lower: {name_generated}, upper: {name_original})")
    plt.savefig(path / Path(f"{CORRELATIONS_NAME}.{IMAGE_FORMAT}"))
    plt.close()


def store_images(samples: torch.Tensor, *, folder: str) -> None:
    for i, sample in enumerate(samples):
        image.imsave(
            (f"{folder}/sample_{i}.{IMAGE_FORMAT}"),
            upscale(sample.numpy()),
            cmap="gray",
        )


def time_series_plot(
    x: np.ndarray, *, path: str, features_names: list, color: str
) -> None:
    k = len(features_names)
    aspect_ratio = 2
    h = 4
    fig, axes = plt.subplots(k, figsize=(h * aspect_ratio, h), sharex=True)
    for i, name in enumerate(features_names):
        axes[i].plot(x[:, :, i].T, alpha=0.5, color=color)
        axes[i].set_ylabel(name)
    plt.xlabel("t")
    fig.savefig(path)
    plt.close()


def time_series_plots(
    samples: np.ndarray,
    *,
    folder: str,
    n_plots: int,
    series_per_plot: int,
    features_names: list,
    color: str = "orange",
) -> None:
    assert samples.shape[-1] == len(features_names)

    for i in tqdm.tqdm(range(n_plots), desc="Generating time-series plots"):
        start = i * series_per_plot
        end = min(start + series_per_plot, len(samples))
        time_series_plot(
            x=samples[start:end],
            features_names=features_names,
            path=f"{folder}/{i}.{IMAGE_FORMAT}",
            color=color,
        )
    plt.close()


def score_plot(score_function, path: Path):
    plt.close()
    x = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=score_function.device)

    t_list = np.exp(np.linspace(np.log(MINIMUM_SAMPLING_T), 0, 20))
    for t in t_list:
        score = score_function(X=x_tensor, t=t).cpu()
        plt.plot(x[:], score[:], alpha=max((1 - t) ** 2, 0.05), color="blue")

    plt.plot(x, -x, label="N(0,1)", color="black")

    plt.plot(
        x,
        score_function(X=x_tensor, t=0.0).cpu(),
        alpha=1,
        color="red",
        label=f"t={0}",
        linewidth=2.0,
    )

    # x = x.squeeze()
    # plt.plot(x[:-1], mixture_score(x), label="mixture_score", alpha=0.8, color="green")
    plt.legend()
    plt.savefig(path / Path(f"score.{IMAGE_FORMAT}"))
    plt.close()


def satisfaction_plot(
    X: Tensor, constraint: Constraint, path: str | Path, label: str, **kwargs
) -> None:
    satisfaction = (torch.exp(constraint.f(X) * constraint.strength)).cpu().numpy()
    store_json(
        {
            "median": float(np.median(satisfaction)),
            "std": float(np.std(satisfaction)),
            "perfect_fraction": np.mean(satisfaction >= 0.5),
        },
        file=Path(path) / Path(f"satisfaction_{label}.json"),
    )
    plt.hist(satisfaction, alpha=0.8, **kwargs)
    plt.savefig(f"{str(path)}/constraint_satisfaction_{label}.{IMAGE_FORMAT}")
    plt.close()
