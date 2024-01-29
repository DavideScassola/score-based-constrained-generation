import json
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = 3.9
SEED = 1234

_, axes = plt.subplots(nrows=2, ncols=1, figsize=(BASE, BASE / 1.2), sharex=True)


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def store_json(d: dict, *, file: str) -> None:
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def positivity(x: np.ndarray):
    return np.all((x >= 0).reshape(len(x), -1), axis=-1)


def constant_population(x: np.ndarray):
    return np.all(np.sum(x, axis=-1) <= 100, axis=-1)


def less_20_infected(x: np.ndarray):
    return np.all(x[:, :, 1] <= 20, axis=-1)


def time_series_plot(
    x: np.ndarray, *, features_names: list, color: str, label: str, alpha=0.5
) -> None:
    k = len(features_names)

    for i, name in enumerate(features_names):
        axes[i].plot(x[:, :, i].T, alpha=alpha, color=color, linewidth=0.8)
        axes[i].set_ylabel(name)
    plt.xlabel("time")


set_seeds(SEED)

paths = sys.argv[1:]

print(paths)

if not isinstance(paths, list):
    paths = [paths]

for path in paths:
    plt.cla()
    colors = {"Guidance": "orange", "No guidance": "blue"}
    x = {
        "Guidance": np.load(path + "/report/guidance.npy"),
        "No guidance": np.load(path + "/report/no_guidance.npy"),
    }

    x["Guidance"] = np.round(x["Guidance"][np.random.permutation(len(x["Guidance"]))])
    x["No guidance"] = np.round(
        x["No guidance"][np.random.permutation(len(x["No guidance"]))]
    )

    index = {"S": 0, "I": 1}

    ALPHA = 0.6
    N_SAMPLES = 10
    bridging = np.mean(np.round(x["Guidance"][:, 0, 0]) == 95.0) > 0.5

    stats = {}

    for i in ("No guidance", "Guidance"):
        samples = np.round(x[i])
        sub_sample = samples[:N_SAMPLES]
        time_series_plot(
            sub_sample, features_names=["S", "I"], color=colors[i], alpha=ALPHA, label=i
        )
        stats[f"{i} positivity"] = np.mean(positivity(samples))
        stats[f"{i} constant population: "] = np.mean(constant_population(samples))

    for ax in axes:
        ax.grid(alpha=0.4, zorder=-1)
        ax.set_ylim(0, None)

    # print("less 20 infected: ", np.mean(less_20_infected(x["Guidance"])))
    stats["less 20 infected"] = np.mean(less_20_infected(x["Guidance"]))
    axes[1].axhline(y=20, color="r", linestyle="--", alpha=0.6, linewidth=1.0)

    legend = axes[1].legend(("No guidance", "Guidance"))
    legend.legend_handles[0].set_color("blue")
    legend.legend_handles[1].set_color("orange")
    print(stats)

    plt.tight_layout(pad=0.1)
    plt.savefig(path + "/paper_plot.svg")
    store_json(stats, file=path + "/plot_stats.json")
    plt.show()
