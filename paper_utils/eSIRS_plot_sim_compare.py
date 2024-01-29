import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from eSIRS_bridging_simulate import simulate_bridging

BASE = 3.9
SEED = 1234

import tqdm

paths = sys.argv[1:]

if not isinstance(paths, list):
    paths = [paths]

for path in tqdm.tqdm(paths):
    plt.cla()
    plt.close("all")

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

    def l1_divergence(x: np.ndarray, y: np.ndarray) -> float:
        bounds = (min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
        bins = np.linspace(-0.5, 100.5, 102)

        def density_estimator(k):
            h = np.histogram(k, density=False, range=bounds, bins=bins)[0]
            return h / np.sum(h)

        px = density_estimator(x)
        py = density_estimator(y)

        return np.sum(np.abs(px - py)) / 2

    def l1_divergences(x, y):
        x = np.round(x)
        y = np.round(y)

        l = x.shape[1]
        d_S = [l1_divergence(x[:, t, 0], y[:, t, 0]) for t in range(l)]
        d_I = [l1_divergence(x[:, t, 1], y[:, t, 1]) for t in range(l)]
        return d_S, d_I

    set_seeds(SEED)

    SIMULATION_FILE = "paper_utils/eSIRS_S0=80_I0=10_S25=30.npy"

    if not os.path.exists(SIMULATION_FILE):
        simulate_bridging()

    # path = sys.argv[1]
    # config_module = load_module(path=path + '/config.py')

    colors = {"Guidance": "orange", "Simulated": "teal"}
    x = {
        "Guidance": np.load(path + "/report/guidance.npy"),
        "Simulated": np.load("paper_utils/eSIRS_S0=80_I0=10_S25=30.npy"),
    }

    x["Guidance"] = x["Guidance"][np.random.permutation(len(x["Guidance"]))]
    x["Simulated"] = x["Simulated"][np.random.permutation(len(x["Simulated"]))]

    index = {"S": 0, "I": 1}

    ALPHA = 0.6
    N_SAMPLES = 10
    h_line = np.mean(less_20_infected(x["Guidance"])) > 0.5
    bridging = np.mean(np.round(x["Guidance"][:, 0, 0]) == 80.0) > 0.5
    stats = {}

    for i in ("Simulated", "Guidance"):
        samples = np.round(x[i])
        sub_sample = samples[:N_SAMPLES]
        time_series_plot(
            sub_sample, features_names=["S", "I"], color=colors[i], alpha=ALPHA, label=i
        )
        stats[f"{i} positivity"] = np.mean(positivity(samples))
        stats[f"{i} constant population: "] = np.mean(constant_population(samples))

    for ax in axes:
        ax.grid(alpha=0.4, zorder=-1)

    if True:
        s = np.round(x["Guidance"])
        stats[f"|S(0)-80|"] = float(np.mean(abs(s[:, 0, 0] - 80)))
        stats[f"|I(0)-10|"] = float(np.mean(abs(s[:, 0, 1] - 10)))
        stats[f"|S(25)-30|"] = float(np.mean(abs(s[:, 25, 0] - 30)))

        dS, dI = l1_divergences(x["Simulated"], x["Guidance"])
        aLl_dims = np.concatenate((dS, dI))
        for d in ("dS", "dI", "aLl_dims"):
            for stat in ("max", "median", "mean"):
                s = eval(f"np.{stat}({d})")
                stats[f"{stat}({d[1]})"] = float(s)

    legend = axes[0].legend(("Simulated", "Guidance"))
    legend.legend_handles[0].set_color(colors["Simulated"])
    legend.legend_handles[1].set_color(colors["Guidance"])

    store_json(stats, file=path + "/plot_stats.json")
    print(stats)

    plt.tight_layout(pad=0.1)
    plt.savefig(path + "/paper_plot.svg")
    plt.show()
