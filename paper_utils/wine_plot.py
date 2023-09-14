import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = sys.argv[1]

# Read the data from CSV files
df_no_guidance = pd.read_csv(path + "/report/no_guidance.csv")
df_guidance = pd.read_csv(path + "/report/guidance.csv")
df_rejection_sampling = pd.read_csv(path + "/report/rejection_sampling.csv")

# select interesting columns
features = ["alcohol", "fixed acidity", "residual sugar", "citric acid"]
n_features = len(features)

# Set up the figure and axes for the two subplots
cols = 2
fig, axes = plt.subplots(
    nrows=n_features // cols, ncols=cols, figsize=(4.5, 3.4), sharex=False, sharey=False
)
axes = axes.flatten()


# Set the number of bins for both subplots
bins = int(len(df_guidance) ** 0.5)

# colors
colors = ["gray", "blue", "red"]

ALPHA = 0.35
BW_ADJUST = {
    "alcohol": 0.5,
    "fixed acidity": 0.15,
    "residual sugar": 0.5,
    "citric acid": 0.4,
}

for i in range(n_features):
    ax = axes[i]
    feature = features[i]
    bw_adjust = BW_ADJUST[feature]

    sns.kdeplot(
        data=df_no_guidance,
        x=df_no_guidance[feature],
        # kde=True,
        # stat="density",
        # bins=bins,
        ax=ax,
        color=colors[0],
        bw_adjust=0.8,
        alpha=0.5,
    )

    # _, copied_bins, _ = ax.hist(
    #    df_guidance[feature], bins=bins, density=True, alpha=0.0
    # )

    if feature == "fixed acidity":
        bw_adjust = 0.2

    sns.kdeplot(
        data=df_rejection_sampling,
        x=df_rejection_sampling[feature],
        # kde=True,
        # stat="density",
        # bins=copied_bins,
        ax=ax,
        color=colors[2],
        alpha=ALPHA,
        bw_adjust=bw_adjust,
        fill=True,
    )

    sns.kdeplot(
        data=df_guidance,
        x=df_guidance[feature],
        # kde=True,
        # stat="density",
        # bins=bins,
        ax=ax,
        color=colors[1],
        alpha=ALPHA,
        bw_adjust=bw_adjust,
        fill=True,
    )

    # ax.set_ylabel(feature)
    # ax.set_xlabel(None)
    ax.tick_params(axis="y", which="both", labelleft=False)
for i in (1, 3):
    axes[i].set_ylabel(None)

axes[-2].legend(labels=["No guidance", "Guidance", "Rej. samp."])


plt.tight_layout(pad=0.2)
plt.savefig(path + "/paper_plot.svg")
plt.show()
