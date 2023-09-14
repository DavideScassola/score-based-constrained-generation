import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = sys.argv[1]

# Read the data from CSV files
df_no_guidance = pd.read_csv(path + "/report/no_guidance.csv")
df_guidance = pd.read_csv(path + "/report/guidance.csv")
df_rejection_sampling = pd.read_csv(path + "/report/rejection_sampling.csv")

# Set up the figure and axes for the two subplots
fig, (ax_upper, ax_lower) = plt.subplots(
    nrows=2, ncols=1, figsize=(5.0, 2), sharex=True
)

# Set the number of bins for both subplots
bins = 50

# Plot histogram with KDE for 'no_guidance.csv' in the upper subplot
sns.histplot(
    data=df_no_guidance,
    x=df_no_guidance.columns[0],
    kde=True,
    stat="density",
    bins=bins,
    ax=ax_upper,
)
# ax_upper.set_title("Histogram of 'no_guidance.csv'")
ax_upper.legend(labels=["No guidance"])

# Calculate histogram bins for the lower plot using all data
# _, bins, _ = ax_upper.hist(df_guidance.iloc[:, 0], bins=bins, density=True)
sns.histplot(
    data=df_guidance,
    x=df_guidance.columns[0],
    kde=True,
    stat="density",
    bins=bins,
    ax=ax_lower,
    color="blue",
    alpha=0.35,
)

_, bins, _ = ax_lower.hist(df_guidance.iloc[:, 0], bins=bins, density=True, alpha=0.0)


sns.histplot(
    data=df_rejection_sampling,
    x=df_rejection_sampling.columns[0],
    kde=True,
    stat="density",
    bins=bins,
    ax=ax_lower,
    color="red",
    alpha=0.35,
)

# ax_lower.set_title("Comparison of 'guidance.csv' and 'rejection_sampling.csv'")
ax_lower.legend(labels=["Guidance", "Rejection sampling"])

# Set common labels
# fig.text(0.5, 0.04, "Values", ha="center")
# fig.text(0.04, 0.5, "Probability Density", va="center", rotation="vertical")
ax_upper.set_xlabel("")
ax_lower.set_xlabel("")

# Set x-limits of the lower subplot to match the upper subplot
lower_xlim = ax_upper.get_xlim()
ax_lower.set_xlim(lower_xlim)
# ax_upper.set_xticks([])  # Remove x-ticks from the upper subplot

plt.tight_layout(pad=0.2)
plt.savefig(path + "/paper_plot.svg")
plt.show()
