from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures" / "fig_nonredundancy_validation.png"

labels = [
    "Full\nn=60",
    "Selected\nn=4",
    "Drop #1\nn=3",
    "Drop #2\nn=3",
    "Drop #3\nn=3",
    "Drop #4\nn=3",
]
values = [0.9878, 0.9989, 0.1418, 0.8723, 0.9708, 0.9863]
colors = ["#8f9a9b", "#3f88e8", "#d94b48", "#f2a65a", "#f2a65a", "#f2a65a"]

features = [
    "[01] total_gen                 gate=0.8830",
    "[02] metered_load_mw           gate=0.6017",
    "[03] prelim_load_avg_hourly    gate=0.3172",
    "[04] total_pjm_rt_load_mwh     gate=0.3096",
]

fig = plt.figure(figsize=(12.4, 4.5), dpi=220)
gs = fig.add_gridspec(1, 2, width_ratios=[4.45, 1.45], wspace=0.08)
ax = fig.add_subplot(gs[0, 0])
ax_text = fig.add_subplot(gs[0, 1])

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.7)

selected = values[1]
ax.axhline(selected, color="#3f88e8", linestyle="--", linewidth=1.2, alpha=0.75)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.018,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.annotate(
    "$R^2$ collapses after removing #01\nnecessary inference source",
    xy=(2, values[2]),
    xytext=(2.50, 0.48),
    arrowprops=dict(arrowstyle="->", lw=1.1, color="#9f2f2d"),
    fontsize=8.8,
    color="#9f2f2d",
    ha="left",
    va="center",
)

ax.set_title("Non-redundancy Validation by DNN Test $R^2$", fontsize=13, pad=10)
ax.set_ylabel("Test $R^2$", fontsize=11)
ax.set_xlabel("Validation Setting", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.13)
ax.grid(axis="y", color="#b7b7b7", alpha=0.35, linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax_text.axis("off")
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)
ax_text.text(0.02, 0.93, "Selected feature IDs", fontsize=10.5, fontweight="bold")
ax_text.text(
    0.02,
    0.82,
    "\n".join(features),
    fontsize=8.7,
    family="monospace",
    va="top",
)
ax_text.text(
    0.02,
    0.22,
    "Sufficiency: Selected ~= Full\nNecessity: Drop #01 causes collapse",
    fontsize=9.5,
    color="#333333",
    va="top",
)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
