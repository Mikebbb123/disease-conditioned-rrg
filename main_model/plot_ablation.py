#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Summary Bar Chart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── DATA ──────────────────────────────────────────────────────────────────
configs = ["Full", "−hint", "−cls", "+visual-token"]

metrics = {
    "BLEU-4":          ([18.49, 10.34, 18.32, 17.92], [0.23, 0.66, 0.20, 0.76]),
    "micro-F1(5)":     ([30.31, 14.90, 28.82, 20.20], [2.21, 0.00, 2.17, 9.14]),
    "macro-F1(5)":     ([18.87,  6.05, 22.39, 10.17], [1.65, 0.00, 5.72, 6.41]),
    "RadGraph-Simple": ([41.67, 28.76, 39.38, 39.77], [3.78, 4.26, 1.65, 1.76]),
}

# ── PALETTE ───────────────────────────────────────────────────────────────
# Four clearly distinct hues, all desaturated to a sophisticated level.
# Hue order: blue · coral · sage · amber  (cool-warm alternation aids separation)
palette = {
    "bar":   ["#6E8FAD", "#C07B72", "#7FA882", "#C9A55C"],
    "alpha_bar": 0.88,
    "edge":  ["#4A6A84", "#9A5048", "#4E8055", "#A07C2A"],  # darker edge per colour
    "error": "#44444466",
    "label_col": ["#000000", "#000000", "#000000", "#000000"],
    "bg":    "#FFFFFF",
    "grid":  "#E8E8E8",
    "spine": "#CCCCCC",
    "text":  "#000000",
    "title": "#000000",
}

# ── STYLE SETUP ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10.5,
    "axes.edgecolor":   palette["spine"],
    "axes.linewidth":   0.7,
    "text.color":       palette["text"],
    "axes.labelcolor":  palette["text"],
    "xtick.color":      palette["text"],
    "ytick.color":      palette["text"],
    "figure.facecolor": palette["bg"],
    "axes.facecolor":   palette["bg"],
})

# ── LAYOUT ────────────────────────────────────────────────────────────────
metric_names = list(metrics.keys())
n_groups = len(configs)
n_series = len(metric_names)

x       = np.arange(n_groups)
bar_w   = 0.20
gap     = 0.035                        # extra gap between metric clusters
offsets = (np.arange(n_series) - (n_series - 1) / 2) * (bar_w + gap)

fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=300)
fig.patch.set_facecolor(palette["bg"])

# ── BARS ──────────────────────────────────────────────────────────────────
for i, m in enumerate(metric_names):
    means = np.array(metrics[m][0], dtype=float)
    sds   = np.array(metrics[m][1], dtype=float)
    xpos  = x + offsets[i]

    ax.bar(
        xpos, means, width=bar_w,
        color=palette["bar"][i],
        edgecolor=palette["edge"][i],
        linewidth=0.8,
        alpha=palette["alpha_bar"],
        zorder=3,
    )

    # error bars – thin, dark, no fill
    ax.errorbar(
        xpos, means, yerr=sds,
        fmt="none",
        ecolor=palette["edge"][i],
        elinewidth=1.0,
        capsize=3.0, capthick=1.0,
        zorder=4,
    )

    # value labels – colour-matched to each series, offset above error cap
    for xi, mu, sd in zip(xpos, means, sds):
        ax.text(
            xi, mu + sd + 1.1,
            f"{mu:.1f}",
            ha="center", va="bottom",
            fontsize=6.8,
            color=palette["label_col"][i],
            fontweight="semibold",
            zorder=5,
        )

# ── AXES COSMETICS ────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11, fontweight="medium")
ax.set_ylabel("Score", fontsize=10.5, labelpad=8)
ax.set_ylim(0, 52)
ax.set_xlim(-0.6, n_groups - 0.4)

# refined grid: major every 10, minor every 5
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.grid(True, which="major", color=palette["grid"], linewidth=0.9, zorder=0)
ax.yaxis.grid(True, which="minor", color=palette["grid"], linewidth=0.4,
              linestyle=":", zorder=0)
ax.set_axisbelow(True)

# remove unnecessary spines
for sp in ["top", "right", "bottom"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(palette["spine"])
ax.tick_params(axis="x", length=0, pad=6)
ax.tick_params(axis="y", length=0)

# ── TITLE ─────────────────────────────────────────────────────────────────
ax.set_title(
    "Ablation Summary Bar Chart",
    fontsize=14, fontweight="normal",
    color=palette["title"], pad=14, loc="center",
)

# ── LEGEND AT BOTTOM ─────────────────────────────────────────────────────
# Custom handles so patch colour matches bar exactly
handles = [
    mpatches.Patch(
        facecolor=palette["bar"][i],
        edgecolor=palette["edge"][i],
        linewidth=0.8,
        alpha=palette["alpha_bar"],
        label=m,
    )
    for i, m in enumerate(metric_names)
]

ax.legend(
    handles=handles,
    title="Metric",
    ncol=4,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    bbox_transform=ax.transAxes,
    handlelength=1.4,
    handleheight=1.0,
    columnspacing=1.6,
    fontsize=9.5,
    title_fontsize=9.5,
)

# ── EXPORT ────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.subplots_adjust(bottom=0.17)

out_png = "ablation_summary.png"
out_pdf = "ablation_summary.pdf"
plt.savefig(out_png, bbox_inches="tight", dpi=1000, facecolor=palette["bg"])
plt.savefig(out_pdf, bbox_inches="tight", facecolor=palette["bg"])
print("Saved:", out_png)
print("Saved:", out_pdf)
