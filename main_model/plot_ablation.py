#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation summary bar chart for Section 5.2 (Table 2 data).

Grouped bars: 4 configurations x 4 representative metrics, with +/- SD error bars.
Morandi (muted, low-saturation) colour palette.

Run:  python plot_ablation.py
Output: ablation_summary.png  and  ablation_summary.pdf  (same folder)

Only dependency: matplotlib + numpy
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401 (kept for easy font tweaking)

# ----------------------------------------------------------------------
# DATA (from Table 2; mean +/- SD over 3 seeds). Edit here if needed.
# ----------------------------------------------------------------------
configs = ["Full", "-hint", "-cls", "+visual-token"]

# metric_name -> (means[4], sds[4])  in the order of `configs`
metrics = {
    "BLEU-4":           ([18.49, 10.34, 18.32, 17.92], [0.23, 0.66, 0.20, 0.76]),
    "micro-F1(5)":      ([30.31, 14.90, 28.82, 20.20], [2.21, 0.00, 2.17, 9.14]),
    "macro-F1(5)":      ([18.87,  6.05, 22.39, 10.17], [1.65, 0.00, 5.72, 6.41]),
    "RadGraph-Simple":  ([41.67, 28.76, 39.38, 39.77], [3.78, 4.26, 1.65, 1.76]),
}

# ----------------------------------------------------------------------
# Morandi palette (muted / greyed pastels) — one colour per metric series
# ----------------------------------------------------------------------
morandi = [
    "#A7B5A0",  # sage green
    "#C9A9A6",  # dusty rose
    "#9FAEC0",  # muted blue-grey
    "#D8C3A5",  # warm sand
]
edge_col = "#5A5A5A"   # soft dark grey for outlines / error bars
text_col = "#3A3A3A"

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",          # change to "sans-serif" if you prefer
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "text.color": text_col,
    "axes.labelcolor": text_col,
    "xtick.color": text_col,
    "ytick.color": text_col,
})

metric_names = list(metrics.keys())
n_groups = len(configs)        # 4 configurations (x positions)
n_series = len(metric_names)   # 4 metrics (bars per group)

x = np.arange(n_groups)
bar_w = 0.19
offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)

for i, m in enumerate(metric_names):
    means, sds = metrics[m]
    means = np.array(means, dtype=float)
    sds = np.array(sds, dtype=float)
    bars = ax.bar(
        x + offsets[i], means, width=bar_w,
        label=m, color=morandi[i],
        edgecolor=edge_col, linewidth=0.7, zorder=3,
    )
    # error bars (SD). caps small; lighter grey.
    ax.errorbar(
        x + offsets[i], means, yerr=sds,
        fmt="none", ecolor=edge_col, elinewidth=0.9,
        capsize=2.5, capthick=0.9, zorder=4,
    )
    # value labels on top of each bar
    for xi, mu, sd in zip(x + offsets[i], means, sds):
        ax.text(xi, mu + sd + 0.8, f"{mu:.1f}",
                ha="center", va="bottom", fontsize=7.0, color=text_col, zorder=5)

# axes cosmetics
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Score")
ax.set_ylim(0, 50)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#E2E2E2", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    title="Metric", ncol=4, frameon=False,
    loc="upper center", bbox_to_anchor=(0.5, 1.13),
    handlelength=1.2, columnspacing=1.4, fontsize=9.5, title_fontsize=9.5,
)

plt.tight_layout()
plt.savefig("ablation_summary.png", bbox_inches="tight", dpi=300)
plt.savefig("ablation_summary.pdf", bbox_inches="tight")
print("Saved: ablation_summary.png / ablation_summary.pdf")
# plt.show()  # uncomment to preview interactively
