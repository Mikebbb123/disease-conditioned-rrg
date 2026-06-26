#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Summary Bar Chart – v3
Colour palette & style matched to reference image.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── DATA ──────────────────────────────────────────────────────────────────
# Protocol: IU-Xray 590-sample TEST split, R2Gen NLG protocol, mean ± SD over
# 3 seeds (42, 13, 87). The "Full" column matches the README headline /
# comparison tables exactly (BLEU-4 18.49, micro-F1(5) 30.31, macro-F1(5)
# 18.87, RadGraph-Simple 41.67) and the +visual-token BLEU-4 (17.92) matches
# README line 31. These are NOT the 349 common-subset numbers used by
# make_plots.py's SCATTER_POINTS — do not cross-compare the two.
# (The −hint F1 SDs of 0.00 are correct, not placeholders.)
configs = ["Full", "−hint", "−cls", "+visual-token"]

metrics = {
    "BLEU-4":          ([18.49, 10.34, 18.32, 17.92], [0.23, 0.66, 0.20, 0.76]),
    "micro-F1(5)":     ([30.31, 14.90, 28.82, 20.20], [2.21, 0.00, 2.17, 9.14]),
    "macro-F1(5)":     ([18.87,  6.05, 22.39, 10.17], [1.65, 0.00, 5.72, 6.41]),
    "RadGraph-Simple": ([41.67, 28.76, 39.38, 39.77], [3.78, 4.26, 1.65, 1.76]),
}

# ── PALETTE  (pixel-sampled from reference image) ─────────────────────────
#   dusty rose · sage green · muted steel-blue · warm taupe
BAR_COLORS  = ["#B8928B", "#8B9D83", "#7A9CAE", "#B4ACA4"]
ERR_COLOR   = "#555555"
TEXT_COLOR  = "#2A2A2A"
GRID_COLOR  = "#EBEBEB"

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10.5,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#CCCCCC",
    "axes.linewidth":   0.6,
    "text.color":       TEXT_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
})

# ── LAYOUT ────────────────────────────────────────────────────────────────
metric_names = list(metrics.keys())
n_groups = len(configs)
n_series = len(metric_names)

x       = np.arange(n_groups)
bar_w   = 0.19
gap     = 0.025
offsets = (np.arange(n_series) - (n_series - 1) / 2) * (bar_w + gap)

fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=300)

# ── BARS ──────────────────────────────────────────────────────────────────
for i, m in enumerate(metric_names):
    means = np.array(metrics[m][0], dtype=float)
    sds   = np.array(metrics[m][1], dtype=float)
    xpos  = x + offsets[i]

    # flat, borderless bars — exactly like reference
    ax.bar(
        xpos, means, width=bar_w,
        color=BAR_COLORS[i],
        edgecolor="none",       # ← no outline, matches reference style
        zorder=3,
    )

    # error bars: thin, subdued
    ax.errorbar(
        xpos, means, yerr=sds,
        fmt="none",
        ecolor=ERR_COLOR,
        elinewidth=0.9,
        capsize=2.5, capthick=0.9,
        zorder=4,
    )

    # value labels
    for xi, mu, sd in zip(xpos, means, sds):
        ax.text(
            xi, mu + sd + 1.0,
            f"{mu:.1f}",
            ha="center", va="bottom",
            fontsize=6.5,
            color=TEXT_COLOR,
            zorder=5,
        )

# ── AXES ──────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11)
ax.set_ylabel("Score", fontsize=10.5, labelpad=8)
ax.set_ylim(0, 52)
ax.set_xlim(-0.6, n_groups - 0.4)

# grid: major every 10, minor every 5 — matching reference lightness
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=0.8, zorder=0)
ax.yaxis.grid(True, which="minor", color=GRID_COLOR, linewidth=0.4,
              linestyle=":", zorder=0)
ax.set_axisbelow(True)

# spines: keep only bottom + left (reference style)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID_COLOR)
ax.spines["bottom"].set_color(GRID_COLOR)
ax.tick_params(axis="x", length=0, pad=6)
ax.tick_params(axis="y", length=0)

# ── TITLE ─────────────────────────────────────────────────────────────────
ax.set_title(
    "Ablation Summary Bar Chart",
    fontsize=13, fontweight="normal",
    color=TEXT_COLOR, pad=12, loc="center",
)

# ── LEGEND AT BOTTOM ─────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=BAR_COLORS[i], edgecolor="none", label=m)
    for i, m in enumerate(metric_names)
]
ax.legend(
    handles=handles,
    title="Metric",
    ncol=4,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    bbox_transform=ax.transAxes,
    handlelength=1.4, handleheight=1.0,
    columnspacing=1.6,
    fontsize=9.5, title_fontsize=9.5,
)

# ── EXPORT ────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.subplots_adjust(bottom=0.17)

import argparse
_parser = argparse.ArgumentParser(description="Ablation summary bar chart.")
_parser.add_argument("--figdir", default="figures",
                     help="Output directory (matches make_plots.py; default: figures/).")
_args, _ = _parser.parse_known_args()
os.makedirs(_args.figdir, exist_ok=True)

out_png = os.path.join(_args.figdir, "fig_ablation_summary.png")
out_pdf = os.path.join(_args.figdir, "fig_ablation_summary.pdf")
plt.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")  # 300 ~ make_plots.py
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")           # vector
print("Saved:", out_png)
print("Saved:", out_pdf)
