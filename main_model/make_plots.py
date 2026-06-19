"""
make_plots.py — Publication-quality figures for the DiseaseT5 project.

Adapted for multi-seed evaluation (seed 13 / 42 / 87) with optional
standard vs oracle comparison and NN top-1 baseline.

Reads:
  - {output_root}/output_resampler_rag_seed{N}/history.json
  - {output_root}/output_resampler_rag_seed{N}/test_final.json           (standard)
  - {output_root}/output_resampler_rag_seed{N}/test_final_oracle.json    (oracle)
  - The IU-Xray annotation JSON (for disease distribution)
  - Optionally, an NN top-1 preds JSON (for lexical-vs-clinical scatter)

Outputs (figures/ directory):
  - fig_disease_distribution.pdf    [Phase 1]
  - fig_training_curves.pdf         [Phase 1]  multi-seed overlay
  - fig_per_disease_f1.pdf          [Phase 2]  seeds + mean±SD
  - fig_lexical_vs_clinical.pdf     [Phase 2]  seed scatter + NN point
  - fig_seed_summary.pdf            [Phase 2]  comprehensive seed comparison table
  - qualitative_examples.md         [Phase 1]

Usage:
  # Default Colab paths:
  python make_plots.py

  # With explicit paths:
  python make_plots.py --output_root ./outputs \\
                       --annotation ./data/iu_xray/annotation.json \\
                       --seeds 13 42 87 \\
                       --figdir figures

  # Include NN top-1 baseline:
  python make_plots.py --nn_preds ./dumps/nn_top1.json

  # Oracle mode only:
  python make_plots.py --mode oracle

  # Standard + oracle comparison:
  python make_plots.py --mode both

Each figure is independent: if its source data file is missing, the script
prints a warning and continues with the others.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Global plot style — restrained, journal-friendly
# ============================================================

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "svg.fonttype":       "none",   # keep text as editable text in SVG
})

# Color palette — Morandi muted tones, journal-friendly
SEED_COLORS = {
    13:  "#7A9CAE",   # muted slate blue
    42:  "#C4A484",   # warm taupe
    87:  "#9BAE8C",   # sage green
}

MODE_COLORS = {
    "standard":  "#7A9CAE",   # muted slate blue
    "oracle":    "#C4A484",   # warm taupe
    "nn":        "#C9A24B",   # muted ochre / gold
}

COLORS = {
    "main":      "#3F6E7D",   # deep petrol teal — reserved for the headline model
    "train":     "#7A9CAE",   # muted slate blue
    "val":       "#C4A484",   # warm taupe
    "bleu":      "#7A9CAE",   # muted slate blue
    "f1":        "#B8938C",   # dusty rose
    "visual":    "#C17C66",   # muted terracotta — for the +visual_token ablation
    "normal":    "#D5CFC7",   # warm light gray
    "abnormal":  "#8B9D83",   # muted sage
    "mean_bar":  "#5D5C5C",   # charcoal gray
    "nn_preds":  "#C9A24B",   # muted ochre / gold (distinct from warm taupe)
    "qwen":      "#A0958B",   # stone gray
    "baseline":  "#B8938C",   # dusty rose
    "r2gen":     "#8C7B9B",   # muted mauve — task-specific RRG baseline (R2Gen)
}

DISEASE_NAMES = ["Cardiomegaly", "Edema", "Consolidation",
                 "Atelectasis", "Pleural Effusion"]


# ============================================================
# Helpers
# ============================================================

def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[skip] Missing file: {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[skip] Failed to load {path}: {e}")
        return None


def _ensure(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


SAVE_FORMATS = ["pdf", "png", "svg"]   # overridden by --formats


def _save(fig, figdir: str, name: str):
    out = os.path.join(figdir, name)
    written = []
    for fmt in SAVE_FORMATS:
        path = f"{out}.{fmt}"
        if fmt == "png":
            fig.savefig(path, dpi=200)   # raster: explicit dpi
        else:
            fig.savefig(path)            # pdf / svg are vector
        written.append(path)
    plt.close(fig)
    print(f"[ok] Wrote {', '.join(written)}")


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Mean ± SD; returns (0, 0) for empty list."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _avg_bleu_f1(paths: List[str]):
    """Mean (BLEU-4, CheXbert macro-F1(5)) over whichever seed files exist.
    Returns (mean_bleu, mean_f1, n_seeds) or (None, None, 0) if none load."""
    BLEU_KEYS = ["R2Gen_BLEU_4", "BLEU-4", "BLEU_4"]
    F1_KEYS = ["ClinicalF1_chexbert_macro_F_5", "CheXbert_macro_F_5",
               "ClinicalF1_chexbert_macro_F"]
    bs, fs = [], []
    for path in paths:
        d = _load_json(path)
        if d is None:
            continue
        m = d.get("metrics", {})
        b = next((m[k] for k in BLEU_KEYS if k in m), None)
        f1 = next((m[k] for k in F1_KEYS if k in m), None)
        if b is not None and f1 is not None:
            bs.append(float(b)); fs.append(float(f1))
    if not bs:
        return None, None, 0
    return float(np.mean(bs)), float(np.mean(fs)), len(bs)


# ============================================================
# Figure 1: Disease distribution across splits
# ============================================================

def fig_disease_distribution(annotation_path: str, figdir: str):
    """Bar chart of per-disease positive counts in train/val/test."""
    data = _load_json(annotation_path)
    if data is None:
        return

    try:
        from data_utils_compat import extract_disease_labels, DISEASE_LIST
    except ImportError:
        print("[warn] data_utils_compat not importable. Run this script from the "
              "project root, or add it to PYTHONPATH.")
        return

    splits = ["train", "val", "test"]
    counts: Dict[str, Dict[str, int]] = {s: {d: 0 for d in DISEASE_LIST} for s in splits}
    n_normal: Dict[str, int] = {s: 0 for s in splits}
    n_total:  Dict[str, int] = {s: 0 for s in splits}

    for split in splits:
        for item in data[split]:
            report = item.get("report", "").lower()
            found = extract_disease_labels(report)
            n_total[split] += 1
            if not found:
                n_normal[split] += 1
            for d in found:
                counts[split][d] += 1

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                             gridspec_kw={"width_ratios": [1.0, 2.2]})

    # Left: normal vs abnormal proportion stacked bar
    ax = axes[0]
    width = 0.55
    xs = np.arange(len(splits))
    n_ab = [n_total[s] - n_normal[s] for s in splits]
    ax.bar(xs, list(n_normal.values()), width, label="Normal",
           color=COLORS["normal"], edgecolor="white", linewidth=0.5)
    ax.bar(xs, n_ab, width, bottom=list(n_normal.values()), label="Abnormal",
           color=COLORS["abnormal"], edgecolor="white", linewidth=0.5)
    for i, s in enumerate(splits):
        pct = 100.0 * n_normal[s] / max(n_total[s], 1)
        ax.text(i, n_total[s] + max(n_total.values()) * 0.02,
                f"{pct:.0f}% N", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of samples")
    ax.set_title("(a) Normal vs. abnormal samples")
    ax.legend(frameon=False, loc="upper right")

    # Right: per-disease counts
    ax = axes[1]
    n_d = len(DISEASE_LIST)
    width = 0.26
    xs = np.arange(n_d)
    for i, s in enumerate(splits):
        vals = [counts[s][d] for d in DISEASE_LIST]
        ax.bar(xs + (i - 1) * width, vals, width, label=s.capitalize(),
               edgecolor="white", linewidth=0.4,
               color=["#8B9D83", "#7A9CAE", "#C4A484"][i])
    ax.set_xticks(xs)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DISEASE_LIST], fontsize=7)
    ax.set_ylabel("Positive cases")
    ax.set_title("(b) Per-disease positive counts (5 CheXpert pathologies)")
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    _save(fig, figdir, "fig_disease_distribution")

    print("\n--- Disease distribution summary ---")
    for s in splits:
        print(f"  {s}: total={n_total[s]}, normal={n_normal[s]} "
              f"({100*n_normal[s]/n_total[s]:.1f}%), "
              f"abnormal={n_total[s] - n_normal[s]}")
    print()


# ============================================================
# Figure 2: Training curves (multi-seed overlay)
# ============================================================

def fig_training_curves(history_paths: Dict[int, str], figdir: str,
                        primary_seed: Optional[int] = None):
    """
    Two-panel: (a) train loss + val BLEU-4 for a single representative seed.
               (b) val BLEU-4 vs val CheXbert F1 across epochs (mean ± band).
    history_paths: {seed: path_to_history.json}
    primary_seed:  which seed panel (a) shows; falls back to first available.
    """
    all_histories = {}
    for seed, path in history_paths.items():
        h = _load_json(path)
        if h is not None and h:
            all_histories[seed] = h

    if not all_histories:
        print("[skip] No history.json files found for any seed.")
        return

    # ---- Collect per-seed data ----
    seed_data = {}
    for seed, history in all_histories.items():
        epochs = [h["epoch"] for h in history]
        train_loss = [h.get("train_loss", np.nan) for h in history]
        val_bleu4 = [h.get("BLEU-4", np.nan) for h in history]

        f1_keys_priority = [
            "ClinicalF1_chexbert_macro_F_5",
            "ClinicalF1_chexbert_macro_F",
            "ClinicalF1_chexbert_micro_F_5",
        ]
        f1_key = None
        for k in f1_keys_priority:
            if any(k in h for h in history):
                f1_key = k
                break
        val_f1 = [h.get(f1_key, np.nan) for h in history] if f1_key else None
        seed_data[seed] = (epochs, train_loss, val_bleu4, val_f1, f1_key)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # ---- (a) Train loss + val BLEU-4 for a single representative seed ----
    ax = axes[0]
    psd = primary_seed if primary_seed in seed_data else sorted(seed_data.keys())[0]
    p_epochs, p_loss, p_bleu, _, _ = seed_data[psd]

    c_loss = COLORS["visual"]   # terracotta — training loss (left axis)
    c_bleu = COLORS["bleu"]     # slate blue — BLEU-4 (right axis), matches panel (b)

    ax.plot(p_epochs, p_loss, color=c_loss, linewidth=1.6, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss", color="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax.set_title(f"(a) Training loss & BLEU-4 (seed {psd})")

    ax2 = ax.twinx()
    ax2.plot(p_epochs, p_bleu, color=c_bleu, linestyle="--", linewidth=1.4,
             marker="s", markersize=3, label="Val BLEU-4")
    ax2.set_ylabel("Validation BLEU-4", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", frameon=False, fontsize=8)

    # ---- (b) Mean BLEU-4 vs CheXbert F1 across seeds ----
    ax = axes[1]
    # Align all to the shortest epoch length
    min_epochs = min(len(d[0]) for d in seed_data.values())
    bleu_matrix = []
    f1_matrix = []
    has_f1 = False
    for seed, (epochs, _, val_bleu4, val_f1, _) in seed_data.items():
        bleu_matrix.append(val_bleu4[:min_epochs])
        if val_f1 is not None:
            f1_matrix.append(val_f1[:min_epochs])
            has_f1 = True

    mean_bleu = np.mean(bleu_matrix, axis=0)
    std_bleu = np.std(bleu_matrix, axis=0, ddof=1 if len(bleu_matrix) > 1 else 0)

    xs = list(range(1, min_epochs + 1))
    ax.plot(xs, mean_bleu, color=COLORS["bleu"], label="BLEU-4 (mean)", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU-4", color="black")
    ax.tick_params(axis="y", labelcolor="black")

    if has_f1:
        mean_f1 = np.mean(f1_matrix, axis=0)
        std_f1 = np.std(f1_matrix, axis=0, ddof=1 if len(f1_matrix) > 1 else 0)
        ax2 = ax.twinx()
        ax2.plot(xs, mean_f1, color=COLORS["f1"], linestyle="--",
                 label="CheXbert F1 (mean)", linewidth=1.8)
        ax2.set_ylabel("CheXbert macro F1 (5)", color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax.set_title("(b) Lexical vs. clinical signal across epochs")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="lower right", frameon=False, fontsize=7)
    else:
        ax.set_title("(b) Validation BLEU-4 across epochs")
        ax.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    _save(fig, figdir, "fig_training_curves")


# ============================================================
# Figure 3: Per-disease F1 across seeds (with mean±SD)
# ============================================================

def fig_per_disease_f1(output_root: str, full_seeds: List[int],
                       oracle_seed: int,
                       ablation_paths: Dict[str, str],
                       figdir: str):
    """
    Grouped bar chart: 5 diseases × (Full + Ours(oracle cue) + ablations).
    Every variant is averaged over ``full_seeds``: Full and oracle read each
    per-seed test_final[_oracle].json; each ablation path carries a ``seedNN``
    token that is expanded across ``full_seeds`` and averaged. Figures show the
    seed mean only (no SD); per-seed SD belongs in the tables.
    ablation_paths: {label: path_to_one_seed's_test_final.json}
    """
    # ---- Collect per-variant per-disease F1 (each variant averaged over seeds) ----
    import re as _re
    variant_data: Dict[str, Dict[str, float]] = {}  # label -> {disease: mean f1}
    COLORS_VAR = {}                                  # label -> color
    variant_order = []

    def _row_from(path):
        d = _load_json(path)
        if d is None:
            return None
        m = d.get("metrics", {})
        return [m.get(f"CheXbert5_{dn}_F1", np.nan) for dn in DISEASE_NAMES]

    def _avg_over_seeds(paths):
        """Mean per-disease F1 over whichever seed paths exist."""
        rows = [r for r in (_row_from(p) for p in paths) if r is not None]
        if not rows:
            return None, 0
        mean = np.nanmean(np.array(rows, dtype=np.float64), axis=0)
        return {dn: float(mean[i]) for i, dn in enumerate(DISEASE_NAMES)}, len(rows)

    summary = []  # (label, {disease: f1}, n_seeds) for the console report

    # Full model: average test_final.json over full_seeds
    full_paths = [os.path.join(output_root, f"output_resampler_rag_seed{s}",
                               "test_final.json") for s in full_seeds]
    full_row, n_full = _avg_over_seeds(full_paths)
    if full_row is not None:
        variant_data["Full"] = full_row
        COLORS_VAR["Full"] = COLORS["mean_bar"]
        variant_order.append("Full")
        summary.append(("Full", full_row, n_full))
    else:
        print("[warn] No full model seed data found for per-disease F1.")

    # Ours (oracle cue): average test_final_oracle.json over full_seeds
    oracle_paths = [os.path.join(output_root, f"output_resampler_rag_seed{s}",
                                 "test_final_oracle.json") for s in full_seeds]
    oracle_row, n_oracle = _avg_over_seeds(oracle_paths)
    if oracle_row is not None:
        variant_data["Ours (oracle cue)"] = oracle_row
        COLORS_VAR["Ours (oracle cue)"] = "#B8938C"  # dusty rose
        variant_order.append("Ours (oracle cue)")
        summary.append(("Ours (oracle cue)", oracle_row, n_oracle))

    # Ablations: the supplied path carries a seedNN token; expand across full_seeds.
    ABLATION_COLORS = ["#8B9D83", "#7A9CAE", "#A0958B", "#D4A574", "#B8938C"]
    for i, (label, ab_path) in enumerate(ablation_paths.items()):
        if _re.search(r"seed\d+", ab_path):
            ab_paths = [_re.sub(r"seed\d+", f"seed{s}", ab_path) for s in full_seeds]
        else:
            ab_paths = [ab_path]   # no seed token -> use the single file as given
        ab_row, n_ab = _avg_over_seeds(ab_paths)
        if ab_row is None:
            print(f"[warn] No data found for ablation '{label}' "
                  f"(looked for {len(ab_paths)} seed file(s)).")
            continue
        variant_data[label] = ab_row
        COLORS_VAR[label] = ABLATION_COLORS[i % len(ABLATION_COLORS)]
        variant_order.append(label)
        summary.append((label, ab_row, n_ab))

    if not variant_data:
        print("[skip] No per-disease F1 data found.")
        return

    # ---- Console summary: seed-averaged per-class F1 + implied macro-F1(5) ----
    print("\n--- Per-disease F1 (seed-averaged) ---")
    print("  {:<18}".format("variant")
          + "".join(f"{d[:10]:>11}" for d in DISEASE_NAMES)
          + f"{'macroF1(5)':>12}{'#seeds':>8}")
    for label, row, n in summary:
        vals = [row[d] for d in DISEASE_NAMES]
        print("  {:<18}".format(label)
              + "".join(f"{v:>11.2f}" for v in vals)
              + f"{float(np.mean(vals)):>12.2f}{n:>8d}")
    print()

    # ---- Plot ----
    n_diseases = len(DISEASE_NAMES)
    n_variants = len(variant_data)
    width = 0.75 / n_variants
    xs = np.arange(n_diseases)

    fig, ax = plt.subplots(figsize=(9.0, 3.5))

    for i, label in enumerate(variant_order):
        row = [variant_data[label].get(d, 0) for d in DISEASE_NAMES]
        offset = (i - (n_variants - 1) / 2) * width
        c = COLORS_VAR.get(label, "#A0958B")
        ax.bar(xs + offset, row, width * 0.9,
               label=label, color=c, edgecolor="white", linewidth=0.4)

    # (Error bars removed: figures show seed means only; per-seed SD lives in the tables.)

    ax.set_xticks(xs)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DISEASE_NAMES], fontsize=8)
    ax.set_ylabel("CheXbert F1 (%)")
    ax.set_title("Per-disease CheXbert F1 — model variants")
    ax.legend(frameon=False, ncol=min(n_variants, 4), loc="upper center",
              bbox_to_anchor=(0.5, -0.20), fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, figdir, "fig_per_disease_f1")


# ============================================================
# Figure 4: Lexical vs clinical scatter
# ============================================================
#
# ── HARDCODED SCATTER POINTS ────────────────────────────────
# All points for fig_lexical_vs_clinical are listed here.
# No file reading is performed — edit bleu4 / f1 below directly.
#
#   label      : text shown next to the marker
#   bleu4      : x-axis value (BLEU-4, R2Gen protocol, %)
#   f1         : y-axis value (CheXbert macro F1 over 5 pathologies, %)
#   color_key  : must be a key in the COLORS dict above
#   error_x/y  : optional error bars (omit or set 0 for none)
#
# To hide a point: set its bleu4 or f1 to None, or delete the row.
# Marker shapes are assigned automatically in list order
# (the "main" / Ours point is emphasised regardless of shape).
#
# TODO: 数值为当前默认/fallback 占位，待新数据上传后逐行替换。
# ────────────────────────────────────────────────────────────
SCATTER_POINTS: List[Dict] = [
    # ── 主对比表数值（BLEU-4 / CheXbert-5 macro，± 取均值）──
    {"label": "NN Retrieval",      "bleu4": 10.69, "f1": 19.40, "color_key": "nn_preds"},
    {"label": "Qwen2-VL + LoRA",   "bleu4": 9.92,  "f1": 0.00,  "color_key": "qwen"},
    {"label": "R2Gen",             "bleu4": 11.61, "f1": 0.00,  "color_key": "r2gen"},
    {"label": "PromptMRG",         "bleu4": 10.77, "f1": 29.24, "color_key": "baseline"},
    {"label": "Qwen2-VL (oracle)", "bleu4": 14.19, "f1": 11.21, "color_key": "qwen"},
    {"label": "Ours (Full)",       "bleu4": 14.65, "f1": 19.71, "color_key": "main"},
    # ── 消融点（BLEU-4 / CheXbert-5 macro，± 取均值）──
    {"label": "−cls",              "bleu4": 14.67, "f1": 23.64, "color_key": "abnormal"},
    {"label": "−hint",             "bleu4": 9.12,  "f1": 9.70,  "color_key": "val"},
    {"label": "+visual-token",     "bleu4": 14.82, "f1": 11.08, "color_key": "visual"},
]


def fig_lexical_vs_clinical(output_root: str,
                             standard_seeds: List[int],
                             oracle_seeds: List[int],
                             mode: str,
                             nn_preds_path: Optional[str],
                             external_points: List[Dict], figdir: str):
    """
    Scatter plot: x = BLEU-4, y = CheXbert macro F1 (5).
    standard_seeds: seeds for standard mode points.
    oracle_seeds:   seeds for oracle mode points (e.g. only [42]).
    mode: "standard", "oracle", or "both".
    external_points: [{"label": str, "bleu4": float, "f1": float, "color_key": str}, ...]
    """
    if mode == "both":
        modes_to_run = ["standard", "oracle"]
    else:
        modes_to_run = [mode]

    seed_map = {"standard": standard_seeds, "oracle": oracle_seeds}
    points = []   # (label, bleu4, f1, color, marker, seed)

    for m in modes_to_run:
        sfx = "_oracle" if m == "oracle" else ""
        for seed in seed_map[m]:
            dirname = f"output_resampler_rag_seed{seed}"
            path = os.path.join(output_root, dirname, f"test_final{sfx}.json")
            data = _load_json(path)
            if data is None:
                continue
            metrics = data.get("metrics", {})
            b = metrics.get("R2Gen_BLEU_4") or metrics.get("BLEU-4")
            f1 = (metrics.get("ClinicalF1_chexbert_macro_F_5") or
                  metrics.get("ClinicalF1_chexbert_macro_F"))
            if b is None or f1 is None:
                print(f"[warn] Missing BLEU-4 or F1 in {path}")
                continue
            label = f"S{seed}" + (" (oracle)" if m == "oracle" else "")
            c = SEED_COLORS.get(seed, "#A0958B")
            marker = "D" if m == "oracle" else "o"
            points.append((label, b, f1, c, marker, seed))

    # --- External comparison points ---
    # NN Top-1 from preds file metrics (if available alongside)
    nn_point = None
    if nn_preds_path:
        nn_metrics_path = nn_preds_path.replace(".json", "_metrics.json")
        nn_m = _load_json(nn_metrics_path)
        if nn_m is not None:
            nn_b = nn_m.get("R2Gen_BLEU_4") or nn_m.get("BLEU-4")
            nn_f1 = (nn_m.get("CheXbert_macro_F_5") or
                     nn_m.get("ClinicalF1_chexbert_macro_F_5") or
                     nn_m.get("ClinicalF1_chexbert_macro_F"))
            if nn_b is not None and nn_f1 is not None:
                nn_point = ("NN Retrieval", nn_b, nn_f1, COLORS["nn_preds"], "s", None)

    if nn_point:
        points.append(nn_point)

    # Other external points (Qwen, Baseline, ablations, etc.)
    MARKERS = ["s", "P", "X", "v", "^", "D", "*"]  # cycle through for different models
    ext_labels_seen = set()
    for i, ep in enumerate(external_points):
        ck = ep.get("color_key", "nn_preds")
        marker = MARKERS[i % len(MARKERS)]
        ext_labels_seen.add(ep["label"])
        points.append((ep["label"], ep["bleu4"], ep["f1"],
                       COLORS.get(ck, "#A0958B"), marker, None,
                       ep.get("error_x", 0), ep.get("error_y", 0)))

    if not points:
        print("[skip] No data for lexical-vs-clinical plot.")
        return

    # --- collision-aware label placement ---
    # Labels read left-to-right, so two labels collide when their markers sit at
    # a similar height (y) and within roughly a label's horizontal span (x).
    # Default placement is up-right; conflicting labels cycle to other corners.
    coords = [(it[1], it[2]) for it in points]   # (bleu4, f1)
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    x_span = ((max(xs) - min(xs)) or 1.0) * 0.18   # horizontal reach of a label
    y_close = ((max(ys) - min(ys)) or 1.0) * 0.06  # "same row" threshold
    # (dx, dy, ha, va) offsets in points; index 0 = default up-right
    LABEL_DIRS = [(6, 4, "left", "bottom"),      # up-right
                  (-6, -9, "right", "top"),      # down-left
                  (6, -9, "left", "top"),        # down-right
                  (-6, 4, "right", "bottom")]    # up-left
    placements = []
    for i, (bx, fy) in enumerate(coords):
        k = sum(1 for j in range(i)
                if abs(coords[j][1] - fy) < y_close and abs(coords[j][0] - bx) < x_span)
        placements.append(LABEL_DIRS[k % len(LABEL_DIRS)])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for idx, item in enumerate(points):
        label, b, f1, c, marker, seed = item[:6]
        err_x, err_y = item[6] if len(item) > 6 else 0, item[7] if len(item) > 7 else 0
        is_main = (c == COLORS["main"])           # headline model gets emphasis
        s = 100 if seed is not None else 120       # larger for external models
        if is_main:
            s = 200
        ax.errorbar(b, f1, xerr=err_x, yerr=err_y,
                    fmt="none" if seed is not None else "none",
                    ecolor=c, alpha=0.5, capsize=4, linewidth=1.2, zorder=2)
        ax.scatter(b, f1, s=s, color=c, marker=marker, edgecolors="black",
                   linewidth=1.1 if is_main else 0.6,
                   zorder=4 if is_main else 3)
        dx, dy, ha, va = placements[idx]
        if label == "Ours (Full)":                # headline always up-right, nudged out
            dx, dy, ha, va = 12, 4, "left", "bottom"
        ax.annotate(label, (b, f1), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, ha=ha, va=va)

    ax.set_xlabel("BLEU-4 (R2Gen protocol, %)")
    ax.set_ylabel("CheXbert macro F1 over 5 pathologies (%)")
    ax.set_title("Lexical vs. clinical performance — multi-seed")
    ax.set_ylim(top=33)   # headroom so the top point (PromptMRG ~29) isn't clipped

    # Annotate
    ax.text(0.98, 0.02, "↑ better clinical content\n→ better lexical overlap",
            transform=ax.transAxes, fontsize=7, color="#6B6B6B",
            ha="right", va="bottom", style="italic")

    plt.tight_layout()
    _save(fig, figdir, "fig_lexical_vs_clinical")


# ============================================================
# Figure 5: Comprehensive seed summary table as a figure
# ============================================================

def fig_seed_summary(output_root: str, seeds: List[int], mode: str, figdir: str):
    """
    Render a publication-quality table figure comparing seeds across all metrics.
    Rows = metrics, Columns = seeds + Mean±SD.
    """
    suffix = "_oracle" if mode == "oracle" else ""

    all_metrics: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        dirname = f"output_resampler_rag_seed{seed}"
        path = os.path.join(output_root, dirname, f"test_final{suffix}.json")
        data = _load_json(path)
        if data is None:
            continue
        m = data.get("metrics", {})
        all_metrics[seed] = {k: v for k, v in m.items()
                             if isinstance(v, (int, float))}

    if not all_metrics:
        print("[skip] No seed data for summary table.")
        return

    # Define the rows we care about, in order
    metric_rows = [
        ("R2Gen_BLEU_4",        "BLEU-4 (R2Gen)"),
        ("R2Gen_METEOR",        "METEOR (R2Gen)"),
        ("R2Gen_ROUGE_L",       "ROUGE-L (R2Gen)"),
        ("CheXbert_accuracy",   "CheXbert accuracy"),
        ("CheXbert_macro_F_5",  "CheXbert macro F1 (5)"),
        ("CheXbert_micro_F_5",  "CheXbert micro F1 (5)"),
        ("CheXbert_macro_F_14", "CheXbert macro F1 (14)"),
        ("CheXbert_micro_F_14", "CheXbert micro F1 (14)"),
        ("RadGraph_Simple",     "RadGraph Simple"),
        ("RadGraph_Partial",    "RadGraph Partial"),
        ("RadGraph_Complete",   "RadGraph Complete"),
    ]

    # Collect values
    table = []  # list of (display_name, [seed_vals], mean, std)
    for key, display in metric_rows:
        vals = []
        for seed in sorted(all_metrics.keys()):
            v = all_metrics[seed].get(key, np.nan)
            vals.append(v)
        m, s = _mean_std(vals)
        table.append((display, vals, m, s))

    sorted_seeds = sorted(all_metrics.keys())
    n_seeds = len(sorted_seeds)
    n_rows = len(table)

    fig, ax = plt.subplots(figsize=(10.0, 0.35 * n_rows + 1.5))
    ax.axis("off")

    col_labels = [f"Seed {s}" for s in sorted_seeds] + ["Mean ± SD"]
    cell_text = []
    for display, vals, m, s in table:
        row = []
        for v in vals:
            row.append(f"{v:.1f}" if not np.isnan(v) else "—")
        row.append(f"{m:.1f} ± {s:.1f}")
        cell_text.append(row)

    row_labels = [display for display, _, _, _ in table]

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.35)

    # Style: bold header, alternating row colors
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D5CFC7")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#E8E4E0")
            cell.set_fontsize(8)
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F3F0")
        else:
            cell.set_facecolor("white")

        # Bold the mean column
        if col == n_seeds:
            cell.set_facecolor("#ECE8E3")
            if row > 0:
                cell.set_text_props(weight="bold")

    mode_label = {"standard": "Standard", "oracle": "Oracle",
                  "both": "Standard+Oracle"}.get(mode, mode)
    ax.set_title(f"Multi-seed evaluation summary — {mode_label} mode",
                 fontsize=10, fontweight="bold", pad=12)

    plt.tight_layout()
    _save(fig, figdir, f"fig_seed_summary_{mode}")


# ============================================================
# Figure 5 (§5.3): Predicted cue vs Oracle cue — the bottleneck figure
# ============================================================

# Each row: (display_name, predicted_mean, predicted_sd, oracle_mean, oracle_sd)
# sd = None when no multi-seed SD is available (per-disease F1).
PRED_VS_ORACLE_LEXICAL = [
    ("BLEU-4",       18.49, 0.23, 18.47, 0.33),
    ("METEOR",       20.25, 1.13, 20.31, 1.11),
    ("ROUGE-L",      37.59, 1.33, 37.84, 1.31),
    ("RadGraph-S",   41.67, 3.78, 42.21, 3.40),
    ("RadGraph-P",   38.53, 3.26, 39.10, 2.90),
]

PRED_VS_ORACLE_CLINICAL = [
    ("macro-F1(5)",  18.87, 1.65, 49.58, 14.25),
    ("micro-F1(5)",  30.31, 2.21, 79.60, 15.18),
    ("macro-F1(14)", 12.33, 0.59, 23.83,  5.86),
    ("micro-F1(14)", 55.10, 2.17, 67.59,  3.95),
    ("Cardiomegaly", 30.13, None, 93.39, None),
    ("Edema",         0.00, None, 16.67, None),
    ("Consolidation", 0.00, None,  0.00, None),
    ("Atelectasis",  28.32, None, 60.66, None),
    ("Pleural Eff.", 35.90, None, 77.17, None),
]


def _pvo_panel(ax, rows, c_full, c_oracle, title, annotate_delta=False):
    """Draw one grouped-bar panel: Full (predicted) vs Ours (oracle)."""
    n = len(rows)
    xs = np.arange(n)
    w = 0.38

    pred = [r[1] for r in rows]
    ora = [r[3] for r in rows]

    ax.bar(xs - w / 2, pred, w, label="Full (predicted cue)",
           color=c_full, edgecolor="white", linewidth=0.4)
    ax.bar(xs + w / 2, ora, w, label="Ours (oracle cue)",
           color=c_oracle, edgecolor="white", linewidth=0.4)

    # Figures show means only; per-seed SD is reported in the tables.
    ymax = max(max(pm, om) for _, pm, ps, om, os_ in rows)
    ax.set_ylim(0, ymax * 1.18)

    # Delta annotations (the punchline: how much oracle gains)
    if annotate_delta:
        for i, (_, pm, ps, om, os_) in enumerate(rows):
            d = om - pm
            top = max(pm, om)
            ax.text(xs[i], top + ymax * 0.02,
                    f"+{d:.0f}" if d >= 0 else f"{d:.0f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=c_oracle if d > 1 else "#9A9A9A", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Score (%)")
    ax.set_title(title, fontsize=9.5)


def fig_predicted_vs_oracle(figdir: str,
                            lexical=PRED_VS_ORACLE_LEXICAL,
                            clinical=PRED_VS_ORACLE_CLINICAL):
    """
    Two-panel grouped bar chart contrasting the predicted-cue model with the
    oracle-cue upper bound:
      (a) lexical/structural metrics — predicted ≈ oracle (no gap)
      (b) clinical F1 metrics + per-disease F1 — oracle ≫ predicted (the gap)
    The visual gap in (b) vs the flatness in (a) localises the bottleneck to
    the cue predictor rather than the generator.
    """
    c_full = COLORS["train"]   # slate blue — the deployed (predicted-cue) model
    c_oracle = COLORS["f1"]    # dusty rose — the oracle-cue upper bound

    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 3.8),
        gridspec_kw={"width_ratios": [len(lexical), len(clinical)]})

    _pvo_panel(axes[0], lexical, c_full, c_oracle,
               "(a) Lexical & structural — predicted ≈ oracle",
               annotate_delta=False)
    _pvo_panel(axes[1], clinical, c_full, c_oracle,
               "(b) Clinical — oracle ≫ predicted  (bottleneck = cue predictor)",
               annotate_delta=True)

    # Single shared legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, figdir, "fig_predicted_vs_oracle")


def fig_predicted_vs_oracle_dumbbell(figdir: str,
                                     lexical=PRED_VS_ORACLE_LEXICAL,
                                     clinical=PRED_VS_ORACLE_CLINICAL):
    """
    Dumbbell (connected-dot) version of the bottleneck figure.
    Each metric is one row: a dot at the predicted-cue score and a dot at the
    oracle-cue score, joined by a line whose length IS the gap. Lexical metrics
    are near-zero-length stubs; clinical metrics are long bars — making it
    visually obvious that the headroom is concentrated in clinical accuracy.
    """
    c_full = COLORS["train"]    # slate blue — predicted cue
    c_oracle = COLORS["f1"]     # dusty rose — oracle cue
    c_line = "#C9C2BA"          # warm light gray connector

    # Within each group, sort by gap so segment lengths read as a clean fan.
    lex = sorted(lexical,  key=lambda r: (r[3] - r[1]))
    clin = sorted(clinical, key=lambda r: (r[3] - r[1]))

    # Build row list top->bottom: lexical block, gap, clinical block.
    # (We assign descending y so the first item sits at the top.)
    groups = [("Lexical / structural", lex), ("Clinical (F1)", clin)]
    rows = []          # (name, pm, ps, om, os_, y)
    group_spans = []   # (label, y_top, y_bottom)
    y = 0.0
    for gi, (glabel, grows) in enumerate(groups):
        if gi > 0:
            y -= 1.2   # gap between groups
        y_top = y
        for r in grows:
            rows.append((r[0], r[1], r[2], r[3], r[4], y))
            y -= 1.0
        group_spans.append((glabel, y_top, y + 1.0))

    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    for name, pm, ps, om, os_, yy in rows:
        # connector
        ax.plot([pm, om], [yy, yy], color=c_line, linewidth=3.0,
                solid_capstyle="round", zorder=1)
        # (SD whiskers removed: means only; per-seed SD is in the tables.)
        # endpoints
        ax.scatter(pm, yy, s=42, color=c_full, edgecolors="white",
                   linewidth=0.6, zorder=3)
        ax.scatter(om, yy, s=42, color=c_oracle, edgecolors="white",
                   linewidth=0.6, zorder=3)
        # gap label at the right end
        d = om - pm
        ax.text(max(pm, om) + 1.5, yy, f"+{d:.0f}" if d >= 0 else f"{d:.0f}",
                va="center", ha="left", fontsize=7,
                color=c_oracle if d > 1 else "#9A9A9A",
                fontweight="bold" if d > 1 else "normal")

    # y ticks = metric names
    ax.set_yticks([r[5] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(rows[-1][5] - 0.8, rows[0][5] + 1.4)

    # group labels + faint band on the left
    for glabel, y_top, y_bot in group_spans:
        ax.text(-0.5, y_top + 0.9, glabel, fontsize=8.5, fontweight="bold",
                color="#5D5C5C", ha="left", va="bottom")

    ax.set_xlim(0, 108)
    ax.set_xlabel("Score (%)")
    ax.set_title("Predicted vs. oracle cue — gap concentrated in clinical metrics",
                 fontsize=10)

    # legend via proxy handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_full,
               markeredgecolor="white", markersize=8, label="Full (predicted cue)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_oracle,
               markeredgecolor="white", markersize=8, label="Ours (oracle cue)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    _save(fig, figdir, "fig_predicted_vs_oracle_dumbbell")


# ============================================================
# Qualitative example table
# ============================================================

def qualitative_examples(output_root: str, seeds: List[int], figdir: str,
                          n_normal: int = 1, n_abnormal: int = 2):
    """Pick cases from the first available seed's test_final.json."""
    # Use first seed that has data
    for seed in seeds:
        dirname = f"output_resampler_rag_seed{seed}"
        path = os.path.join(output_root, dirname, "test_final.json")
        data = _load_json(path)
        if data is not None:
            break
    else:
        print("[skip] No test_final.json found for any seed.")
        return

    try:
        from data_utils_compat import extract_disease_labels
    except ImportError:
        extract_disease_labels = lambda x: []

    candidates = []
    for s in data.get("samples", []):
        ref = s["reference"]
        pred = s["prediction"]
        ref_dis = set(extract_disease_labels(ref))
        pred_dis = set(extract_disease_labels(pred))
        is_normal = len(ref_dis) == 0
        score = 0
        if not is_normal:
            score += 2 * len(ref_dis & pred_dis)
            score -= len(pred_dis - ref_dis)
        candidates.append((score, is_normal, s))

    normals   = sorted([c for c in candidates if c[1]],     key=lambda x: -x[0])
    abnormals = sorted([c for c in candidates if not c[1]], key=lambda x: -x[0])

    picks = normals[:n_normal] + abnormals[:n_abnormal]
    if not picks:
        print("[skip] No samples found for qualitative table.")
        return

    lines = [
        "# Qualitative examples\n",
        f"Selected from seed {seed}, comparing model output against the reference.\n",
    ]
    for i, (score, is_normal, s) in enumerate(picks, 1):
        kind = "Normal" if is_normal else "Abnormal"
        lines.append(f"\n## Example {i} ({kind} case) — ID: {s['id']}\n")
        lines.append("| Source | Output |")
        lines.append("|---|---|")
        lines.append(f"| **Reference** | {s['reference']} |")
        lines.append(f"| **Model** | {s['prediction']} |")

    out_path = os.path.join(figdir, "qualitative_examples.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[ok] Wrote {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality figures for DiseaseT5 (multi-seed)")
    parser.add_argument("--output_root",
                        default=os.environ.get("OUTPUT_ROOT", "./outputs"),
                        help="Directory containing output_resampler_rag_seed{N}/ subdirs")
    parser.add_argument("--annotation",
                        default=os.environ.get("IU_XRAY_ANNOTATION", "./data/iu_xray/annotation.json"),
                        help="IU-Xray annotation JSON")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 87],
                        help="Seeds for standard mode (default: 13 42 87)")
    parser.add_argument("--primary_seed", type=int, default=42,
                        help="Seed shown in training-curve panel (a) (default: 42)")
    parser.add_argument("--oracle_seeds", type=int, nargs="+", default=[42],
                        help="Seeds for oracle mode (default: 42 only)")
    parser.add_argument("--mode", default="both",
                        choices=["standard", "oracle", "both"],
                        help="Evaluation mode: standard, oracle, or both")
    parser.add_argument("--nn_preds", default=None,
                        help="Path to NN top-1 preds JSON for comparison point")
    parser.add_argument("--nn_bleu", type=float, default=12.87,
                        help="NN Top-1 BLEU-4 (R2Gen protocol). Default from eval.")
    parser.add_argument("--nn_f1", type=float, default=17.73,
                        help="NN Top-1 CheXbert macro F1 (5). Default from eval.")
    parser.add_argument("--hide_nn", action="store_true",
                        help="Hide the built-in NN Top-1 point.")
    parser.add_argument("--qwen_bleu", type=float, default=12.92,
                        help="Qwen2-VL+LoRA (standard) BLEU-4. Set to -1 to hide.")
    parser.add_argument("--qwen_f1", type=float, default=0.0,
                        help="Qwen2-VL+LoRA (standard) CheXbert macro F1 (5). Set to -1 to hide.")
    parser.add_argument("--qwen_oracle_bleu", type=float, default=20.06,
                        help="Qwen2-VL+LoRA Oracle BLEU-4. Set to -1 to hide.")
    parser.add_argument("--qwen_oracle_f1", type=float, default=11.11,
                        help="Qwen2-VL+LoRA Oracle CheXbert macro F1 (5). Set to -1 to hide.")
    parser.add_argument("--hide_qwen", action="store_true",
                        help="Hide all built-in Qwen points.")
    parser.add_argument("--hide_qwen_oracle", action="store_true",
                        help="Hide the built-in Qwen Oracle point.")
    parser.add_argument("--r2gen_bleu", type=float, default=17.40,
                        help="R2Gen (official ckpt) BLEU-4 for the scatter point. Set to -1 to hide.")
    parser.add_argument("--r2gen_f1", type=float, default=0.0,
                        help="R2Gen (official ckpt) CheXbert macro F1 (5) for the scatter point. Set to -1 to hide.")
    parser.add_argument("--hide_r2gen", action="store_true",
                        help="Hide the built-in R2Gen point.")
    parser.add_argument("--baseline_bleu", type=float, default=-1,
                        help="Pure Baseline BLEU-4. Default: hidden.")
    parser.add_argument("--baseline_f1", type=float, default=-1,
                        help="Pure Baseline CheXbert macro F1 (5). Default: hidden.")
    parser.add_argument("--hide_ablations", action="store_true",
                        help="Hide the built-in ablation points (--cls_loss, --hint, +visual).")
    parser.add_argument("--ablation_paths", default=(
                             "−cls=./outputs/"
                             "output_resampler_rag_noCls_seed42/test_final.json,"
                             "−hint=./outputs/"
                             "output_resampler_rag_noHint_seed42/test_final.json,"
                             "+visual-token=./outputs/"
                             "output_visual_token_seed42/test_final.json"),
                        help="Comma-separated label=path pairs for ablation test_final.json.")
    parser.add_argument("--extra", default=None,
                        help="Path to JSON file with extra scatter points:\n"
                             "[{\"label\":\"...\", \"bleu4\":N, \"f1\":N, \"color_key\":\"...\"}, ...]\n"
                             "color_key must be a key in the COLORS dict (e.g. main, nn_preds, qwen, baseline, bleu, f1, visual, abnormal)")
    parser.add_argument("--figdir", default="figures",
                        help="Output directory for figures")
    parser.add_argument("--formats", default="pdf,png,svg",
                        help="Comma-separated output formats (e.g. 'svg' or 'pdf,svg'). "
                             "Default: pdf,png,svg")
    parser.add_argument("--predoracle_style", default="bar",
                        choices=["bar", "dumbbell", "both"],
                        help="Style for the predicted-vs-oracle figure (default: bar)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["disease", "training", "perdisease",
                                 "scatter", "summary", "qualitative",
                                 "predoracle"],
                        help="Subset of figures to skip")
    args = parser.parse_args()

    global SAVE_FORMATS
    valid_fmts = {"pdf", "png", "svg", "eps", "jpg", "tiff"}
    fmts = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in fmts if f not in valid_fmts]
    if bad:
        print(f"[warn] Ignoring unknown format(s): {bad}. "
              f"Valid: {sorted(valid_fmts)}")
    fmts = [f for f in fmts if f in valid_fmts]
    if fmts:
        SAVE_FORMATS = fmts
    print(f"[info] Output formats: {SAVE_FORMATS}")

    _ensure(args.figdir)

    # Parse ablation 'label=path' pairs once (used by per-disease F1 and the scatter).
    ablation_paths: Dict[str, str] = {}
    if args.ablation_paths:
        for pair in args.ablation_paths.split(","):
            label, path = pair.split("=", 1)
            ablation_paths[label.strip()] = path.strip()

    # ---- Phase 1 ----
    if "disease" not in args.skip:
        print("\n=== Fig: disease distribution ===")
        fig_disease_distribution(args.annotation, args.figdir)

    if "training" not in args.skip:
        print("\n=== Fig: training curves (multi-seed overlay) ===")
        history_paths = {}
        for seed in args.seeds:
            dirname = f"output_resampler_rag_seed{seed}"
            p = os.path.join(args.output_root, dirname, "history.json")
            if os.path.exists(p):
                history_paths[seed] = p
        if history_paths:
            fig_training_curves(history_paths, args.figdir, args.primary_seed)
        else:
            print("[skip] No history.json found for any seed.")

    if "qualitative" not in args.skip:
        print("\n=== Qualitative examples ===")
        qualitative_examples(args.output_root, args.seeds, args.figdir)

    # ---- Phase 2 ----
    if "perdisease" not in args.skip:
        print("\n=== Fig: per-disease F1 ===")
        if ablation_paths:
            print(f"[info] Ablation paths: {list(ablation_paths.keys())}")
        else:
            print("[info] No --ablation_paths given; showing Full+Oracle only.")

        fig_per_disease_f1(args.output_root, args.seeds,
                           args.oracle_seeds[0] if args.oracle_seeds else 42,
                           ablation_paths, args.figdir)

    if "scatter" not in args.skip:
        print("\n=== Fig: lexical vs clinical scatter ===")
        # All scatter points are hardcoded in SCATTER_POINTS (top of the
        # "Figure 4" section). No test_final.json / metrics files are read.
        # Edit values there directly; a point is skipped if bleu4 or f1 is None.
        external_points = []
        for p in SCATTER_POINTS:
            if p.get("bleu4") is None or p.get("f1") is None:
                continue
            ep = dict(p)                      # copy so we never mutate the source list
            external_points.append(ep)
            print(f"[info] {ep['label']} scatter point: "
                  f"BLEU {ep['bleu4']}, macroF1(5) {ep['f1']} (hardcoded).")

        # Optional: still allow --extra JSON to append more points on top of
        # the hardcoded list. Leave --extra unset to use the hardcoded set only.
        if args.extra:
            extra_data = _load_json(args.extra)
            if extra_data is not None:
                for ep in extra_data:
                    external_points.append({
                        "label": ep["label"],
                        "bleu4": ep["bleu4"],
                        "f1": ep["f1"],
                        "color_key": ep.get("color_key", "nn_preds"),
                    })
                print(f"[info] Loaded {len(extra_data)} extra points from {args.extra}")

        # Per-seed standard/oracle points are intentionally not drawn here
        # ([], []): the seed means are baked into SCATTER_POINTS instead.
        # nn_preds_path=None so the function's internal NN logic stays off
        # (the NN point now lives in SCATTER_POINTS).
        fig_lexical_vs_clinical(args.output_root,
                                 [], [], args.mode,
                                 None, external_points, args.figdir)

    if "summary" not in args.skip:
        modes_to_run = ["standard", "oracle"] if args.mode == "both" else [args.mode]
        for m in modes_to_run:
            print(f"\n=== Fig: seed summary table ({m}) ===")
            fig_seed_summary(args.output_root, args.seeds, m, args.figdir)

    if "predoracle" not in args.skip:
        print("\n=== Fig: predicted vs oracle cue (bottleneck) ===")
        if args.predoracle_style in ("bar", "both"):
            fig_predicted_vs_oracle(args.figdir)
        if args.predoracle_style in ("dumbbell", "both"):
            fig_predicted_vs_oracle_dumbbell(args.figdir)

    print("\nDone. Figures written to:", os.path.abspath(args.figdir))


if __name__ == "__main__":
    main()
