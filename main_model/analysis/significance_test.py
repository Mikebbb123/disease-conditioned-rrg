"""
Significance + effect size for the headline comparisons (Sect. 5 / Table 1b).

Primary tool: PAIRED bootstrap over test studies.
  - Single-run baselines (R2Gen, PromptMRG, Qwen, NN) have no seed variance, but
    we CAN resample test studies. This needs no retraining.
  - F1 is a corpus-level metric: we resample studies and recompute micro-F1 on the
    resample. Same resampled indices for both methods => the comparison is PAIRED,
    which is what your "paired X-ray" wording should cash out to.
  - Output per comparison: delta F1, 95% bootstrap CI, two-sided bootstrap p.
    The CI on the difference IS the (absolute) effect size.

Secondary: Wilcoxon signed-rank + Cliff's delta on per-study metrics (RadGraph,
sentence-level BLEU) that you already compute per study.

DOES NOT invent numbers. Run it where YOUR_DUMP / BASELINE_DUMPS live.

HONESTY NOTES to keep in the paper:
  * The bootstrap CI reflects TEST-SET sampling only. Baselines are single
    checkpoints, so it does NOT include baseline retraining variance. State this.
  * Do NOT bootstrap per-class Edema/Consolidation F1 (n=5/4 on the subset):
    underpowered, CIs span [0,100]. Test micro-F1 and the well-represented classes.
  * If you test many pairs, apply Holm correction or report CIs as primary.
"""
import os, sys, json
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

# ----- reuse the exact paths/conventions from chexbert_eval.py -----
YOUR_DUMP = "./dumps/ours_seed42_test_final.json"
BASELINE_DUMPS = {
    "R2Gen":     "./dumps/r2gen_test_generated.json",
    "PromptMRG": "./dumps/promptmrg_test_generated.json",
    "42": "./dumps/ours_seed42_test_final.json",
    "87": "./dumps/ours_seed87_test_final.json",
    "13": "./dumps/ours_seed13_test_final.json",
    "NN": "./dumps/nn_top1_test_generated.json",
}
B = 2000           # bootstrap resamples
SEED = 0
EXPECTED_OURS_MICRO = 31.61   # your Table-1b point estimate on the n=349 subset,
                              # used ONLY as a self-check that label extraction is right.
# -------------------------------------------------------------------

CHEX5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def _load_pred_by_id(path):
    obj = json.load(open(path))
    recs = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    chosen = {}
    for d in recs:
        sid = d["id"]; pred = d.get("generated", d.get("prediction"))
        vw = (d.get("view") or "").lower()
        frontal = any(t in vw for t in ("pa", "ap", "frontal"))
        if sid not in chosen or (frontal and not chosen[sid][1]):
            chosen[sid] = (pred, frontal)
    return {k: v[0] for k, v in chosen.items()}


def label_matrix(reports, labeler):
    """Per-study 5-class binary matrix (N,5), positive iff CheXbert label == 1.
    Uses f1chexbert.get_label; if your version differs, print(dir(labeler))."""
    names5 = getattr(labeler, "target_names_5",
                     ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"])
    names14 = getattr(labeler, "target_names", None)
    M = np.zeros((len(reports), 5), dtype=int)
    for i, r in enumerate(reports):
        lab = labeler.get_label(r if r else "")          # 14-dim, CheXbert order
        lab = list(lab.values()) if isinstance(lab, dict) else list(lab)
        if names14 and len(lab) == len(names14):
            d = dict(zip(names14, lab))
            vec = [d.get(c, 0) for c in CHEX5]
        else:                                             # already 5-dim
            d = dict(zip(names5, lab)); vec = [d.get(c, 0) for c in CHEX5]
        M[i] = [1 if v == 1 else 0 for v in vec]
    return M


def paired_bootstrap_micro(y_true, yp_a, yp_b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    pt = (f1_score(y_true, yp_a, average="micro", zero_division=0)
          - f1_score(y_true, yp_b, average="micro", zero_division=0)) * 100
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, N)                       # SAME idx for both => paired
        diffs[b] = (f1_score(y_true[idx], yp_a[idx], average="micro", zero_division=0)
                    - f1_score(y_true[idx], yp_b[idx], average="micro", zero_division=0)) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())  # two-sided bootstrap p
    return pt, lo, hi, min(p, 1.0)


def wilcoxon_cliffs(a, b):
    """Per-study paired test + non-parametric effect size for a per-study metric."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        _, p = wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        p = 1.0
    gt = np.sum(a[:, None] > b[None, :]); lt = np.sum(a[:, None] < b[None, :])
    cliff = (gt - lt) / (len(a) * len(b))                  # in [-1,1]
    return p, cliff


# ---- load refs + methods on the common subset ----
your = json.load(open(YOUR_DUMP))
ref_by_id = {s["id"]: s["reference"]  for s in your["samples"]}
methods = {"Ours": {s["id"]: s["prediction"] for s in your["samples"]}}
for n, p in BASELINE_DUMPS.items():
    if os.path.exists(p): methods[n] = _load_pred_by_id(p)
    else: print(f"[stat] missing {n}: {p}")

common = set(ref_by_id)
for d in methods.values(): common &= set(d)
common = [i for i in (s["id"] for s in your["samples"]) if i in common]
print(f"[stat] common subset n = {len(common)} over {list(methods)}")
references = [ref_by_id[i] for i in common]

from chexbert_eval import get_chexbert_labeler, _undo_tokenizer_patch
labeler = get_chexbert_labeler(device="cuda")

Y = {"_ref": label_matrix(references, labeler)}
for n in methods:
    Y[n] = label_matrix([methods[n][i] for i in common], labeler)
_undo_tokenizer_patch()

# ---- SELF-CHECK: recomputed point estimate must match your Table 1b ----
ours_micro = f1_score(Y["_ref"], Y["Ours"], average="micro", zero_division=0) * 100
print(f"[stat] self-check Ours micro-F1 = {ours_micro:.2f} "
      f"(expected ~{EXPECTED_OURS_MICRO}). If far off, label extraction/binarization "
      f"is wrong -- FIX before trusting CIs.")

# ---- paired bootstrap on the headline F1 gaps ----
pairs = [("Ours", "PromptMRG"), ("Ours", "R2Gen"), ("Ours", "NN"),
         ("PromptMRG", "R2Gen")]
print(f"\n=== Paired bootstrap, CheXbert-5 micro-F1 difference (B={B}, n={len(common)}) ===")
print(f"{'A vs B':22s}{'dF1':>8s}{'95% CI':>20s}{'p':>10s}")
print("-" * 60)
rows = {}
for a, b in pairs:
    if a not in Y or b not in Y:
        continue
    d, lo, hi, p = paired_bootstrap_micro(Y["_ref"], Y[a], Y[b], B=B, seed=SEED)
    sig = "*" if (lo > 0 or hi < 0) else " "
    print(f"{a+' vs '+b:22s}{d:8.2f}   [{lo:6.2f},{hi:6.2f}]{p:10.4f} {sig}")
    rows[f"{a}_vs_{b}"] = {"delta_micro_F1": round(d, 2),
                          "ci95": [round(lo, 2), round(hi, 2)],
                          "boot_p": round(p, 4)}

json.dump(rows, open(YOUR_DUMP.replace(".json", "_significance.json"), "w"), indent=2)
print("\n[stat] CI excludes 0  => significant at ~0.05. The CI on dF1 is the effect size.")
print("[stat] For per-study metrics (RadGraph F1 per report, sentence BLEU):")
print("       feed two equal-length per-study arrays to wilcoxon_cliffs(a, b)")
print("       -> (p, Cliff's delta).  |delta|: 0.15 small, 0.33 medium, 0.47 large.")
