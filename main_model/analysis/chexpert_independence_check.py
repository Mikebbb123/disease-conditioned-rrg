"""
Independent rule-based (CheXpert) labeler check for Sect. 6.2.

WHY: CheXbert is the primary metric, but the cue labels, the auxiliary
supervision AND the metric all come from CheXbert. The robustness check is:
re-score the SAME reports on the SAME 349 common subset with a labeler that
shares NO parameters with CheXbert, and see whether the 5-class F1 ordering
survives. This script produces those numbers. It does NOT invent them.

INDEPENDENCE CAVEAT (put this in the paper, a reviewer will raise it):
CheXbert was *distilled to imitate* the CheXpert rule-based labeler. So
"CheXpert rules" satisfies the literal claim "shares no parameters with
CheXbert", but the two are NOT statistically independent -- high agreement is
expected by construction, and does not by itself rule out labeler-specific
phrasing bias. Treat this as a parameter-independent sanity check, not as proof
of labeler-independent clinical content. A genuinely independent estimate needs
a differently-derived labeler or radiologist review.

SETUP (once, in Colab):
    git clone https://github.com/stanfordmlgroup/chexpert-labeler
    # follow its README: NegBio + NLTK (punkt, universal_tagset) + bllipparser.
    # The labeler is invoked as:
    #   python label.py --reports_path R.csv --output_path O.csv --verbose
    # Input CSV must have a single column whose header is "Report Impression".
    # Output CSV has one column per CheXpert category; values:
    #   1.0 = positive, 0.0 = negative, -1.0 = uncertain, blank = not mentioned.

Then set CHEXPERT_REPO below and run this file with the same Python env that
has access to YOUR_DUMP / BASELINE_DUMPS.
"""
import os
import sys
import json
import subprocess
import tempfile

import pandas as pd
from sklearn.metrics import f1_score

# ----- EDIT THESE (reuse the exact paths from chexbert_eval.py) -----
YOUR_DUMP = "./dumps/ours_seed42_test_final.json"
BASELINE_DUMPS = {
    # Point each entry at that method's eval dump (schema: {"samples":[{id,prediction,reference}]}).
    "R2Gen":     "./dumps/r2gen_test_generated.json",
    "PromptMRG": "./dumps/promptmrg_test_generated.json",
    "42": "./dumps/ours_seed42_test_final.json",
    "87": "./dumps/ours_seed87_test_final.json",
    "13": "./dumps/ours_seed13_test_final.json",
    "qwen": "./dumps/qwen_test_generated.json",
    "nn": "./dumps/nn_top1_test_generated.json",
}
CHEXPERT_REPO = os.environ.get("CHEXPERT_REPO", "./chexpert-labeler")   # cloned stanfordmlgroup/chexpert-labeler
NEGBIO_REPO   = os.environ.get("NEGBIO_REPO", "./NegBio")   # cloned ncbi-nlp/NegBio (must be importable)
# label.py must run INSIDE the chexpert-label conda env, not base Colab python.
# condacolab route: conda run. Docker route: set to e.g. [] and call inside container.
LABELER_CMD = ["conda", "run", "-n", "chexpert-label", "python"]
UNCERTAIN_AS_POSITIVE = False  # default: uncertain(-1)->negative, matches the
                               # strict "positive-F1" reading f1chexbert uses.
                               # Flip to True to report the u-ones policy too.
# -------------------------------------------------------------------

FIVE = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def _load_pred_by_id(path):
    """Identical to your chexbert_eval._load_pred_by_id: one report per study,
    prefer the frontal view for per-image dumps with duplicate ids."""
    obj = json.load(open(path))
    recs = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    chosen = {}
    for d in recs:
        sid = d["id"]
        pred = d.get("generated", d.get("prediction"))
        vw = (d.get("view") or "").lower()
        is_frontal = any(t in vw for t in ("pa", "ap", "frontal"))
        if sid not in chosen or (is_frontal and not chosen[sid][1]):
            chosen[sid] = (pred, is_frontal)
    return {k: v[0] for k, v in chosen.items()}


def chexpert_label(reports):
    """Run the CheXpert rule-based labeler on a list of report strings.
    Returns a DataFrame with one row per report and the CheXpert category cols."""
    if not os.path.isdir(CHEXPERT_REPO):
        sys.exit(f"[chexpert] repo not found at {CHEXPERT_REPO} -- clone it first.")
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "reports.csv")
        op = os.path.join(td, "labeled.csv")
        # Official stanfordmlgroup labeler expects a HEADERLESS single-column csv.
        # pandas QUOTE_MINIMAL handles commas / multi-line reports automatically.
        pd.DataFrame([r if r else "" for r in reports]).to_csv(
            rp, index=False, header=False)
        cmd = LABELER_CMD + ["label.py",
                             "--reports_path", rp, "--output_path", op, "--verbose"]
        env = dict(os.environ)
        # conda run often drops the notebook's %env PYTHONPATH; set it explicitly.
        env["PYTHONPATH"] = NEGBIO_REPO + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=CHEXPERT_REPO, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            # Surface the REAL traceback from label.py instead of a bare exit code.
            sys.stderr.write("\n===== label.py STDOUT =====\n" + (proc.stdout or ""))
            sys.stderr.write("\n===== label.py STDERR =====\n" + (proc.stderr or ""))
            raise RuntimeError(f"chexpert label.py failed (exit {proc.returncode}); "
                               f"see STDERR above.")
        out = pd.read_csv(op)
    return out


def to_binary(df):
    """5-class multi-hot. positive iff value == 1.0 (and uncertain if enabled).
    blank / 0 / -1 -> 0, matching the strict positive-F1 convention."""
    import numpy as np
    M = np.zeros((len(df), len(FIVE)), dtype=int)
    for j, c in enumerate(FIVE):
        col = df[c] if c in df.columns else pd.Series([float("nan")] * len(df))
        pos = (col == 1.0)
        if UNCERTAIN_AS_POSITIVE:
            pos = pos | (col == -1.0)
        M[:, j] = pos.fillna(False).astype(int).values
    return M


# ---- references + Ours from YOUR_DUMP ----
your = json.load(open(YOUR_DUMP))
ref_by_id = {s["id"]: s["reference"]  for s in your["samples"]}
methods = {"Ours": {s["id"]: s["prediction"] for s in your["samples"]}}
for name, path in BASELINE_DUMPS.items():
    if os.path.exists(path):
        methods[name] = _load_pred_by_id(path)
    else:
        print(f"[chexpert] WARNING: dump missing for {name}: {path} -- skipped")

# ---- SAME common subset as Table 1b ----
common = set(ref_by_id)
for d in methods.values():
    common &= set(d)
common = [i for i in (s["id"] for s in your["samples"]) if i in common]
print(f"[chexpert] common subset = {len(common)} ids over {list(methods)}")
if not common:
    sys.exit("[chexpert] empty common subset.")

references = [ref_by_id[i] for i in common]

# ---- label references once (y_true) + report 5-class positive COUNTS ----
ref_df = chexpert_label(references)
y_true = to_binary(ref_df)
print(f"\n=== 5-class test-positive counts on the n={len(common)} subset "
      f"(CheXpert, uncertain_as_positive={UNCERTAIN_AS_POSITIVE}) ===")
for j, c in enumerate(FIVE):
    print(f"  {c:16s}: n_pos = {int(y_true[:, j].sum())}")
print("  (compare to full-590 test counts Edema=7, Consolidation=3 from Sect 5.3)")

# ---- score each method on y_true ----
print(f"\n=== Independent CheXpert 5-class F1 (n={len(common)}) ===")
hdr = f"{'Method':12s}{'micro':>8s}{'macro':>8s}  " + "".join(f"{c[:4]:>6s}" for c in FIVE)
print(hdr); print("-" * len(hdr))
results = {}
for name in methods:
    preds = [methods[name][i] for i in common]
    y_pred = to_binary(chexpert_label(preds))
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0) * 100
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    per = f1_score(y_true, y_pred, average=None, zero_division=0) * 100
    results[name] = {"micro": round(micro, 2), "macro": round(macro, 2),
                     **{c: round(p, 2) for c, p in zip(FIVE, per)}}
    print(f"{name:12s}{micro:8.2f}{macro:8.2f}  " + "".join(f"{p:6.1f}" for p in per))

out = YOUR_DUMP.replace(".json", "_chexpert_independent_scores.json")
json.dump({"common_ids": common, "n": len(common),
           "uncertain_as_positive": UNCERTAIN_AS_POSITIVE,
           "ref_pos_counts": {c: int(y_true[:, j].sum()) for j, c in enumerate(FIVE)},
           "rows": results}, open(out, "w"), indent=2)
print(f"\n[chexpert] -> {out}")
print("[chexpert] Report micro-F1 as primary (macro is unstable on Edema/Consolidation).")
print("[chexpert] Also run once with UNCERTAIN_AS_POSITIVE=True and report both.")
