"""
score_r2gen.py  (self-contained — does NOT import evaluate.py)
=============================================================
Avoids the `data` name clash (R2Gen's data/ folder vs your paper's data.py) by
inlining the two R2Gen-protocol helpers instead of importing evaluate.py.

Needs importable (and data-independent): chexbert_eval.py, radgraph_eval.py, config.py
Set R2GEN_REPO to the cloned R2Gen repo (for its cleaner + scorer).
"""
import os
import sys
import json
import functools
import importlib

# ----- EDIT THESE -----
R2GEN_REPO = os.environ.get("R2GEN_REPO", "./R2Gen")
YOUR_DUMP  = "./dumps/ours_seed42_test_final.json"
R2GEN_DUMP = "./dumps/r2gen_test_generated.json"
# ----------------------

os.environ["R2GEN_REPO"] = R2GEN_REPO

# ============================================================
# inlined from your evaluate.py (R2Gen official protocol) — no `data` dependency
# ============================================================
_r2gen_clean = None
_r2gen_compute = None
_r2gen_loaded = False


def _load_r2gen_protocol():
    global _r2gen_clean, _r2gen_compute, _r2gen_loaded
    if _r2gen_loaded:
        return _r2gen_clean, _r2gen_compute
    _r2gen_loaded = True
    if not os.path.isdir(R2GEN_REPO):
        print(f"[Eval] R2Gen protocol skipped — repo not found at {R2GEN_REPO}")
        return None, None
    for mod in list(sys.modules):
        if mod.startswith("pycocoevalcap"):
            del sys.modules[mod]
    if R2GEN_REPO not in sys.path:
        sys.path.insert(0, R2GEN_REPO)
    try:
        metrics_mod = importlib.import_module("modules.metrics")
        compute_scores = metrics_mod.compute_scores
        tok_mod = importlib.import_module("modules.tokenizers")

        def clean_fn(report, _impl=tok_mod.Tokenizer.clean_report_iu_xray):
            class _Dummy:
                pass
            return _impl(_Dummy(), report)

        _r2gen_clean, _r2gen_compute = clean_fn, compute_scores
        print(f"[Eval] R2Gen official protocol loaded from {R2GEN_REPO}")
    except Exception as e:
        print(f"[Eval] R2Gen protocol skipped — failed to import ({e})")
        _r2gen_clean, _r2gen_compute = None, None
    return _r2gen_clean, _r2gen_compute


@functools.lru_cache(maxsize=4)
def _raw_reports_by_id(annotation_file):
    with open(annotation_file) as f:
        data = json.load(f)
    out = {}
    for split in ("train", "val", "test"):
        for ex in data.get(split, []):
            out[ex["id"]] = ex["report"]
    return out


def compute_r2gen_metrics(predictions, references):
    clean_fn, compute_scores = _load_r2gen_protocol()
    if clean_fn is None or compute_scores is None:
        return {}
    gts, res = {}, {}
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        res[i] = [clean_fn(pred)]
        gts[i] = [clean_fn(ref)]
    try:
        scores = compute_scores(gts, res)
    except Exception as e:
        print(f"[Eval] R2Gen compute_scores failed (METEOR needs Java?) — {e}")
        return {}
    return {f"R2Gen_{k}": round(v * 100, 2) for k, v in scores.items()}


# ============================================================
# clean imports (no `data` dependency)
# ============================================================
from chexbert_eval import (get_chexbert_labeler,
                           compute_clinical_f1_chexbert, _undo_tokenizer_patch)
from radgraph_eval import compute_radgraph_f1
from config import DataConfig

# (1) YOUR model's eval dump -> exact ids + references (pins the GT side)
your = json.load(open(YOUR_DUMP))
samples = your["samples"]
ids_order = [s["id"] for s in samples]
ref_by_id = {s["id"]: s["reference"] for s in samples}

# (2) R2Gen predictions, aligned by id
r2gen = json.load(open(R2GEN_DUMP))
pred_by_id = {d["id"]: d["generated"] for d in r2gen}
keep_ids = [i for i in ids_order if i in pred_by_id]
missing  = [i for i in ids_order if i not in pred_by_id]
print(f"[score] aligned {len(keep_ids)}/{len(ids_order)} test ids"
      + (f"  ({len(missing)} missing: {missing[:5]}...)" if missing else ""))
predictions = [pred_by_id[i] for i in keep_ids]
references  = [ref_by_id[i]  for i in keep_ids]

metrics = {}

# (3) NLG under R2Gen protocol (vs RAW annotation refs by id)
raw_by_id  = _raw_reports_by_id(DataConfig().annotation_file)
r2gen_refs = [raw_by_id[i] for i in keep_ids]
metrics.update(compute_r2gen_metrics(predictions, r2gen_refs))

# (4) CheXbert vs the SAME in-house references
labeler = get_chexbert_labeler(device="cuda")
metrics.update(compute_clinical_f1_chexbert(predictions, references, labeler))
_undo_tokenizer_patch()

# (5) RadGraph vs the same references
metrics.update(compute_radgraph_f1(predictions, references))

# ----- Table-1 row -----
print("\n=== R2Gen (official ckpt) — Table 1 numbers ===")
row = {
    "BLEU4":             metrics.get("R2Gen_BLEU_4"),
    "ROUGE-L":           metrics.get("R2Gen_ROUGE_L"),
    "METEOR":            metrics.get("R2Gen_METEOR"),
    "CheXbert-5 micro":  metrics.get("ClinicalF1_chexbert_micro_F_5"),
    "CheXbert-5 macro":  metrics.get("ClinicalF1_chexbert_macro_F_5"),
    "CheXbert-14 micro": metrics.get("ClinicalF1_chexbert_micro_F_14"),
    "CheXbert-14 macro": metrics.get("ClinicalF1_chexbert_macro_F_14"),
    "RG-Simple":         metrics.get("RadGraph_Simple"),
    "RG-Partial":        metrics.get("RadGraph_Partial"),
}
for k, v in row.items():
    print(f"  {k:18s}: {v}")

out = R2GEN_DUMP.replace(".json", "_scored.json")
json.dump(metrics, open(out, "w"), indent=2)
print(f"\n[score] full metrics (incl. per-class) -> {out}")
