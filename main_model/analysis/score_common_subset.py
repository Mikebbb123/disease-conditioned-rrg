"""
score_common_subset.py   (companion to score_r2gen.py, frontal-dedup)
=============================================================
PromptMRG's IU 'test' selection overlaps your R2Gen 590-test split on only ~349
studies (IU-Xray has no official split), AND its annotation is per-image (two
rows per study). This script:
  (1) loads each method's dump, keeping ONE report per study — for per-image
      dumps (PromptMRG) it prefers the FRONTAL (PA/AP) view, which is
      PromptMRG's in-distribution view;
  (2) takes the common id subset across ALL methods;
  (3) re-scores every method on that SAME subset under your identical pipeline
      (R2Gen-NLG + CheXbert-5/14 + RadGraph).

Keep the MAIN Table 1 at 590; present this as a controlled common-subset
comparison (Table 1b / footnote). Run in YOUR paper env (NOT the PromptMRG env).

Needs importable (data-independent): chexbert_eval.py, radgraph_eval.py, config.py
Set R2GEN_REPO to the cloned R2Gen repo (for its cleaner + scorer).
"""
import os
import sys
import json
import functools
import importlib

# ----- EDIT THESE -----
R2GEN_REPO = "/content/R2Gen"

# YOUR model's eval dump. schema: {"samples":[{"id","prediction","reference"}]}
# Pins the references and the id universe.
YOUR_DUMP = "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed42/test_final_oracle_42.json"

# Other methods. List dumps: [{"id","generated"(,"view")}]  or  {"samples":[...]}.
BASELINE_DUMPS = {
    "PromptMRG": "/content/drive/MyDrive/promptmrg/promptmrg_test_generated.json",
    "seed42_oracle" : "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed42/test_final_oracle.json",
    "seed87_oracle" : "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed87/test_final_oracle.json",
    "seed13_oracle" : "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed13/test_final_oracle.json",
    "seed42" : "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed42/test_final.json",
    "seed13" : "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed13/test_final.json",
    "seed87" : "/content/drive/MyDrive/iu_xray_rag/output_resampler_rag_seed87/test_final.json",
    "R2Gen": "/content/drive/MyDrive/r2gen_test_generated.json",
    "Qwen":  "/content/drive/MyDrive/eval_baseline_preds.json",
    "NN":    "/content/drive/MyDrive/preds_nn_top1.json",
    "Qwen_oracle": "/content/drive/MyDrive/eval_oracle_preds.json",
  
}
# ----------------------

os.environ["R2GEN_REPO"] = R2GEN_REPO

# ============================================================
# R2Gen official protocol helpers (inlined — avoids the `data` name clash)
# ============================================================
_r2gen_clean = _r2gen_compute = None
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
        compute_scores = importlib.import_module("modules.metrics").compute_scores
        tok_mod = importlib.import_module("modules.tokenizers")

        def clean_fn(report, _impl=tok_mod.Tokenizer.clean_report_iu_xray):
            class _Dummy:
                pass
            return _impl(_Dummy(), report)

        _r2gen_clean, _r2gen_compute = clean_fn, compute_scores
        print(f"[Eval] R2Gen official protocol loaded from {R2GEN_REPO}")
    except Exception as e:
        print(f"[Eval] R2Gen protocol skipped — failed to import ({e})")
    return _r2gen_clean, _r2gen_compute


@functools.lru_cache(maxsize=4)
def _raw_reports_by_id(annotation_file):
    data = json.load(open(annotation_file))
    out = {}
    for split in ("train", "val", "test"):
        for ex in data.get(split, []):
            out[ex["id"]] = ex["report"]
    return out


def compute_r2gen_metrics(predictions, references):
    clean_fn, compute_scores = _load_r2gen_protocol()
    if clean_fn is None or compute_scores is None:
        return {}
    res = {i: [clean_fn(p)] for i, p in enumerate(predictions)}
    gts = {i: [clean_fn(r)] for i, r in enumerate(references)}
    try:
        scores = compute_scores(gts, res)
    except Exception as e:
        print(f"[Eval] R2Gen compute_scores failed (METEOR needs Java?) — {e}")
        return {}
    return {f"R2Gen_{k}": round(v * 100, 2) for k, v in scores.items()}


# ============================================================
# data-independent imports
# ============================================================
from chexbert_eval import (get_chexbert_labeler,
                           compute_clinical_f1_chexbert, _undo_tokenizer_patch)
from radgraph_eval import compute_radgraph_f1
from config import DataConfig


def _load_pred_by_id(path):
    """One report per study. For per-image dumps with duplicate ids (PromptMRG),
    prefer the FRONTAL (PA/AP) view; safe for single-entry / no-view dumps too."""
    obj = json.load(open(path))
    recs = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    chosen = {}  # id -> (pred, is_frontal)
    for d in recs:
        sid  = d["id"]
        pred = d.get("generated", d.get("prediction"))
        vw   = (d.get("view") or "").lower()
        is_frontal = any(t in vw for t in ("pa", "ap", "frontal"))
        if sid not in chosen or (is_frontal and not chosen[sid][1]):
            chosen[sid] = (pred, is_frontal)
    return {k: v[0] for k, v in chosen.items()}


# ---- references + your model's predictions from YOUR_DUMP ----
your = json.load(open(YOUR_DUMP))
ref_by_id  = {s["id"]: s["reference"]  for s in your["samples"]}
ours_by_id = {s["id"]: s["prediction"] for s in your["samples"]}

methods = {"Ours": ours_by_id}
for name, path in BASELINE_DUMPS.items():
    if os.path.exists(path):
        methods[name] = _load_pred_by_id(path)
    else:
        print(f"[score] WARNING: dump missing for {name}: {path} — skipped")

# ---- common id subset: present in EVERY method AND has a reference ----
common = set(ref_by_id)
for name, d in methods.items():
    common &= set(d)
common = [i for i in (s["id"] for s in your["samples"]) if i in common]  # stable order
print(f"\n[score] common subset = {len(common)} ids "
      f"(intersection of {list(methods)} + references)")
for name, d in methods.items():
    print(f"        {name:10s}: covers {len(set(d) & set(ref_by_id))}/{len(ref_by_id)} of test")

if not common:
    sys.exit("[score] empty common subset — check id conventions across dumps.")

raw_by_id = _raw_reports_by_id(DataConfig().annotation_file)
references = [ref_by_id[i] for i in common]
r2gen_refs = [raw_by_id[i] for i in common]

# ---- score each method on the SAME common subset ----
labeler = get_chexbert_labeler(device="cuda")
rows = {}
for name, pred_by_id in methods.items():
    preds = [pred_by_id[i] for i in common]
    m = {}
    m.update(compute_r2gen_metrics(preds, r2gen_refs))
    m.update(compute_clinical_f1_chexbert(preds, references, labeler))
    _undo_tokenizer_patch()
    m.update(compute_radgraph_f1(preds, references))
    rows[name] = m
    print(f"[score] scored {name} on {len(common)} ids")

# ---- compact Table-1b ----
cols = [
    ("BLEU-4",        "R2Gen_BLEU_4"),
    ("ROUGE-L",       "R2Gen_ROUGE_L"),
    ("METEOR",        "R2Gen_METEOR"),
    ("CheX-5 micro",  "ClinicalF1_chexbert_micro_F_5"),
    ("CheX-5 macro",  "ClinicalF1_chexbert_macro_F_5"),
    ("CheX-14 micro", "ClinicalF1_chexbert_micro_F_14"),
    ("RG-Simple",     "RadGraph_Simple"),
    ("RG-Partial",    "RadGraph_Partial"),
]
print(f"\n=== Common-subset comparison (n={len(common)}) ===")
hdr = f"{'Method':12s}" + "".join(f"{c[0]:>15s}" for c in cols)
print(hdr); print("-" * len(hdr))
for name, m in rows.items():
    print(f"{name:12s}" + "".join(f"{str(m.get(k)):>15s}" for _, k in cols))

out = YOUR_DUMP.replace(".json", "_common_subset_scores.json")
json.dump({"common_ids": common, "n": len(common), "rows": rows},
          open(out, "w"), indent=2)
print(f"\n[score] full metrics -> {out}")
print("[score] NOTE: also report 5-class test-positive counts on this subset "
      "(Edema/Consolidation may drop below n=7/3); keep main Table 1 at 590.")
