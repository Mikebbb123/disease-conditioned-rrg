"""
Clean evaluation: generation metrics + CheXbert clinical F1.

NLG metrics are reported under TWO protocols:
  1. legacy in-house (BLEU-1..4 / ROUGE-* / CIDEr / METEOR) — kept for
     backward-comparability with earlier runs and for training/early-stop.
  2. R2Gen OFFICIAL protocol (R2Gen_BLEU_1..4 / R2Gen_METEOR / R2Gen_ROUGE_L) —
     uses R2Gen's clean_report_iu_xray normalization + R2Gen's compute_scores
     (vendored pycocoevalcap). THIS is the protocol to use when comparing
     against published IU-Xray numbers (R2Gen / R2GenCMN / PPKED / METransformer).

The legacy protocol systematically UNDER-reports BLEU/ROUGE on this dataset
(e.g. BLEU-4 11.96 legacy vs 18.49 R2Gen on the same predictions) because it
does not apply the standard punctuation/spacing normalization. Always cite the
R2Gen_* numbers for external comparison.

To enable the R2Gen protocol, clone the official repo and point R2GEN_REPO at it:
    git clone https://github.com/zhjohnchan/R2Gen.git
    export R2GEN_REPO=/content/R2Gen          # or set the default below
METEOR additionally needs Java:  apt-get install -y default-jre
If R2Gen repo / Java is unavailable, the R2Gen_* metrics are silently skipped
and only the legacy protocol is reported (training is never interrupted).
"""
import os
import json
import functools
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import rag_collate_fn
from config import DataConfig
from chexbert_eval import get_chexbert_labeler, compute_clinical_f1_chexbert


# Path to the cloned official R2Gen repo (override via env var R2GEN_REPO).
R2GEN_REPO = os.environ.get("R2GEN_REPO", "/content/R2Gen")


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_predictions(model, dataset, train_config, device="cuda",
                         oracle_hint=False):
    """If oracle_hint=True, feeds GT disease_labels as the hint — isolates
    decoder quality from the image->findings predictor quality."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        collate_fn=rag_collate_fn,
        num_workers=train_config.num_workers,
    )

    predictions, references, ids = [], [], []

    for batch in tqdm(loader, desc="Generating"):
        frontal = batch["frontal"].to(device)
        lateral = batch["lateral"].to(device)

        gt_labels = batch["disease_labels"].to(device) if oracle_hint else None

        preds = model.generate(
            frontal, lateral,
            num_beams=getattr(train_config, "num_beams", 8),
            max_length=train_config.max_target_length,
            use_sampling=getattr(train_config, "use_sampling", False),
            length_penalty=getattr(train_config, "length_penalty", None),
            min_length=getattr(train_config, "min_length", None),
            no_repeat_ngram_size=getattr(train_config, "no_repeat_ngram_size", 3),
            gt_labels=gt_labels,
        )

        predictions.extend(preds)
        references.extend(batch["reports"])
        ids.extend(batch["ids"])

    return predictions, references, ids


# ============================================================
# R2Gen OFFICIAL protocol — clean_report_iu_xray + compute_scores
# ============================================================

_r2gen_clean = None
_r2gen_compute = None
_r2gen_loaded = False


def _load_r2gen_protocol():
    """Lazy-load R2Gen's official cleaner + scorer from the cloned repo.
    Returns (clean_fn, compute_scores_fn) or (None, None) if unavailable."""
    global _r2gen_clean, _r2gen_compute, _r2gen_loaded
    if _r2gen_loaded:
        return _r2gen_clean, _r2gen_compute
    _r2gen_loaded = True

    import sys, importlib
    if not os.path.isdir(R2GEN_REPO):
        print(f"[Eval] R2Gen protocol skipped — repo not found at {R2GEN_REPO} "
              f"(set R2GEN_REPO env var or clone zhjohnchan/R2Gen).")
        return None, None

    # ensure R2Gen's vendored pycocoevalcap wins over any pip-installed one
    for mod in list(sys.modules):
        if mod.startswith("pycocoevalcap"):
            del sys.modules[mod]
    if R2GEN_REPO not in sys.path:
        sys.path.insert(0, R2GEN_REPO)

    try:
        # clean_report_iu_xray lives as a method on Tokenizer; we only need the
        # function logic, so replicate the call via a throwaway namespace OR
        # import the module and grab the unbound method.
        metrics_mod = importlib.import_module("modules.metrics")
        compute_scores = metrics_mod.compute_scores

        tok_mod = importlib.import_module("modules.tokenizers")

        # clean_report_iu_xray only uses `self` for nothing relevant; bind a
        # dummy object exposing it.
        def clean_fn(report, _impl=tok_mod.Tokenizer.clean_report_iu_xray):
            class _Dummy:  # minimal stand-in; method ignores all self.* state
                pass
            return _impl(_Dummy(), report)

        _r2gen_clean, _r2gen_compute = clean_fn, compute_scores
        print(f"[Eval] R2Gen official protocol loaded from {R2GEN_REPO}")
    except Exception as e:
        print(f"[Eval] R2Gen protocol skipped — failed to import ({e})")
        _r2gen_clean, _r2gen_compute = None, None

    return _r2gen_clean, _r2gen_compute


@functools.lru_cache(maxsize=4)
def _raw_reports_by_id(annotation_file: str) -> Dict[str, str]:
    """Raw (UNcleaned) reports keyed by sample id, straight from
    annotation.json. The R2Gen protocol must score against THESE — the same
    references R2Gen uses — not our in-house clean_report_text output (which
    strips 'xxxx', expands w/o->without, etc.). clean_report_iu_xray is applied
    downstream inside compute_r2gen_metrics, exactly as in R2Gen."""
    with open(annotation_file) as f:
        data = json.load(f)
    out = {}
    for split in ("train", "val", "test"):
        for ex in data.get(split, []):
            out[ex["id"]] = ex["report"]
    return out


def compute_r2gen_metrics(predictions, references) -> Dict[str, float]:
    """NLG metrics under the official R2Gen protocol. Returns {} if the
    R2Gen repo / Java is unavailable (never raises)."""
    clean_fn, compute_scores = _load_r2gen_protocol()
    if clean_fn is None or compute_scores is None:
        return {}

    gts, res = {}, {}
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        res[i] = [clean_fn(pred)]
        gts[i] = [clean_fn(ref)]

    try:
        scores = compute_scores(gts, res)  # {BLEU_1..4, METEOR, ROUGE_L} in 0-1
    except Exception as e:
        print(f"[Eval] R2Gen compute_scores failed (METEOR needs Java?) — {e}")
        # try BLEU/ROUGE only by catching METEOR; fall back to empty
        return {}

    out = {}
    for k, v in scores.items():
        out[f"R2Gen_{k}"] = round(v * 100, 2)
    return out


# ============================================================
# Legacy in-house captioning metrics (kept for back-comparability)
# ============================================================

def compute_captioning_metrics(predictions, references) -> Dict[str, float]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        has_cocoeval = True
    except ImportError:
        has_cocoeval = False

    if not has_cocoeval:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sm = SmoothingFunction().method1
        refs_tok = [[r.split()] for r in references]
        pred_tok = [p.split() for p in predictions]
        results = {}
        for n in (1, 2, 3, 4):
            weights = tuple([1.0/n]*n + [0.0]*(4-n))
            score = corpus_bleu(refs_tok, pred_tok, weights=weights, smoothing_function=sm)
            results[f"BLEU-{n}"] = round(score * 100, 2)
        return results

    gts = {i: [r] for i, r in enumerate(references)}
    res = {i: [p] for i, p in enumerate(predictions)}

    results = {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    for n, s in zip(range(1, 5), bleu_scores):
        results[f"BLEU-{n}"] = round(s * 100, 2)

    # ROUGE-1, ROUGE-2, ROUGE-L (all from rouge-score for consistency)
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        import numpy as np
        results["ROUGE-1"] = round(np.mean(r1) * 100, 2)
        results["ROUGE-2"] = round(np.mean(r2) * 100, 2)
        results["ROUGE-L"] = round(np.mean(rl) * 100, 2)
    except Exception:
        pass
    results["CIDEr"]   = round(Cider().compute_score(gts, res)[0] * 100, 2)

    try:
        from pycocoevalcap.meteor.meteor import Meteor
        results["METEOR"] = round(Meteor().compute_score(gts, res)[0] * 100, 2)
    except Exception:
        pass
    return results


# ============================================================
# End-to-end
# ============================================================

def evaluate_model(model, dataset, train_config, device="cuda",
                   save_name=None, cache_dir=None,
                   oracle_hint=False, annotation_file=None):
    predictions, references, ids = generate_predictions(
        model, dataset, train_config, device,
        oracle_hint=oracle_hint,
    )

    # legacy protocol (kept for training/early-stop continuity)
    metrics = compute_captioning_metrics(predictions, references)

    # R2Gen OFFICIAL protocol — score against the RAW annotation refs (matched
    # by id), NOT our in-house-cleaned `references`, so the numbers are directly
    # comparable to published R2Gen / IU-Xray results. clean_report_iu_xray is
    # applied to these inside compute_r2gen_metrics, same as R2Gen does.
    ann = annotation_file or DataConfig().annotation_file
    raw_by_id = _raw_reports_by_id(ann)
    r2gen_refs, n_missing = [], 0
    for i, ref in zip(ids, references):
        raw = raw_by_id.get(i)
        if raw is None:                 # id not in annotation -> fall back, warn
            n_missing += 1
            raw = ref
        r2gen_refs.append(raw)
    if n_missing:
        print(f"[Eval] R2Gen protocol: {n_missing}/{len(ids)} ids not found in "
              f"{ann} — used in-house ref for those (check id alignment).")
    metrics.update(compute_r2gen_metrics(predictions, r2gen_refs))

    # CheXbert F1
    try:
        labeler = get_chexbert_labeler(cache_dir=cache_dir, device=device)
        metrics.update(compute_clinical_f1_chexbert(predictions, references, labeler))
    except Exception as e:
        print(f"[Eval] CheXbert skipped — {e}")

    # Undo f1chexbert's global tokenizer monkey-patch so it can't corrupt the
    # T5 tokenizer used by training (weighted-CE offset mapping) in the next epoch.
    try:
        from chexbert_eval import _undo_tokenizer_patch
        _undo_tokenizer_patch()
    except Exception:
        pass

    # RadGraph F1 (Simple / Partial / Complete)
    try:
        from radgraph_eval import compute_radgraph_f1
        metrics.update(compute_radgraph_f1(predictions, references))
    except Exception as e:
        print(f"[Eval] RadGraph skipped — {e}")

    if save_name:
        os.makedirs(train_config.output_dir, exist_ok=True)
        path = os.path.join(train_config.output_dir, save_name)
        # Save only R2Gen + CheXbert + RadGraph — strip legacy BLEU/ROUGE/CIDEr/METEOR
        legacy_keys = {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                       "ROUGE-1", "ROUGE-2", "ROUGE-L", "CIDEr", "METEOR"}
        save_metrics = {k: v for k, v in metrics.items() if k not in legacy_keys}
        with open(path, "w") as f:
            json.dump({
                "metrics": save_metrics,
                "samples": [
                    {"id": i, "prediction": p, "reference": r}
                    for i, p, r in zip(ids, predictions, references)
                ],
            }, f, indent=2)

    # print both protocols side by side if available
    if "R2Gen_BLEU_4" in metrics:
        print(f"[Eval] R2Gen protocol: BLEU-4={metrics.get('R2Gen_BLEU_4')} "
              f"METEOR={metrics.get('R2Gen_METEOR')} ROUGE_L={metrics.get('R2Gen_ROUGE_L')}")
    chex_f1 = metrics.get("ClinicalF1_chexbert_macro_F")
    if chex_f1 is not None:
        print(f"[Eval] CheXbert Clinical F1 (macro): {chex_f1}")
    return metrics
