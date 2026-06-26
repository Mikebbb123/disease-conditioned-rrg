"""
mimic_textonly_validation.py
=============================
Image-free, low-resource validation of the lexical-clinical metric tension on a
subsampled MIMIC-CXR *reports* corpus. Designed for Colab where the images cannot
be downloaded but the text reports + the CheXpert label CSV are available.

WHAT IT REPLICATES (no images required)
---------------------------------------
1. MODE="pure_baseline":  a SciFive-T5 fed a CONSTANT input -> it can only learn
   the marginal p(report), i.e. the normal-finding template. Expectation: high
   BLEU/ROUGE/CIDEr/RadGraph, ~zero CheXbert macro-F1(5). This is the exact
   analog of your IU-Xray "Pure Baseline" and is the must-run experiment to kill
   the "single-dataset artifact" objection.

2. MODE="label_conditioned" (+ CLASS_BALANCED=True): the encoder is fed the
   ground-truth 5-class CheXpert labels as a cue prompt, with normal-sample
   undersampling. Expectation: recovers non-zero CheXbert F1 at a small lexical
   cost -> mirrors your class-balancing claim on a 2nd corpus.
   CAVEAT (state this in the paper): this conditions on GT labels, so it is a
   controlled *text-only proxy* for the recovery effect, NOT a transfer of your
   full image-based pipeline. It is honest as long as you label it as such.

METRIC PROTOCOL
---------------
Reuses your chexbert_eval.py and radgraph_eval.py UNCHANGED (import), and a
verbatim copy of compute_captioning_metrics from your evaluate.py, so the
reported numbers are computed by identical code to your IU-Xray results.

INSTALL (Colab)
---------------
!pip install -q transformers sentencepiece datasets pandas \
    pycocoevalcap rouge-score nltk f1chexbert radgraph
# Place chexbert_eval.py and radgraph_eval.py next to this file.
"""

import os, re, json, random, gzip
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)

# Reuse YOUR evaluators unchanged --------------------------------------------
from chexbert_eval import get_chexbert_labeler, compute_clinical_f1_chexbert
from radgraph_eval import compute_radgraph_f1


# ============================================================
# R2Gen OFFICIAL NLG protocol (BLEU + CIDEr) — Java-free.
# Table 6 must use THESE numbers so the protocol matches the IU-Xray
# results; the legacy compute_captioning_metrics below UNDER-reports.
# ============================================================
import sys, importlib
R2GEN_REPO = os.environ.get("R2GEN_REPO", "/content/R2Gen")
_r2 = {}


def _r2gen_clean():
    """Load R2Gen's clean_report_mimic_cxr (MIMIC text) + ensure R2Gen's
    vendored pycocoevalcap wins over any pip-installed one."""
    if "clean" in _r2:
        return _r2["clean"]
    for m in list(sys.modules):
        if m.startswith("pycocoevalcap"):
            del sys.modules[m]
    if os.path.isdir(R2GEN_REPO) and R2GEN_REPO not in sys.path:
        sys.path.insert(0, R2GEN_REPO)
    Tok = importlib.import_module("modules.tokenizers").Tokenizer
    impl = Tok.clean_report_mimic_cxr          # MIMIC cleaner for MIMIC reports
    def clean(rep, _impl=impl):
        class _D:  # clean_report_* ignores all self.* state
            pass
        return _impl(_D(), rep)
    _r2["clean"] = clean
    return clean


def compute_r2gen_nlg(predictions, references) -> Dict[str, float]:
    """R2Gen-protocol BLEU-1..4 + CIDEr (clean_report_mimic_cxr + vendored
    pycocoevalcap). Pure-python, NO Java. Put R2Gen_BLEU_4 / R2Gen_CIDEr in
    Table 6 (native CIDEr scale — do NOT *100)."""
    try:
        clean = _r2gen_clean()
    except Exception as e:
        print(f"[Eval] R2Gen NLG skipped — repo/import failed ({e}); "
              f"set R2GEN_REPO or `git clone https://github.com/zhjohnchan/R2Gen.git`.")
        return {}
    gts = {i: [clean(r)] for i, r in enumerate(references)}
    res = {i: [clean(p)] for i, p in enumerate(predictions)}
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    bleu, _ = Bleu(4).compute_score(gts, res)
    out = {f"R2Gen_BLEU_{n}": round(bleu[n - 1] * 100, 2) for n in range(1, 5)}
    out["R2Gen_CIDEr"] = round(Cider().compute_score(gts, res)[0], 4)
    return out


# ============================================================
# CONFIG  — edit these
# ============================================================
@dataclass
class Config:
    # --- paths ---
    reports_dir: str = "/content/mimic_reports/files"   # LOCAL: fast small-file reads
    chexpert_csv: str = "/content/mimic-cxr-2.0.0-chexpert.csv.gz"  # LOCAL
    output_dir: str = "/content/drive/MyDrive/111"      # DRIVE: persist results JSON
    ckpt_dir: str = "/content/mimic_ckpt"               # LOCAL: epoch checkpoints (do NOT put on Drive)

    # --- experiment switches ---
    mode: str = "pure_baseline"        # "pure_baseline" | "label_conditioned"
    class_balanced: bool = False       # only meaningful for label_conditioned

    # --- low-resource subsample (match your IU-Xray regime) ---
    n_train: int = 2069
    n_val: int = 296
    n_test: int = 590
    normal_ratio_eval: float = 0.85    # skew of val/test (drives the tension)
    abn_to_normal_train: float = 1.5   # used when class_balanced=True (1.5:1)
    uncertain_as_positive: bool = False
    min_report_len: int = 15           # == DataConfig.min_report_len (your config.py)

    # --- model / training (mirrors your TrainConfig) ---
    model_name: str = "razent/SciFive-base-Pubmed_PMC"
    max_source_length: int = 64        # no visual tokens here, so source is tiny
    max_target_length: int = 200       # == TrainConfig.max_target_length
    epochs: int = 50                   # == num_epochs (BLEU-4 early stop, patience 10)
    early_stop_patience: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    batch_size: int = 16
    num_workers: int = 2
    # generation (matches your inference)
    num_beams: int = 8
    no_repeat_ngram_size: int = 3
    min_length: int = 40            # match main pipeline (paper §3.7)
    length_penalty: float = 1.5     # match main pipeline (paper §3.7)

    seed: int = 42

CFG = Config()
FIVE = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
CONST_INPUT = "generate chest x-ray radiology report findings"  # pure-baseline input


# ============================================================
# 1. Report section parsing (FINDINGS)
# ============================================================
_SECTION = re.compile(
    r"(FINDINGS|IMPRESSION|CONCLUSION|RECOMMENDATION|NOTIFICATION|"
    r"COMPARISON|INDICATION|TECHNIQUE|HISTORY|EXAMINATION)\s*:",
    re.IGNORECASE,
)

def extract_findings(text: str) -> str:
    """Return the FINDINGS section; fall back to IMPRESSION; else ''."""
    text = text.replace("\n", " ")
    matches = list(_SECTION.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.setdefault(name, text[start:end].strip())
    body = sections.get("FINDINGS") or sections.get("IMPRESSION") or ""
    body = re.sub(r"\s+", " ", body).strip()
    return body


def report_path(reports_dir: str, subject_id: int, study_id: int) -> str:
    sid = str(subject_id)
    return os.path.join(reports_dir, f"p{sid[:2]}", f"p{sid}", f"s{study_id}.txt")


# ============================================================
# 2. Labels  ->  normal / abnormal  +  5-class positives
# ============================================================
def load_labels(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.chexpert_csv)
    pos = 1.0
    def positives(row):
        out = []
        for c in FIVE:
            v = row.get(c, np.nan)
            if v == pos or (cfg.uncertain_as_positive and v == -1.0):
                out.append(c)
        return out
    df["pos5"] = df.apply(positives, axis=1)
    path_cols = [c for c in df.columns if c not in ("subject_id", "study_id", "pos5")]
    patho = [c for c in path_cols if c != "No Finding"]
    def is_normal(row):
        if row.get("No Finding", np.nan) == 1.0:
            return True
        return not any(row.get(c, np.nan) == 1.0 for c in patho)
    df["is_normal"] = df.apply(is_normal, axis=1)
    return df[["subject_id", "study_id", "pos5", "is_normal"]]


# ============================================================
# 3. Build a low-resource, imbalance-matched subsample
# ============================================================
def _collect(df_rows, reports_dir, n_needed, seen) -> List[Dict]:
    out = []
    for _, r in df_rows.iterrows():
        if len(out) >= n_needed:
            break
        key = (r.subject_id, r.study_id)
        if key in seen:
            continue
        p = report_path(reports_dir, r.subject_id, r.study_id)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            findings = extract_findings(f.read())
        if len(findings.split()) < CFG.min_report_len:   # match DataConfig.min_report_len
            continue
        seen.add(key)
        out.append({"study_id": int(r.study_id), "findings": findings,
                    "pos5": list(r.pos5), "is_normal": bool(r.is_normal)})
    return out


def build_splits(cfg: Config):
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    df = load_labels(cfg).sample(frac=1.0, random_state=cfg.seed)
    normal_df = df[df.is_normal]
    abn_df = df[~df.is_normal]

    # how many normal/abnormal each split needs
    def split_counts(n, ratio):  # returns (n_normal, n_abnormal)
        n_norm = round(n * ratio); return n_norm, n - n_norm
    te = split_counts(cfg.n_test, cfg.normal_ratio_eval)
    va = split_counts(cfg.n_val,  cfg.normal_ratio_eval)
    if cfg.class_balanced:
        norm_ratio_tr = 1.0 / (1.0 + cfg.abn_to_normal_train)   # 1.5:1 -> 0.40
    else:
        norm_ratio_tr = cfg.normal_ratio_eval
    tr = split_counts(cfg.n_train, norm_ratio_tr)

    need_norm = te[0] + va[0] + tr[0]
    need_abn  = te[1] + va[1] + tr[1]
    seen = set()
    norm_pool = _collect(normal_df, cfg.reports_dir, need_norm, seen)
    abn_pool  = _collect(abn_df,  cfg.reports_dir, need_abn,  seen)
    print(f"[data] collected normal={len(norm_pool)}/{need_norm}  "
          f"abnormal={len(abn_pool)}/{need_abn}")

    # dedupe normal templates by normalized surface form (your undersampling step)
    if cfg.class_balanced:
        seen_txt, dedup = set(), []
        for s in norm_pool:
            k = re.sub(r"\s+", " ", s["findings"].lower()).strip()
            if k not in seen_txt:
                seen_txt.add(k); dedup.append(s)
        norm_pool = dedup
        print(f"[data] normal after dedup: {len(norm_pool)}")

    def take(pool, n):  # pop n from front
        head, del_ = pool[:n], pool[n:]
        pool.clear(); pool.extend(del_); return head
    test  = take(norm_pool, te[0]) + take(abn_pool, te[1])
    val   = take(norm_pool, va[0]) + take(abn_pool, va[1])
    train = take(norm_pool, tr[0]) + take(abn_pool, tr[1])
    for s in (train, val, test): random.shuffle(s)
    print(f"[data] train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ============================================================
# 4. Dataset
# ============================================================
def make_input(sample: Dict, cfg: Config) -> str:
    if cfg.mode == "pure_baseline":
        return CONST_INPUT
    pos = sample["pos5"]
    cues = ", ".join(pos) if pos else "no acute cardiopulmonary findings"
    return f"Key findings to check: {cues}. Write a radiology report for this chest X-ray."


class ReportDS(Dataset):
    def __init__(self, samples, tok, cfg):
        self.s, self.tok, self.cfg = samples, tok, cfg
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        x = self.tok(make_input(s, self.cfg), max_length=self.cfg.max_source_length,
                     truncation=True)
        y = self.tok(text_target=s["findings"], max_length=self.cfg.max_target_length,
                     truncation=True)
        x["labels"] = y["input_ids"]
        return x


# ============================================================
# 5. Metric helpers
# ============================================================
# --- verbatim copy of compute_captioning_metrics from YOUR evaluate.py ---
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
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for pred, ref in zip(predictions, references):
            sc = scorer.score(ref, pred)
            r1.append(sc["rouge1"].fmeasure); r2.append(sc["rouge2"].fmeasure)
            rl.append(sc["rougeL"].fmeasure)
        results["ROUGE-1"] = round(np.mean(r1) * 100, 2)
        results["ROUGE-2"] = round(np.mean(r2) * 100, 2)
        results["ROUGE-L"] = round(np.mean(rl) * 100, 2)
    except Exception:
        pass
    results["CIDEr"] = round(Cider().compute_score(gts, res)[0] * 100, 2)
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        results["METEOR"] = round(Meteor().compute_score(gts, res)[0] * 100, 2)
    except Exception:
        pass
    return results
# --- end verbatim copy ---


def quick_bleu4(preds, refs) -> float:
    """Lightweight BLEU-4 used only for checkpoint selection (your val criterion)."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    sm = SmoothingFunction().method1
    return corpus_bleu([[r.split()] for r in refs], [p.split() for p in preds],
                       weights=(.25, .25, .25, .25), smoothing_function=sm) * 100


# ============================================================
# 6. Train + evaluate
# ============================================================
def run(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    train, val, test = build_splits(cfg)
    ds_tr, ds_va = ReportDS(train, tok, cfg), ReportDS(val, tok, cfg)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    val_refs = [s["findings"] for s in val]
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        dec = tok.batch_decode(preds, skip_special_tokens=True)
        return {"bleu4": quick_bleu4(dec, val_refs)}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.ckpt_dir,                # LOCAL — checkpoints stay off Drive
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio, lr_scheduler_type="cosine",
        max_grad_norm=cfg.grad_clip,
        label_smoothing_factor=cfg.label_smoothing,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        dataloader_num_workers=cfg.num_workers,
        bf16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        generation_num_beams=4,                 # faster for selection
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="bleu4",
        greater_is_better=True, save_total_limit=1,
        logging_steps=50, report_to="none", seed=cfg.seed,
    )
    from transformers import EarlyStoppingCallback
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stop_patience)],
    )
    trainer.train()

    # ---- final generation on test (matches your inference settings) ----
    model.eval(); device = model.device
    preds, refs = [], [s["findings"] for s in test]
    for i in range(0, len(test), cfg.batch_size):
        chunk = test[i:i + cfg.batch_size]
        enc = tok([make_input(s, cfg) for s in chunk], return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=cfg.max_source_length).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, num_beams=cfg.num_beams, max_length=cfg.max_target_length,
                min_length=cfg.min_length, length_penalty=cfg.length_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))

    # ---- metrics: identical code path to your IU-Xray pipeline ----
    metrics = compute_captioning_metrics(preds, refs)            # legacy (sanity only)
    metrics.update(compute_r2gen_nlg(preds, refs))              # R2Gen protocol -> Table 6
    try:
        labeler = get_chexbert_labeler(device="cuda" if torch.cuda.is_available() else "cpu")
        metrics.update(compute_clinical_f1_chexbert(preds, refs, labeler))
    except Exception as e:
        print(f"[Eval] CheXbert skipped — {e}")
    try:
        metrics.update(compute_radgraph_f1(preds, refs))
    except Exception as e:
        print(f"[Eval] RadGraph skipped — {e}")

    tag = f"{cfg.mode}{'_balanced' if cfg.class_balanced else ''}_seed{cfg.seed}"
    with open(os.path.join(cfg.output_dir, f"results_{tag}.json"), "w") as f:
        json.dump({"config": tag, "metrics": metrics,
                   "samples": [{"prediction": p, "reference": r}
                               for p, r in zip(preds, refs)]}, f, indent=2)

    print(f"\n===== {tag} =====")
    for k in ("R2Gen_BLEU_4", "R2Gen_CIDEr", "BLEU-4", "ROUGE-L", "CIDEr",
              "RadGraph_Simple", "ClinicalF1_chexbert_macro_F_5",
              "ClinicalF1_chexbert_micro_F_14"):
        if k in metrics:
            print(f"  {k:32s} {metrics[k]}")
    for c in FIVE:
        k = f"CheXbert5_{c}_F1"
        if k in metrics:
            print(f"  {k:32s} {metrics[k]}")
    return metrics


if __name__ == "__main__":
    # Three seeds, matching the IU-Xray runs. NOTE: change the Table 6 caption
    # from "four seeds" to "three seeds (42, 13, 87)" to match this.
    SEEDS = [42, 13, 87]

    runs = []
    for _sd in SEEDS:
        print(f"\n########## seed {_sd} ##########")
        runs.append(run(Config(seed=_sd)))

    # ---- aggregate the Table 6 rows: mean ± SD over seeds ----
    # Verify these key names against your evaluators' actual output on run 1.
    TABLE6_KEYS = [
        "R2Gen_BLEU_4", "R2Gen_CIDEr",
        "RadGraph_Simple", "RadGraph_Partial",
        "ClinicalF1_chexbert_macro_F_5",
    ]
    print("\n===== Table 6 (Pure Baseline, mean ± SD over seeds 42/13/87) =====")
    for _k in TABLE6_KEYS:
        _vals = [m[_k] for m in runs if _k in m]
        if _vals:
            print(f"  {_k:34s} {np.mean(_vals):.2f} ± {np.std(_vals):.2f}")
        else:
            print(f"  {_k:34s} (key not found — inspect metrics.keys())")
