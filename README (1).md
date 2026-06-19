# DiseaseT5: Disease-Conditioned Radiology Report Generation

Generate radiology reports from dual-view (frontal + lateral) chest X-rays on the
**IU-Xray** dataset (R2Gen official split). The model conditions a SciFive-T5
decoder on disease findings predicted from the image, and is trained with a
finding-token-weighted objective that prevents the template-collapse failure mode
common in this task.

> **Note on naming.** Several modules and identifiers still carry a legacy `RAG`
> prefix (`DualViewRAGDataset`, `rag_collate_fn`, `RetrievalConfig`). The current
> model is **not** retrieval-augmented — the retrieval path was removed and replaced
> by disease-conditioned generation. The names are kept only to avoid a risky
> rename right before release; they are inert.

---

## Repository layout (two variants)

This repo ships two self-contained variants in separate folders. Each folder runs
on its own — `cd` into it and launch directly, no shared imports across folders.

| Folder | Variant | Notes |
|--------|---------|-------|
| `main_model/` | **Full model (canonical)** | Paper headline numbers (BLEU-4 = 18.49). `use_visual_tokens=False`; supports the `--no_hint` / `--no_cls_loss` ablations. **Start here.** |
| `visual_token_ablation/` | `+visual_token` ablation | Adds a zero-init gated visual-token path (`use_visual_tokens=True`). Reported as an ablation (BLEU-4 = 17.04); kept for reproducibility. |

Unless you specifically want the visual-token ablation, use `main_model/`. The
two folders intentionally duplicate the shared modules (`data.py`, `evaluate.py`,
etc.) so each is runnable in isolation.

All commands below are run **from inside a variant folder**, e.g. `cd main_model`.

---

## Method

```
frontal ─┐
         ├─► frozen ResNet-50 (torchxrayvision) ─► layer4 feats [B, 2048, 16, 16]
lateral ─┘                │
                          ▼
        Perceiver Visual Resampler (32 learnable queries, 2 cross-attn layers)
                          │  ─► visual tokens [B, 32, 768]
                          ▼
        ┌──────────────────────────────────┐
        │ disease head (768→384→5 MLP)      │  auxiliary cls_loss
        │   gradients flow back into the    │  (makes visual tokens
        │   Perceiver                       │   disease-discriminative)
        └──────────────────────────────────┘
                          │
   disease findings ──► text prompt ("Key findings to check: ...")
                          ▼
        SciFive-T5 (LoRA r=32, α=64) ─► radiology report
```

Key components:

- **Disease hints** are produced at inference by an ensemble of the frozen
  torchxrayvision classifier and the trained disease head, then injected as a
  text prompt. During training, ground-truth findings are scheduled-sampled with
  the predicted ones.
- **Finding-token-weighted cross-entropy** (×5.0 on tokens inside positive-finding
  sentences) prevents the decoder from ignoring clinically salient tokens.
- **Training labels come from CheXbert** (cached once), so the training signal is
  aligned with the CheXbert F1 evaluation metric.

Target findings (5-class): Cardiomegaly, Edema, Consolidation, Atelectasis,
Pleural Effusion.

---

## Files (within each variant folder)

```
disease_t5.py            Main model (DiseaseT5)
train.py                 Training loop (composite early-stopping on micro-F1 + BLEU)
config.py                DataConfig / ModelConfig / TrainConfig (paths via env vars)
run_main.py              CLI entry point (train / eval / ablations)
data.py                  Dataset, X-ray transforms, collate
data_utils_compat.py     Regex disease-label extraction (fallback + token weighting)

evaluate.py              Full eval: R2Gen BLEU/METEOR/ROUGE + CheXbert F1 + RadGraph
eval_preds.py            Standalone eval for a predictions JSON (no model needed)
chexbert_eval.py         CheXbert clinical F1 (via f1chexbert)
radgraph_eval.py         RadGraph factual-consistency F1

build_chexbert_cache.py  Build the CheXbert training-label cache (run ONCE first)
check_cls_grad.py        Sanity check: cls_loss gradients reach the Perceiver
check_finding_weights.py Sanity check: finding-token weighting
decollapse_check.py      Sanity check: template collapse in generated reports
make_plots.py            Publication figures   (TODO: add)
```

---

## Installation

Python ≥ 3.9, a CUDA-capable GPU recommended.

```bash
pip install -r requirements.txt
```

`f1chexbert` and `radgraph` download their own model weights on first use.

`f1chexbert` monkey-patches `transformers.PreTrainedTokenizerBase._batch_encode_plus`;
the code restores the original after every call (`_undo_tokenizer_patch()`), so the
T5 tokenizer used during training is never corrupted. If you upgrade `transformers`,
re-run `check_finding_weights.py` to confirm offset-mapping still works.

### Optional: R2Gen official NLG protocol

BLEU/METEOR/ROUGE reported for paper comparison use R2Gen's normalization +
vendored `pycocoevalcap`. Clone the official repo and point an env var at it:

```bash
git clone https://github.com/zhjohnchan/R2Gen.git
export R2GEN_REPO=/path/to/R2Gen
sudo apt-get install -y default-jre   # METEOR needs Java
```

If `R2GEN_REPO` / Java are unavailable, the `R2Gen_*` metrics are silently skipped
and only the legacy in-house metrics are reported (training is never interrupted).

---

## Data preparation

The IU-Xray images and the R2Gen split annotation are **not** redistributed here.

1. Download IU-Xray images and the R2Gen `annotation.json`
   (R2Gen official split): https://github.com/zhjohnchan/R2Gen
2. Arrange them like this (or point the env vars below anywhere you like):

```
data/iu_xray/
├── annotation.json
└── images/
    └── <case_id>/{0.png,1.png}
```

Paths are configured via environment variables (defaults shown):

```bash
export IU_XRAY_ANNOTATION=./data/iu_xray/annotation.json
export IU_XRAY_IMAGES=./data/iu_xray/images
export OUTPUT_DIR=./outputs/disease_t5
export CACHE_DIR=./cache
```

---

## Usage

Run everything from inside a variant folder. Install once from the repo root,
then `cd` in:

```bash
pip install -r requirements.txt   # from repo root
cd main_model                     # or visual_token_ablation
```

### 1. Build the CheXbert label cache (run once)

```bash
python build_chexbert_cache.py
```

Writes `${CACHE_DIR}/chexbert_labels.pt`. Training **requires** this — `data.py`
raises if a report is missing from the cache, to guarantee labels are never a
silent CheXbert/regex mix.

### 2. Train

```bash
python run_main.py --device cuda
```

Useful flags:

```bash
python run_main.py --seed 13            # per-seed output dir (multi-seed runs)
python run_main.py --smoke_test         # 2 epochs on a tiny slice
python run_main.py --no_hint            # ablation A3: remove disease hints
python run_main.py --no_cls_loss        # ablation A4: remove auxiliary cls loss
```

The best checkpoint (`best.pt`) is selected by a composite score
`micro-F1(5) + BLEU-4`, gated by a `BLEU-4 ≥ 8.0` floor so a degenerate run can't
win on noisy F1.

### 3. Evaluate

`run_main.py` runs the full test evaluation automatically after training. To
evaluate an existing checkpoint:

```bash
python run_main.py --eval_only --checkpoint outputs/disease_t5/best.pt
python run_main.py --eval_only --checkpoint outputs/disease_t5/best.pt --oracle
```

`--oracle` feeds ground-truth findings as the hint, isolating decoder quality from
the image→findings predictor.

### 4. Standalone evaluation of a predictions file

```bash
python eval_preds.py preds.json --annotation data/iu_xray/annotation.json
python eval_preds.py preds.json --simple   # fast collapse check only
```

Input format: `[{"id": "...", "reference": "...", "generated": "..."}, ...]`

---

## Results (IU-Xray test, 590 samples)

Full model, mean ± SD over 3 seeds. NLG metrics use the R2Gen official protocol.

| Metric             | Value         |
|--------------------|---------------|
| BLEU-4             | 18.49 ± 0.23  |
| METEOR             | 20.25 ± 1.13  |
| ROUGE-L            | 37.59 ± 1.33  |
| CheXbert micro-F1(14) | 55.10 ± 2.17 |
| CheXbert macro-F1(5)  | 18.87 ± 1.65 |
| RadGraph-Complete  | 30.43 ± 4.44  |

**Limitations.** Per-class F1 for rare findings is high-variance: e.g.
Consolidation has very few positive cases in the split, and one seed
underperforms on Atelectasis/Effusion. Treat per-class numbers for low-support
classes as noisy.

---

## Citation

If you use this code, please cite:

```bibtex
@article{TODO_your_paper_key,
  title   = {TODO: paper title},
  author  = {TODO: authors},
  journal = {TODO},
  year    = {TODO}
}
```

---

## Acknowledgements

This work builds on:

- [R2Gen](https://github.com/zhjohnchan/R2Gen) — IU-Xray split + official NLG protocol
- [SciFive](https://huggingface.co/razent/SciFive-base-Pubmed_PMC) — biomedical T5 backbone
- [torchxrayvision](https://github.com/mlmed/torchxrayvision) — frozen vision encoder + pathology classifier
- [f1chexbert](https://pypi.org/project/f1chexbert/) — CheXbert clinical F1
- [RadGraph](https://pypi.org/project/radgraph/) — factual-consistency F1
- [PEFT](https://github.com/huggingface/peft) — LoRA

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
