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

> **Companion repository.** The Qwen2-VL and nearest-neighbour (NN) baselines used
> in the comparison tables are produced by the companion repository
> [`vlm-radiology-report`](https://github.com/Mikebbb123/vlm-radiology-report).
> Only the prediction dumps cross over; they are re-scored here under one protocol.
> See [Baseline reproduction](#baseline-reproduction).

---

## Repository layout (two variants)

This repo ships two self-contained variants in separate folders. Each folder runs
on its own — `cd` into it and launch directly, no shared imports across folders.

| Folder | Variant | Notes |
|--------|---------|-------|
| `main_model/` | **Full model (canonical)** | Paper headline numbers (BLEU-4 = 18.49). `use_visual_tokens=False`; supports the `--no_hint` / `--no_cls_loss` ablations. Also holds the sanity checks, plotting, baseline dumps, and paper analyses. **Start here.** |
| `visual_token_ablation/` | `+visual_token` ablation | Adds a zero-init gated visual-token path (`use_visual_tokens=True`). Reported as an ablation (BLEU-4 = 17.04); kept for reproducibility. Contains only the core training/eval modules. |

Unless you specifically want the visual-token ablation, use `main_model/`. The
two folders intentionally duplicate the **core** modules (`data.py`,
`evaluate.py`, `disease_t5.py`, etc.) so each trains and evaluates in isolation.

The extra tooling — sanity-check scripts, `make_plots.py`, and the `baselines/`
and `analysis/` directories — lives **only in `main_model/`**, since baselines
and figures are produced once for the full model.

`SETUP.md` (repo root) describes the isolated environments needed to reproduce the
cross-model comparison; see [Baseline reproduction](#baseline-reproduction).

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

## Files

### Core modules (present in both variant folders)

```
disease_t5.py            Main model (DiseaseT5)
train.py                 Training loop (composite early-stopping on micro-F1 + BLEU)
config.py                DataConfig / ModelConfig / TrainConfig (paths via env vars)
run_main.py              CLI entry point (train / eval / ablations)
data.py                  Dataset, X-ray transforms, collate
data_utils_compat.py     Regex disease-label extraction (fallback + token weighting)

evaluate.py              Full eval: R2Gen BLEU/METEOR/ROUGE + CheXbert F1 + RadGraph
chexbert_eval.py         CheXbert clinical F1 (via f1chexbert)
radgraph_eval.py         RadGraph factual-consistency F1

build_chexbert_cache.py  Build the CheXbert training-label cache (run ONCE first)
```

### main_model only — sanity checks & figures

```
check_cls_grad.py        Sanity check: cls_loss gradients reach the Perceiver
check_finding_weights.py Sanity check: finding-token weighting
decollapse_check.py      Sanity check: template collapse in generated reports
make_plots.py            Publication figures (multi-seed; standard/oracle, NN baseline)
```

### main_model only — baselines & analysis

```
baselines/
  r2gen_test_dump.py        Run the official R2Gen IU-Xray ckpt -> predictions JSON
  promptmrg_test_dump.py    Run the official PromptMRG ckpt (MIMIC->IU transfer) -> JSON
  score_r2gen.py            Score a dump through the same R2Gen-NLG/CheXbert/RadGraph pipeline

analysis/
  score_common_subset.py          Re-score all methods on the common n=349 subset (Table 1b)
  significance_test.py            Paired bootstrap CI + p-value; Wilcoxon / Cliff's delta
  chexpert_independence_check.py  Independent CheXpert rule-based labeler robustness check
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

> Reproducing the **cross-model comparison** (R2Gen / PromptMRG / Qwen2-VL / NN
> baselines) needs several mutually incompatible environments. See
> [`SETUP.md`](SETUP.md) and the [Baseline reproduction](#baseline-reproduction)
> section.

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
the image→findings predictor. Each eval writes a predictions dump
(`test_final.json` / `test_final_oracle.json`) with schema
`{"samples": [{"id", "prediction", "reference"}, ...]}`.

### 4. Figures

```bash
python make_plots.py --output_root ./outputs --annotation data/iu_xray/annotation.json \
                     --seeds 13 42 87 --figdir figures
python make_plots.py --mode both        # standard + oracle comparison
python make_plots.py --nn_preds ./dumps/nn_top1.json   # add NN top-1 baseline
```

Each figure is independent: if its source data file is missing, the script warns
and continues. Outputs land in `figures/` (disease distribution, training curves,
per-disease F1, lexical-vs-clinical scatter, seed-summary table, qualitative
examples).

---

## Baseline reproduction

The cross-model comparison is run **outside** the training pipeline: each baseline
dumps its predictions to JSON, and every method is re-scored through *the same*
R2Gen-NLG / CheXbert / RadGraph pipeline. Because the baselines have mutually
incompatible dependencies (different `transformers` / `numpy`), they run in
**separate environments** — see [`SETUP.md`](SETUP.md) for the five isolated setups
(A: this repo + scoring, B: R2Gen, C: PromptMRG, D: optional CheXpert labeler,
E: Qwen2-VL / NN via the companion repo).

The **Qwen2-VL + LoRA**, **Qwen2-VL (oracle)**, and **NN retrieval** baselines are
produced in our companion repository,
[`vlm-radiology-report`](https://github.com/Mikebbb123/vlm-radiology-report); only
their prediction dumps are copied here. Re-scoring them through this repo's
pipeline is what makes the comparison protocol-consistent: clinical scores match
across the two repos (same CheXbert/RadGraph code), but the NLG protocol differs,
so the dumps must be re-scored here rather than copied as numbers.

Typical flow (from `main_model/`):

```bash
# 1a. R2Gen / PromptMRG dumps (each in its OWN env — see SETUP.md B / C)
python baselines/r2gen_test_dump.py        # -> dumps/r2gen_test_generated.json
python baselines/promptmrg_test_dump.py    # -> dumps/promptmrg_test_generated.json

# 1b. Qwen2-VL / NN dumps (companion repo, env E — see SETUP.md E)
#     produce them in vlm-radiology-report, then copy in, e.g.:
#       dumps/qwen_test_generated.json
#       dumps/qwen_oracle_test_generated.json
#       dumps/nn_top1_test_generated.json

# 2. Score a dump through our pipeline (env A)
python baselines/score_r2gen.py            # scores YOUR_DUMP vs a baseline dump

# 3. Common-subset comparison + significance (env A)
python analysis/score_common_subset.py     # re-score all methods on the common n=349 ids
python analysis/significance_test.py       # paired bootstrap CI + p-value
python analysis/chexpert_independence_check.py   # optional, needs env D
```

Dump schema for baselines is `[{"id", "generated"(, "view")}]`; our own dumps use
`{"samples": [{"id", "prediction", "reference"}, ...]}`. The scoring scripts accept
either (they read the `generated` / `prediction` field). External repos,
checkpoints, and generated dumps are git-ignored — clone/download them locally.

> **Framing notes (keep in the paper).** PromptMRG's released model is
> MIMIC-CXR-trained and applied to IU-Xray as cross-dataset transfer; it is
> retrieval-augmented and single-view. Qwen2-VL + LoRA is a generic-VLM transfer
> baseline with no radiology-specific pretraining. The oracle rows leak GT disease
> labels and are upper-bound references only. The CheXpert independence check is a
> *parameter*-independent sanity check, not proof of labeler-independent content —
> CheXbert was distilled to imitate the CheXpert rule-based labeler. Bootstrap CIs
> reflect test-set sampling only, not baseline retraining variance.

---

## Results

### Headline (full model, IU-Xray test, 590 samples)

Mean ± SD over 3 seeds. NLG metrics use the R2Gen official protocol.

| Metric             | Value         |
|--------------------|---------------|
| BLEU-4             | 18.49 ± 0.23  |
| METEOR             | 20.25 ± 1.13  |
| ROUGE-L            | 37.59 ± 1.33  |
| CheXbert micro-F1(14) | 55.10 ± 2.17 |
| CheXbert macro-F1(5)  | 18.87 ± 1.65 |
| RadGraph-Complete  | 30.43 ± 4.44  |

### Comparison with baselines (590-sample test, R2Gen split)

All rows re-scored under the same protocol. ★ proposed method; † leaked-label
upper-bound references (excluded from the non-leaked comparison). RG = RadGraph
entity F1. Best among non-leaked rows in bold.

| Method | BLEU-4 | ROUGE-L | METEOR | CheXbert-5 micro/macro | CheXbert-14 micro/macro | RG-Simple | RG-Partial |
|--------|--------|---------|--------|------------------------|-------------------------|-----------|------------|
| Qwen2-VL + LoRA | 12.92 | 34.62 | 17.18 | 0.00 / 0.00 | 51.63 / 5.26 | 35.75 | 33.15 |
| NN Retrieval (zero training) | 12.87 | 32.54 | 17.42 | 25.33 / 17.73 | 41.45 / 15.47 | 33.06 | 30.29 |
| R2Gen (official ckpt) | 17.40 | 36.62 | **20.38** | 0.00 / 0.00 | 53.08 / 5.09 | 37.81 | 35.11 |
| **Ours (Full Model)** ★ | **18.49 ± 0.23** | **37.59 ± 1.33** | 20.25 ± 1.13 | **30.31 ± 2.21 / 18.87 ± 1.65** | **55.10 ± 2.17 / 12.33 ± 0.59** | **41.67 ± 3.78** | **38.53 ± 3.26** |
| *Qwen2-VL (oracle)* † | 20.06 | 39.55 | 20.33 | 23.78 / 11.11 | 54.51 / 12.77 | 43.08 | 40.19 |
| *Ours (oracle cue)* † | 18.47 ± 0.33 | 37.84 ± 1.31 | 20.31 ± 1.11 | 79.60 ± 15.18 / 49.58 ± 14.25 | 67.59 ± 3.95 / 23.83 ± 5.86 | 42.21 ± 3.40 | 39.10 ± 2.90 |

The full model obtains non-zero CheXbert-5 efficacy on the target pathologies,
where the Qwen2-VL and R2Gen baselines collapse to 0.00 despite non-trivial lexical
and 14-class scores — the template-collapse pattern the paper analyses.

### Common-subset comparison with PromptMRG (349 shared studies)

PromptMRG's IU-Xray partition overlaps the R2Gen test split on only 349 studies, so
all methods are re-scored on that shared subset.

| Method | BLEU-4 | ROUGE-L | METEOR | CheXbert-5 micro/macro | CheXbert-14 micro/macro | RG-Simple | RG-Partial |
|--------|--------|---------|--------|------------------------|-------------------------|-----------|------------|
| Qwen2-VL + LoRA | 9.92 | 31.82 | 14.80 | 0.00 / 0.00 | 21.57 / 3.06 | 32.71 | 30.49 |
| NN Retrieval (zero training) | 10.69 | 29.77 | 15.25 | 28.02 / 19.40 | 30.75 / 15.89 | 29.84 | 27.16 |
| R2Gen (official ckpt) | 11.61 | 32.01 | 17.01 | 0.00 / 0.00 | 22.45 / 2.75 | 32.57 | 30.14 |
| PromptMRG | 10.77 | 31.67 | 15.42 | **44.34 / 29.24** | **39.88 / 22.40** | 30.82 | 29.17 |
| **Ours (Full Model)** ★ | **14.65 ± 0.35** | **34.87 ± 1.22** | **17.62 ± 0.99** | 31.61 ± 1.69 / 19.71 ± 1.31 | 29.28 ± 0.65 / 10.35 ± 0.52 | **37.30 ± 3.96** | **34.55 ± 3.62** |
| *Ours (oracle cue)* † | 14.52 ± 0.37 | 35.15 ± 1.07 | 17.65 ± 0.95 | 79.60 ± 15.18 / 49.58 ± 14.25 | 46.48 ± 7.21 / 21.58 ± 5.94 | 37.96 ± 3.24 | 35.35 ± 2.94 |

On the common subset, PromptMRG leads CheXbert-5 F1 under a retrieval-augmented
cross-dataset setting (paired bootstrap Δ = 12.46, 95% CI [3.51, 21.89], p = 0.008),
while our retrieval-free in-domain model leads NLG and RadGraph. Our advantage over
R2Gen is significant (Δ = 31.88, 95% CI [22.60, 40.71], p < 0.001); the gap over NN
Retrieval is not (Δ = 3.86, 95% CI [−6.09, 13.17], p = 0.44).

All comparison tables, the significance tests, and the independent-labeler check are
reproducible via the `baselines/` and `analysis/` scripts ([Baseline
reproduction](#baseline-reproduction)).

**Limitations.** Per-class F1 for rare findings is high-variance: e.g.
Consolidation has very few positive cases in the split, and one seed
underperforms on Atelectasis/Effusion. Treat per-class numbers for low-support
classes as noisy.

---

## Citation

If you use this code, please cite the main paper:

```bibtex
@article{TODO_your_paper_key,
  title   = {TODO: paper title},
  author  = {TODO: authors},
  journal = {TODO},
  year    = {TODO}
}
```

The Qwen2-VL and NN baselines are produced by the companion repository
[`vlm-radiology-report`](https://github.com/Mikebbb123/vlm-radiology-report).

---

## Acknowledgements

This work builds on:

- [R2Gen](https://github.com/zhjohnchan/R2Gen) — IU-Xray split + official NLG protocol + baseline
- [PromptMRG](https://github.com/jhb86253817/PromptMRG) — retrieval-augmented baseline
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) — generic-VLM baseline (via the companion repo)
- [SciFive](https://huggingface.co/razent/SciFive-base-Pubmed_PMC) — biomedical T5 backbone
- [torchxrayvision](https://github.com/mlmed/torchxrayvision) — frozen vision encoder + pathology classifier
- [f1chexbert](https://pypi.org/project/f1chexbert/) — CheXbert clinical F1
- [CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler) — independent rule-based label check
- [RadGraph](https://pypi.org/project/radgraph/) — factual-consistency F1
- [PEFT](https://github.com/huggingface/peft) — LoRA

Companion repository:
[`vlm-radiology-report`](https://github.com/Mikebbb123/vlm-radiology-report) — the
Qwen2-VL fine-tuning and NN-retrieval empirical study these baselines come from.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

