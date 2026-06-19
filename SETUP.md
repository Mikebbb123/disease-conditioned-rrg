# Baseline reproduction — environment setup

Reproducing the cross-model comparison (Table 1 / 1b) involves running several
external baselines through our scoring pipeline. **These baselines have mutually
incompatible dependencies** — most importantly different `transformers` /
`numpy` versions — so they must NOT share one Python environment. Use the five
isolated setups below.

External repos (`R2Gen/`, `PromptMRG/`, `chexpert-labeler/`, `NegBio/`,
`vlm-radiology-report/`), downloaded checkpoints (`ckpts/`), and generated dumps
(`dumps/`) are all git-ignored — clone / download them locally, don't commit them.

| # | Environment | Used by | Key constraint |
|---|-------------|---------|----------------|
| A | Main (this repo) | our model, all scoring scripts | `transformers<5`, recent numpy |
| B | R2Gen | `baselines/r2gen_test_dump.py` | `numpy==1.23.5` (old) |
| C | PromptMRG | `baselines/promptmrg_test_dump.py` | `transformers==4.25.0`, isolated venv |
| D | CheXpert labeler | `analysis/chexpert_independence_check.py` | conda env + Java 8 (optional) |
| E | Qwen2-VL / NN (`vlm-radiology-report` repo) | Qwen2-VL + LoRA, Qwen oracle, NN retrieval dumps | `transformers>=4.45,<5`, isolated venv |

---

## A. Main environment (our model + scoring)

```bash
pip install -r requirements.txt

# CheXbert weights (used by f1chexbert / the CheXbert-5 metric)
mkdir -p ~/.cache/chexbert
wget -q https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/main/chexbert.pth \
     -O ~/.cache/chexbert/chexbert.pth

# METEOR (R2Gen NLG protocol) needs Java
sudo apt-get install -y default-jre        # or openjdk-11-jre-headless

# R2Gen repo — provides the official NLG cleaner + vendored pycocoevalcap
git clone https://github.com/zhjohnchan/R2Gen.git
export R2GEN_REPO=$PWD/R2Gen
```

**Two gotchas from the original notebooks** (handle if you hit import errors):
- Do **not** keep a pip-installed `pycocoevalcap` alongside R2Gen — R2Gen vendors
  its own and the two clash. If you installed it, `pip uninstall -y pycocoevalcap`.
- `pip uninstall -y torchao` if its presence breaks the transformers import.

---

## B. R2Gen baseline dump

R2Gen requires an **old numpy** (`<1.24`). Run it in its own venv so it doesn't
downgrade numpy in environment A.

```bash
python -m venv .venv-r2gen && source .venv-r2gen/bin/activate
pip install torch torchvision numpy==1.23.5
git clone https://github.com/zhjohnchan/R2Gen.git   # if not already cloned
```

Download the **official R2Gen IU-Xray checkpoint** (`model_iu_xray.pth`) from the
R2Gen project, place it at `./ckpts/model_iu_xray.pth`, then:

```bash
# generate predictions
python baselines/r2gen_test_dump.py        # -> dumps/r2gen_test_generated.json
# score them through our pipeline (run in environment A)
python baselines/score_r2gen.py            # -> *_r2gen_scored.json
```

`r2gen_test_dump.py` drives R2Gen's own `main_test.py` parser so every default
matches the checkpoint (beam=3, max_len=60 — the published IU setup; do not change).

---

## C. PromptMRG baseline dump

PromptMRG pins old `transformers==4.25.0` etc. — **isolate it completely** in a
dedicated venv (the original used `uv`):

```bash
pip install uv
uv venv --python 3.10 .venv-pmrg
uv pip install --python .venv-pmrg/bin/python \
    torch torchvision \
    transformers==4.25.0 huggingface_hub==0.11.1 tokenizers==0.13.2 \
    timm fairscale opencv-python scipy pandas scikit-learn gdown

git clone https://github.com/jhb86253817/PromptMRG.git
```

Download PromptMRG's assets (IDs from the original notebook — verify they're still
current on the PromptMRG project page):

```bash
mkdir -p PromptMRG/data/iu_xray PromptMRG/data/mimic_cxr PromptMRG/results/promptmrg
gdown 1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS -O PromptMRG/data/iu_xray/iu_annotation_promptmrg.json
gdown 1Zyq-84VOzc-TOZBzlhMyXLwHjDNTaN9A -O PromptMRG/data/mimic_cxr/clip_text_features.json
gdown 1s4AoLnnGOysOQkdILhhFCL59LyQtRHGa -O PromptMRG/results/promptmrg/model_best.pth
ln -sf /path/to/iu_xray/images PromptMRG/data/iu_xray/images
```

Run the dump **with the isolated interpreter**, then score in environment A:

```bash
.venv-pmrg/bin/python baselines/promptmrg_test_dump.py   # -> dumps/promptmrg_test_generated.json
```

> Note for the paper: PromptMRG's released model is MIMIC-CXR-trained and applied
> to IU-Xray as cross-dataset transfer (no IU-Xray training). State this when
> reporting its numbers.

---

## D. CheXpert rule-based labeler (optional robustness check)

Only needed for `analysis/chexpert_independence_check.py` (Sect. 6.2). Heavy and
finicky — a separate conda env, Java 8 (CoreNLP), and the BLLIP parser model.

```bash
# In Colab the original used condacolab (restarts the runtime); locally use conda/mamba.
sudo apt-get install -y openjdk-8-jdk-headless   # CoreNLP needs Java 8

git clone https://github.com/stanfordmlgroup/chexpert-labeler
git clone https://github.com/ncbi-nlp/NegBio

cd chexpert-labeler
# environment.yml needs a gcc fix (upstream issue #20):
sed -i '/dependencies:/a\  - gcc_linux-64\n  - gxx_linux-64' environment.yml
conda env create -f environment.yml
conda run -n chexpert-label python -m nltk.downloader universal_tagset punkt wordnet
conda run -n chexpert-label python -c \
    "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"
export PYTHONPATH=$PWD/../NegBio:$PYTHONPATH
```

Then point `CHEXPERT_REPO` / `NEGBIO_REPO` env vars at the two clones before
running `analysis/chexpert_independence_check.py`.

> Independence caveat (keep in the paper): CheXbert was distilled to imitate the
> CheXpert rule-based labeler, so they are not statistically independent. Treat
> this as a parameter-independent sanity check, not proof of labeler-independent
> clinical content.

---

## E. Qwen2-VL / NN baselines (`vlm-radiology-report` repo)

The **Qwen2-VL + LoRA**, **Qwen2-VL (oracle)**, and **NN retrieval** rows in
Table 1 / 1b are produced in our separate empirical-study repo
(`vlm-radiology-report`), not here. Only the prediction **dumps** cross over: we
copy them into `dumps/` and re-score them through *this* repo's pipeline
(environment A), so every method is scored under the same R2Gen-NLG / CheXbert /
RadGraph protocol. (Clinical scores match across the two repos because both use
the same CheXbert/RadGraph code; only the NLG protocol differs, which is exactly
why the dumps must be re-scored in environment A rather than copied as numbers.)

Qwen2-VL inference needs a **recent** `transformers` (`>=4.45`) plus
`qwen-vl-utils`, which conflicts with environments B/C — isolate it.

```bash
git clone https://github.com/Mikebbb123/vlm-radiology-report.git
cd vlm-radiology-report

python -m venv .venv-qwen && source .venv-qwen/bin/activate
pip install "transformers>=4.45,<5" accelerate peft qwen-vl-utils \
    torch torchvision torchxrayvision nltk rouge-score
```

Provide the IU-Xray data and the trained Qwen2-VL LoRA adapter (see that repo's
README; the adapter is git-ignored — download or train it there). Then generate
the three dumps:

```bash
# --- Qwen2-VL + LoRA (no hint = the plain baseline row) ---
python evaluate.py --no_disease \
    --model_path .../lora_discourse \
    --output_file eval_qwen.json          # -> eval_qwen_preds.json

# --- Qwen2-VL (oracle): GT findings routed through the CAT hint channel ---
python make_oracle_hints.py --output_file oracle_hints.json
python evaluate.py --use_cat_hints \
    --cat_hint_file oracle_hints.json \
    --model_path .../lora_discourse \
    --output_file eval_qwen_oracle.json   # -> eval_qwen_oracle_preds.json

# --- NN retrieval (zero training) ---
python vff/precompute_densenet_features.py   # builds densenet_feats.npz (once)
python nn/NN.py                              # -> nn_top1_test_generated.json
```

`evaluate.py` writes the full per-sample predictions to `<output_file>_preds.json`
as `[{"id", "reference", "generated"}, ...]`; `nn/NN.py` writes the same schema.
These are consumed directly by our scorers (they read the `generated` field).

Copy the dumps into this repo and re-score in environment A:

```bash
cp eval_qwen_preds.json         /path/to/disease-rrg/main_model/dumps/qwen_test_generated.json
cp eval_qwen_oracle_preds.json  /path/to/disease-rrg/main_model/dumps/qwen_oracle_test_generated.json
cp nn_top1_test_generated.json  /path/to/disease-rrg/main_model/dumps/

cd /path/to/disease-rrg/main_model       # environment A
python baselines/score_r2gen.py          # per-method scoring
python analysis/score_common_subset.py   # 349-study common subset (Table 1b)
python analysis/significance_test.py     # paired bootstrap CI + p-value
```

> Note for the paper: Qwen2-VL + LoRA is a generic-VLM transfer baseline (no
> radiology-specific pretraining); the oracle row leaks GT disease labels and is
> an upper-bound reference only.

