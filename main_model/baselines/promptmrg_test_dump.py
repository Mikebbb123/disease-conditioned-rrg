"""
promptmrg_test_dump.py   (mirror of your r2gen_test_dump.py)
=============================================================
Load the OFFICIAL PromptMRG checkpoint (MIMIC-trained), run inference on the
IU-Xray transfer set, and dump generated reports (+ ids) to JSON so they can be
scored through YOUR own R2Gen-NLG / CheXbert-5 / RadGraph pipeline for a
same-protocol Table-1 comparison — exactly as you did for R2Gen.

NO training. Just test + dump.  In Colab:
    python promptmrg_test_dump.py

IMPORTANT — PromptMRG vs your setup (note these in the paper):
  * PromptMRG's released model is trained on MIMIC-CXR and applied to IU-Xray as
    a cross-dataset transfer (it has no IU-Xray training script). Report it as a
    stronger-but-not-data-matched disease-conditioned reference, like Qwen2-VL.
  * It is retrieval-augmented (cross-modal enhancement over a CLIP report DB),
    which your method deliberately forgoes ("no retrieval").
  * On IU it generates from a SINGLE (frontal) view; your model uses dual view.
None of these block the comparison — every method is re-scored on the SAME 590
test ids through the SAME pipeline — but they belong in the comparison framing.

PREREQUISITES (all from PromptMRG's README Google-Drive links; run in their env):
  1. Clone + install PromptMRG's own conda env (transformers is version-pinned;
     generate() uses the 4.25-era max_new_tokens API — do NOT use your paper's env).
         git clone https://github.com/jhb86253817/PromptMRG.git
         conda create -n promptmrg python=3.10 && conda activate promptmrg
         pip install -r PromptMRG/requirements.txt
  2. Put under PROMPTMRG_REPO/data/ :
         data/iu_xray/images/                         (R2Gen IU images — you have these)
         data/iu_xray/iu_annotation_promptmrg.json    (has labels + clip_indices)
         data/mimic_cxr/clip_text_features.json       (CLIP feature DB; loaded even for IU)
  3. The MIMIC-trained checkpoint (raw state_dict), e.g. results/promptmrg/model_best.pth
  chexbert.pth is NOT needed here (we bypass their Tester).
"""

import os
import sys
import json

# ============================================================
# CONFIG  — edit these
# ============================================================
PROMPTMRG_REPO = os.environ.get("PROMPTMRG_REPO", "./PromptMRG")         # the cloned repo (has main_test.py)
IMAGE_DIR  = "./PromptMRG/data/iu_xray/images/"                          # IU images dir
ANN_PATH   = "./PromptMRG/data/iu_xray/iu_annotation_promptmrg.json"     # THEIR IU annotation (labels+clip_indices)
CKPT       = "./ckpts/model_promptmrg_20240305.pth"                     # <-- downloaded official MIMIC ckpt
DUMP_PATH  = "./dumps/promptmrg_test_generated.json"

# Match PromptMRG's published IU setup (test_iu_xray.sh) — do NOT change.
GEN_MAX_LEN = 110
GEN_MIN_LEN = 60
BEAM_SIZE   = 3
CLIP_K      = 21
SEED        = 456789
IMAGE_SIZE  = 224

# ============================================================
# make the repo importable
# ============================================================
assert os.path.isdir(PROMPTMRG_REPO) and os.path.exists(os.path.join(PROMPTMRG_REPO, "main_test.py")), \
    f"没在 {PROMPTMRG_REPO} 找到 main_test.py，先确认 PROMPTMRG_REPO 路径对、仓库已 clone"
if PROMPTMRG_REPO not in sys.path:
    sys.path.insert(0, PROMPTMRG_REPO)
# the dataset hard-codes a relative './data/mimic_cxr/clip_text_features.json',
# so we MUST run from the repo root or that open() fails.
os.chdir(PROMPTMRG_REPO)

import torch
import numpy as np

# ---- build args via the repo's OWN parser so every default matches the ckpt ----
sys.argv = [
    "main_test.py",
    "--image_dir", IMAGE_DIR,
    "--ann_path", ANN_PATH,
    "--dataset_name", "iu_xray",
    "--image_size", str(IMAGE_SIZE),
    "--gen_max_len", str(GEN_MAX_LEN),
    "--gen_min_len", str(GEN_MIN_LEN),
    "--batch_size", "16",
    "--beam_size", str(BEAM_SIZE),
    "--clip_k", str(CLIP_K),
    "--seed", str(SEED),
    "--load_pretrained", CKPT,
]
from main_test import parse_agrs                  # noqa: E402  (PromptMRG's entry file)
args = parse_agrs()

# seeds for parity with their run (does not affect beam search determinism much)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from transformers import BertTokenizer            # noqa: E402
from dataset import create_dataset_test, create_loader   # noqa: E402
from models.blip import blip_decoder              # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# ============================================================
# tokenizer  (must replicate main_test.py EXACTLY — adds [DEC] + 4 score tokens)
# ============================================================
print("[PromptMRG] building tokenizer …")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})
tokenizer.add_tokens(["[BLA]", "[POS]", "[NEG]", "[UNC]"])

# ============================================================
# test dataset + loader  (shuffle=False => dataset order is preserved)
# ============================================================
print("[PromptMRG] building IU test dataset + loader …")
test_dataset = create_dataset_test("generation_%s" % args.dataset_name, tokenizer, args)
print("[PromptMRG] number of testing samples: %d" % len(test_dataset))
test_loader = create_loader([test_dataset], [None], batch_size=[args.batch_size],
                            num_workers=[4], is_trains=[False], collate_fns=[None])[0]

# ============================================================
# model  (same prompt-length placeholder as main_test.py) + load full state_dict
# ============================================================
print("[PromptMRG] building model …")
prompt_temp = " ".join(["[BLA]"] * 18) + " "      # 18 conditions, length placeholder only
model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)

print(f"[PromptMRG] loading checkpoint: {CKPT}")
try:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
except TypeError:                                  # older torch w/o weights_only kwarg
    ckpt = torch.load(CKPT, map_location="cpu")
# the released PromptMRG ckpt is a RAW state_dict; be defensive about wrappers
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state = ckpt["state_dict"]
elif isinstance(ckpt, dict) and "model" in ckpt:
    state = ckpt["model"]
else:
    state = ckpt
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[PromptMRG] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
      f"(both should be ~0; large mismatch => wrong ckpt/vocab or wrong transformers version).")
model = model.to(device)
model.eval()

# ============================================================
# id recovery  — the loader does NOT yield ids, so we read them from the
# ordered annotation list. shuffle=False + drop_last=False guarantees the
# prediction order equals test_dataset.ann order.
# IU id convention must match YOUR R2Gen ids (e.g. 'CXR1000_IM-0003') so that
# score_r2gen.py can align by id. We prefer an explicit 'id' field; otherwise we
# derive it from the image_path directory, which equals the R2Gen study id.
# ============================================================
def _derive_id(ann):
    if "id" in ann and ann["id"]:
        return ann["id"]
    ip = ann["image_path"]
    ip0 = ip[0] if isinstance(ip, (list, tuple)) else ip
    d = os.path.dirname(ip0)                       # 'CXR1000_IM-0003/0.png' -> 'CXR1000_IM-0003'
    return d if d else os.path.splitext(ip0)[0]

ids_in_order = [_derive_id(a) for a in test_dataset.ann]
print(f"[PromptMRG] sample ids: {ids_in_order[:5]}  <-- eyeball these against YOUR_DUMP ids")

# ============================================================
# inference + dump  (id-aligned JSON, same schema spirit as your r2gen dump)
# ============================================================
print("[PromptMRG] generating on IU test split …")
all_res, all_gts = [], []
with torch.no_grad():
    for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(test_loader):
        images = images.to(device)
        clip_memory = clip_memory.to(device)
        reports, _, _ = model.generate(
            images, clip_memory,
            sample=False,
            num_beams=args.beam_size,
            max_length=args.gen_max_len,
            min_length=args.gen_min_len,
        )
        all_res.extend(reports)
        all_gts.extend(captions)                   # PromptMRG's pre-cleaned GT (for sanity NLG only)
        if batch_idx % 10 == 0:
            print(f"  {batch_idx}/{len(test_loader)}")

assert len(all_res) == len(ids_in_order), \
    (f"prediction count {len(all_res)} != annotation count {len(ids_in_order)}; "
     f"id alignment would be wrong. Check that the loader ran in order.")

records = [{"id": sid, "generated": pred, "ground_truth_promptmrg": gt}
           for sid, pred, gt in zip(ids_in_order, all_res, all_gts)]
print(f"[PromptMRG] generated {len(records)} reports.")

os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)
with open(DUMP_PATH, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"[PromptMRG] dumped -> {DUMP_PATH}")

# ============================================================
# sanity: PromptMRG-protocol NLG on these dumps (their compute_scores).
# This is just a "did the checkpoint load right" smoke test; the NUMBERS THAT GO
# IN THE PAPER come from re-scoring DUMP_PATH['generated'] through YOUR pipeline
# (score_r2gen.py), NOT from here.
# ============================================================
try:
    from modules.metrics import compute_scores
    res_d = {i: [r] for i, r in enumerate(all_res)}
    gts_d = {i: [g] for i, g in enumerate(all_gts)}
    scores = compute_scores(gts_d, res_d)
    print("[PromptMRG] sanity NLG (their protocol):")
    for k, v in scores.items():
        print(f"        {k}: {round(v * 100, 2)}")
except Exception as e:
    print(f"[PromptMRG] sanity NLG skipped ({e}); the dump above is unaffected.")

print("\n[PromptMRG] DONE. Next: point score_r2gen.py's R2GEN_DUMP at this file and run it:\n"
      f"    R2GEN_DUMP = \"{DUMP_PATH}\"\n"
      "  It will align by id against YOUR_DUMP (your 590 ids + references) and re-score\n"
      "  through R2Gen-NLG + CheXbert-5/14 + RadGraph — the real Table-1 row.\n"
      "  CHECK the printed 'aligned N/590': if N < 590, PromptMRG's IU set doesn't fully\n"
      "  cover your R2Gen test split — note the sample count or build a 590-matched\n"
      "  annotation (needs recomputing clip_indices via their MIMIC CLIP).")
