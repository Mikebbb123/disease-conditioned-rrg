"""
r2gen_test_dump.py  (matched to the main_train/main_test fork)
=============================================================
Load the OFFICIAL R2Gen IU-Xray checkpoint, run inference on the test split,
and dump generated reports (+ ids) to JSON so they can be scored through YOUR
own CheXbert-5 / RadGraph pipeline for a same-protocol Table-1 comparison.

NO training. Just test + dump.  In Colab:
    python r2gen_test_dump.py
"""

import os
import sys
import json
import functools

# ============================================================
# CONFIG  — edit these
# ============================================================
R2GEN_REPO = os.environ.get("R2GEN_REPO", "./R2Gen")           # the cloned fork (has main_test.py)
IMAGE_DIR  = os.environ.get("IU_XRAY_IMAGES", "./data/iu_xray/images")   # IU images dir
ANN_PATH   = "./data/iu_xray/annotation.json"     # MUST be the annotation the
                                                               # checkpoint was trained on
CKPT       = "./ckpts/model_iu_xray.pth"     # <-- your downloaded official ckpt
DUMP_PATH  = "./dumps/r2gen_test_generated.json"

BEAM_SIZE  = 3        # R2Gen IU default — do NOT change (matches published setup)
MAX_LEN    = 60

# ============================================================
# make the fork importable
# ============================================================
assert os.path.isdir(R2GEN_REPO) and os.path.exists(os.path.join(R2GEN_REPO, "main_test.py")), \
    f"没在 {R2GEN_REPO} 找到 main_test.py,先确认 R2GEN_REPO 路径对、仓库已 clone"
if R2GEN_REPO not in sys.path:
    sys.path.insert(0, R2GEN_REPO)
os.chdir(R2GEN_REPO)

import torch
import numpy as np

# ---- build args via the fork's OWN parser so every default matches the ckpt ----
sys.argv = [
    "main_test.py",
    "--image_dir", IMAGE_DIR,
    "--ann_path", ANN_PATH,
    "--dataset_name", "iu_xray",
    "--max_seq_length", str(MAX_LEN),
    "--threshold", "3",
    "--batch_size", "16",
    "--beam_size", str(BEAM_SIZE),
    "--load", CKPT,
]
from main_test import parse_agrs          # noqa: E402  (this fork's entry file)
args = parse_agrs()

# we load a full checkpoint, so ImageNet init is irrelevant — skip the download
# AND sidestep the new-torchvision `pretrained=` API removal.
args.visual_extractor_pretrained = False

# ---- torchvision compat shim: let R2Gen's `pretrained=` kwarg work on new tv ----
import torchvision.models as _tvm


def _compat(fn):
    @functools.wraps(fn)
    def wrapper(*a, pretrained=None, **kw):
        if pretrained is not None and "weights" not in kw:
            kw["weights"] = "IMAGENET1K_V1" if pretrained else None
        return fn(*a, **kw)
    return wrapper


for _name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "densenet121"]:
    if hasattr(_tvm, _name):
        setattr(_tvm, _name, _compat(getattr(_tvm, _name)))

from modules.tokenizers import Tokenizer        # noqa: E402
from modules.dataloaders import R2DataLoader     # noqa: E402
from models.r2gen import R2GenModel              # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# build tokenizer / dataloader / model
# ============================================================
print("[R2Gen] building tokenizer + test loader …")
tokenizer = Tokenizer(args)
test_loader = R2DataLoader(args, tokenizer, split="test", shuffle=False)

print("[R2Gen] building model …")
model = R2GenModel(args, tokenizer).to(device)

# ---- load official checkpoint (same key as tester._load_checkpoint) ----
print(f"[R2Gen] loading checkpoint: {CKPT}")
try:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
except TypeError:                      # older torch w/o weights_only kwarg
    ckpt = torch.load(CKPT, map_location=device)
state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[R2Gen] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
      f"(both should be ~0; a size mismatch on embeddings => wrong annotation.json/vocab).")
model.eval()

# ============================================================
# inference + dump  (id-aligned JSON)
# ============================================================
print("[R2Gen] generating on test split …")
records, all_res, all_gts = [], [], []
with torch.no_grad():
    for images_id, images, reports_ids, reports_masks in test_loader:
        images = images.to(device)
        output = model(images, mode="sample")
        reports = model.tokenizer.decode_batch(output.cpu().numpy())
        gts = model.tokenizer.decode_batch(reports_ids[:, 1:].numpy())
        for sid, pred, gt in zip(images_id, reports, gts):
            records.append({"id": sid, "generated": pred, "ground_truth_r2gen": gt})
        all_res.extend(reports)
        all_gts.extend(gts)

print(f"[R2Gen] generated {len(records)} reports.")

os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)
with open(DUMP_PATH, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"[R2Gen] dumped -> {DUMP_PATH}")

# ============================================================
# sanity: R2Gen-protocol NLG on these dumps (should land near published ~16.5)
# (needs Java for METEOR; skipped gracefully if unavailable)
# ============================================================
try:
    from modules.metrics import compute_scores
    res_d = {i: [r] for i, r in enumerate(all_res)}
    gts_d = {i: [g] for i, g in enumerate(all_gts)}
    scores = compute_scores(gts_d, res_d)
    print("[R2Gen] sanity NLG (R2Gen protocol, x100):")
    for k, v in scores.items():
        print(f"        {k}: {round(v * 100, 2)}")
    print("        ^ BLEU_4 should be ~16-16.5 if the checkpoint loaded correctly.")
except Exception as e:
    print(f"[R2Gen] sanity NLG skipped ({e}); the dump above is unaffected.")

print("\n[R2Gen] DONE. Next: feed DUMP_PATH['generated'] into YOUR "
      "compute_r2gen_metrics + CheXbert + RadGraph pipeline, aligned by id.")
