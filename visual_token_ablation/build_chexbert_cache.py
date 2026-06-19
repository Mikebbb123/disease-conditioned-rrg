"""
Build CheXbert label cache for training — replaces regex labels.

Run ONCE before training:
  python build_chexbert_cache.py

Output: {cache_dir}/chexbert_labels.pt
         dict: report_text (cleaned) → [5-class binary multihot]

Uses labeler.get_label() for eval-identical 14-class binary labels,
then maps to our 5-class subset.
"""
import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DataConfig, RetrievalConfig
from data import load_r2gen_data, clean_report_text

data_cfg = DataConfig()
retr_cfg = RetrievalConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
BATCH = 32

# ---- 1. Load data ----
print("Loading data …")
train_raw, val_raw, test_raw = load_r2gen_data(data_cfg.annotation_file)

# Collect all unique reports
all_reports = {}
for item in train_raw + val_raw + test_raw:
    r = clean_report_text(item.get("report", ""))
    if len(r) >= data_cfg.min_report_len:
        all_reports[r] = True
report_list = sorted(all_reports.keys())
print(f"  Unique reports: {len(report_list)}")

# ---- 2. Load CheXbert and discover class mapping ----
print("Loading f1chexbert …")
from chexbert_eval import get_chexbert_labeler, _undo_tokenizer_patch
labeler = get_chexbert_labeler(device=DEVICE)

# f1chexbert's internal CheXbert model outputs 14 classes.
# Discover the mapping by running a test pair and inspecting class_report_14.
dummy = "No acute cardiopulmonary findings."
accuracy, _, cr14, cr5 = labeler(hyps=[dummy], refs=[dummy])
_undo_tokenizer_patch()

# Get the 14 class names from the report
all14 = [k for k in cr14.keys() if k not in ("macro avg", "micro avg", "accuracy")]
print(f"  CheXbert 14 classes: {all14}")

# Map our 5 to the 14
IDX14 = {}
for t5 in TARGET_5:
    if t5 in all14:
        IDX14[t5] = all14.index(t5)
    elif t5 == "Edema" and "Pulmonary Edema" in all14:
        IDX14[t5] = all14.index("Pulmonary Edema")
    elif t5 == "Pleural Effusion" and "Effusion" in all14:
        IDX14[t5] = all14.index("Effusion")
    else:
        raise KeyError(f"Cannot find {t5} in CheXbert 14-class output: {all14}")
print(f"  5-class indices: {IDX14}")
IDX_LIST = [IDX14[t] for t in TARGET_5]

# ---- 3. Label all reports using labeler.get_label() (eval-identical logic) ----
# f1chexbert's get_label() handles tokenization + 4-way→binary internally.
# Same pipeline as evaluation — labels are naturally consistent.
print("Labeling all reports (get_label, eval-identical) …")
labels_5 = []
for r in tqdm(report_list, desc="CheXbert"):
    vec14 = labeler.get_label(r)                    # list of 14 ints (0/1)
    vec14 = torch.as_tensor(vec14, dtype=torch.float32)
    labels_5.append(vec14[IDX_LIST])                # keep our 5 classes
labels_5 = torch.stack(labels_5)                    # [N, 5]
_undo_tokenizer_patch()

# ---- 4. Build cache ----
cache = {}
for r, vec in zip(report_list, labels_5):
    cache[r] = vec

# Stats
n = len(cache)
n_ab = int((labels_5.sum(dim=1) > 0).sum().item())
counts = labels_5.sum(dim=0).long().tolist()
print(f"\nCheXbert labels ({n} reports):")
print(f"  Abnormal (any): {n_ab}/{n} ({100*n_ab/n:.1f}%)")
for i, name in enumerate(TARGET_5):
    print(f"  {name:22s}: {counts[i]:5d} ({100*counts[i]/n:.1f}%)")

# Compare to regex
from data_utils_compat import report_to_multihot
regex_vecs = torch.stack([report_to_multihot(r) for r in report_list])
regex_counts = regex_vecs.sum(dim=0).long().tolist()
n_regex_ab = int((regex_vecs.sum(dim=1) > 0).sum().item())
print(f"\nRegex labels (same {n} reports, for comparison):")
print(f"  Abnormal (any): {n_regex_ab}/{n} ({100*n_regex_ab/n:.1f}%)")
for i, name in enumerate(TARGET_5):
    print(f"  {name:22s}: {regex_counts[i]:5d} ({100*regex_counts[i]/n:.1f}%)")

# ---- 5. Save ----
cache_path = os.path.join(retr_cfg.cache_dir, "chexbert_labels.pt")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
torch.save(cache, cache_path)
print(f"\nSaved to {cache_path}")
print("Done.")
