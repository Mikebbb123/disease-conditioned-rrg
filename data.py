"""
Data loading for IU-Xray (R2Gen split).
Uses CheXbert labels for training when cache is available (aligns with eval metric).
"""
import os
import re
import json
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from data_utils_compat import report_to_multihot


# ============================================================
# CheXbert label cache — loaded once at import time
# ============================================================
_chexbert_cache = None
_CHEXBERT_CACHE_PATH = None


def _load_chexbert_cache(cache_dir: str = None) -> dict:
    """Load the CheXbert label cache if available.  Returns None if not found."""
    global _chexbert_cache, _CHEXBERT_CACHE_PATH
    if cache_dir is None:
        return None
    path = os.path.join(cache_dir, "chexbert_labels.pt")
    if _chexbert_cache is not None and _CHEXBERT_CACHE_PATH == path:
        return _chexbert_cache
    if os.path.exists(path):
        _chexbert_cache = torch.load(path, map_location="cpu")
        _CHEXBERT_CACHE_PATH = path
        print(f"[Data] Loaded CheXbert label cache: {len(_chexbert_cache)} reports from {path}")
        return _chexbert_cache
    return None


# ============================================================
# Data augmentation transforms
# ============================================================

train_augmentations = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((512, 512)),
])


# ============================================================
# Loading + cleaning
# ============================================================

def load_r2gen_data(annotation_file: str) -> Tuple[list, list, list]:
    print(f"[Data] Loading annotation: {annotation_file}")
    with open(annotation_file) as f:
        data = json.load(f)
    train, val, test = data["train"], data["val"], data["test"]
    print(f"[Data] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def clean_report_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\bxxxx\b', '', text)
    for old, new in {"w/": "with", "w/o": "without", "b/l": "bilateral"}.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = text.strip()
    if text and not text.endswith('.'):
        text += '.'
    return text


# ============================================================
# X-ray transform (torchxrayvision)
# ============================================================

class XRayTransform:
    def __init__(self, size: int = 512):
        import torchxrayvision as xrv
        self.normalize = xrv.datasets.normalize
        self.crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(size)

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        img = pil_image.convert("L")
        img = np.array(img, dtype=np.float32)
        img = self.normalize(img, 255)
        img = img[None, :, :]
        img = self.crop(img)
        img = self.resize(img)
        return torch.from_numpy(img).float()


# ============================================================
# Train transform wrapper
# ============================================================

class TrainTransform:
    def __init__(self):
        self.xray_xform = XRayTransform()

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        img = pil_image.convert("RGB")
        img = train_augmentations(img)
        img = img.convert("L")
        return self.xray_xform(img)


# ============================================================
# Undersampling
# ============================================================

def _deduplicate_normal_reports(samples: list) -> list:
    seen = set()
    deduped = []
    n_dup = 0
    for s in samples:
        is_normal = s["disease_labels"].sum().item() == 0
        if is_normal:
            key = re.sub(r'\s+', ' ', s["report"].strip().lower())
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
        deduped.append(s)
    if n_dup > 0:
        print(f"[Dedup] Removed {n_dup} duplicate normal reports "
              f"({len(samples)} -> {len(deduped)})")
    return deduped


def _undersample_normal(samples: list, target_ratio: float = 1.5,
                        seed: int = 42) -> list:
    rng = random.Random(seed)
    normal_idx = [i for i, s in enumerate(samples)
                  if s["disease_labels"].sum().item() == 0]
    abnormal_idx = [i for i, s in enumerate(samples)
                    if s["disease_labels"].sum().item() > 0]
    n_ab = len(abnormal_idx)
    n_norm = len(normal_idx)
    if n_ab == 0 or n_norm == 0:
        return samples
    target_norm = max(1, int(n_ab / target_ratio))
    if n_norm <= target_norm:
        return samples
    kept_norm = set(rng.sample(normal_idx, target_norm))
    result = [s for i, s in enumerate(samples)
              if i not in normal_idx or i in kept_norm]
    rng.shuffle(result)
    return result


# ============================================================
# Dual-view dataset
# ============================================================

class DualViewRAGDataset(Dataset):
    def __init__(
        self,
        raw_data: list,
        images_dir: str,
        transform: Optional[object] = None,
        min_report_len: int = 15,
        is_train: bool = False,
        undersample_normal: bool = True,
        abnormal_normal_ratio: float = 1.5,
        chexbert_cache_dir: str = None,
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.is_train = is_train
        self.train_transform = TrainTransform() if is_train else None
        self.samples = []

        # Load CheXbert label cache (one-time, shared across datasets)
        cache = _load_chexbert_cache(chexbert_cache_dir)
        n_chex, n_regex = 0, 0

        skipped = {"short_report": 0, "wrong_image_count": 0, "missing_file": 0}
        for item in raw_data:
            report = clean_report_text(item.get("report", ""))
            image_paths = item.get("image_path", [])

            if len(report) < min_report_len:
                skipped["short_report"] += 1
                continue
            if len(image_paths) < 2:
                skipped["wrong_image_count"] += 1
                continue

            frontal = os.path.join(images_dir, image_paths[0])
            lateral = os.path.join(images_dir, image_paths[1])
            if not os.path.exists(frontal) or not os.path.exists(lateral):
                skipped["missing_file"] += 1
                continue

            # Use CheXbert labels if cache available; count hits vs fallback.
            if cache is not None:
                if report in cache:
                    disease_label_tensor = cache[report].clone()
                    n_chex += 1
                else:
                    disease_label_tensor = report_to_multihot(report)
                    n_regex += 1
            else:
                disease_label_tensor = report_to_multihot(report)
                n_regex += 1

            self.samples.append({
                "id": item.get("id", ""),
                "frontal_path": frontal,
                "lateral_path": lateral,
                "report": report,
                "disease_labels": disease_label_tensor,
            })

        # Report label source coverage
        if cache is not None:
            n_total = n_chex + n_regex
            print(f"[DualViewRAGDataset] Label source: {n_chex}/{n_total} CheXbert, "
                  f"{n_regex}/{n_total} regex fallback")
            if n_regex > 0:
                print(f"[DualViewRAGDataset] ERROR: {n_regex} reports NOT found in "
                      f"CheXbert cache — labels are MIXED (CheXbert + regex). "
                      f"This corrupts training. Rebuild the cache with "
                      f"build_chexbert_cache.py and fix the key mismatch.")
                raise RuntimeError(
                    f"CheXbert cache miss: {n_regex} reports not in cache. "
                    f"Rebuild: python build_chexbert_cache.py")

        # Undersample (uses disease_labels, works with either regex or CheXbert)
        if is_train and undersample_normal:
            self.samples = _deduplicate_normal_reports(self.samples)
            self.samples = _undersample_normal(self.samples, abnormal_normal_ratio)

        all_labels = torch.stack([s["disease_labels"] for s in self.samples])
        n_abnormal = (all_labels.sum(dim=1) > 0).sum().item()
        per_disease = all_labels.sum(dim=0).long().tolist()
        mode = "train" if is_train else "eval"
        label_src = "CheXbert" if (cache is not None and n_chex > 0 and n_regex == 0) else \
                    ("mixed" if n_regex > 0 and n_chex > 0 else "regex")
        print(f"[DualViewRAGDataset:{mode}] Loaded {len(self.samples)} samples ({label_src} labels); skipped: {skipped}")
        print(f"[DualViewRAGDataset:{mode}]   Abnormal samples: "
              f"{n_abnormal}/{len(self.samples)} ({100*n_abnormal/max(len(self.samples),1):.1f}%)")
        print(f"[DualViewRAGDataset:{mode}]   Per-disease positive counts: {per_disease}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frontal_pil = Image.open(sample["frontal_path"])
        lateral_pil = Image.open(sample["lateral_path"])

        if self.is_train and self.train_transform is not None:
            frontal = self.train_transform(frontal_pil)
            lateral = self.train_transform(lateral_pil)
        elif self.transform is not None:
            frontal = self.transform(frontal_pil)
            lateral = self.transform(lateral_pil)
        else:
            t = XRayTransform()
            frontal = t(frontal_pil)
            lateral = t(lateral_pil)

        return {
            "idx": idx,
            "id": sample["id"],
            "frontal": frontal,
            "lateral": lateral,
            "report": sample["report"],
            "disease_labels": sample["disease_labels"],
        }

def rag_collate_fn(batch):
    return {
        "idx":            torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "frontal":        torch.stack([b["frontal"] for b in batch]),
        "lateral":        torch.stack([b["lateral"] for b in batch]),
        "reports":        [b["report"] for b in batch],
        "ids":            [b["id"] for b in batch],
        "disease_labels": torch.stack([b["disease_labels"] for b in batch]),
    }
