"""
Gradient connectivity check: verify cls_loss gradients flow into perceiver.

MUST run BEFORE starting training after the disease_head rewire.
Success criterion: perceiver_grad > 0 after cls_loss-only backward.

Usage: python check_cls_grad.py [--device cuda]
"""
import argparse
import torch
import torch.nn.functional as F

from config import DataConfig, ModelConfig, TrainConfig
from data import load_r2gen_data, DualViewRAGDataset, XRayTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device

    print("=" * 60)
    print("Gradient Connectivity Check: cls_loss → perceiver")
    print("=" * 60)

    # ---- 1. Load a tiny slice of data ----
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    train_raw, _, _ = load_r2gen_data(data_cfg.annotation_file)
    xform = XRayTransform()
    ds = DualViewRAGDataset(
        train_raw, data_cfg.images_dir, xform, data_cfg.min_report_len,
        is_train=False, undersample_normal=False,
    )

    # Take just 4 samples for a quick check
    from torch.utils.data import DataLoader, Subset
    from data import rag_collate_fn
    subset = Subset(ds, range(min(4, len(ds))))
    loader = DataLoader(subset, batch_size=4, collate_fn=rag_collate_fn)
    batch = next(iter(loader))

    # ---- 2. Build model ----
    from disease_t5 import DiseaseT5
    model = DiseaseT5(model_cfg, train_cfg)
    model = model.to(device)
    model.vision_encoder = model.vision_encoder.to(device)
    model.set_disease_class_weights(ds)
    model.train()

    # ---- 3. Forward + cls_loss-only backward ----
    frontal = batch["frontal"].to(device)
    lateral = batch["lateral"].to(device)
    d_labels = batch["disease_labels"].to(device)

    spatial = model.encode_visual(frontal, lateral)
    visual_tokens = model._get_visual_tokens(spatial, frontal, lateral)

    pooled = visual_tokens.mean(dim=1)        # [B, 768]
    logits = model.disease_head(pooled)

    cls_loss = F.binary_cross_entropy_with_logits(
        logits, d_labels.float(),
        pos_weight=model._pos_weight.to(device) if model._pos_weight is not None else None,
    )
    print(f"\n[Check] cls_loss = {cls_loss.item():.4f}")

    model.zero_grad()
    cls_loss.backward()

    # ---- 4. Check perceiver gradients ----
    perceiver_grad = 0.0
    perceiver_params = 0
    zero_grad_params = []
    for n, p in model.named_parameters():
        if "perceiver" in n:
            perceiver_params += 1
            if p.grad is not None:
                g = p.grad.abs().sum().item()
                perceiver_grad += g
                if g == 0:
                    zero_grad_params.append(n)
            else:
                zero_grad_params.append(f"{n} (grad is None)")

    # Also check disease_head gradients
    head_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "disease_head" in n and p.grad is not None
    )

    # Check that NO vision_encoder params got gradients
    vision_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "vision_encoder" in n and p.grad is not None
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  perceiver params:     {perceiver_params}")
    print(f"  perceiver grad sum:   {perceiver_grad:.6f}")
    print(f"  disease_head grad sum: {head_grad:.6f}")
    print(f"  vision_encoder grad:   {vision_grad:.6f} (should be ~0)")
    if zero_grad_params:
        print(f"  ⚠️  ZERO-GRAD params: {zero_grad_params}")

    print()
    if perceiver_grad > 0:
        print("✅ PASSED — cls_loss gradients flow into perceiver.")
        print("   The rewire is working. Safe to start training.")
    else:
        print("❌ FAILED — perceiver received ZERO gradient from cls_loss.")
        print("   visual_tokens is likely detached or recomputed.")
        print("   Check: 1) _get_visual_tokens has NO .detach()")
        print("         2) visual_tokens is reused, not recomputed")
        print("         3) cls_loss is computed INSIDE torch.no_grad()")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
