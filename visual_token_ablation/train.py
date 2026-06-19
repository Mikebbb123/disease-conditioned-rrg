"""
Training loop — uniform sampling, online visual retrieval, macro-F1 early stopping.
"""
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from data import rag_collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**31
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model, train_dataset, val_dataset, train_config, device="cuda"):
    set_seed(train_config.seed)
    g = torch.Generator()
    g.manual_seed(train_config.seed)

    if train_config.use_bf16 and not torch.cuda.is_bf16_supported():
        print("[Train] bf16 not supported; falling back to fp32.")
        train_config.use_bf16 = False
    os.makedirs(train_config.output_dir, exist_ok=True)

    from chexbert_eval import _undo_tokenizer_patch
    _undo_tokenizer_patch()

    # Uniform sampling — no WeightedRandomSampler (it was a non-lever).
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=rag_collate_fn,
        num_workers=train_config.num_workers,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    vision_params = [p for p in model.parameters() if p.requires_grad and
                     any(p is param for _, param in model.vision_encoder.named_parameters())]
    t5_params = [p for p in model.parameters() if p.requires_grad and p not in set(vision_params)]

    use_lora = getattr(model.model_config, "USE_LORA", False)
    t5_lr = train_config.lora_lr if use_lora else train_config.learning_rate

    optimizer = AdamW([
        {"params": t5_params, "lr": t5_lr},
        {"params": vision_params, "lr": train_config.vision_lr},
    ], weight_decay=train_config.weight_decay)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable params: {n_trainable:,} "
          f"(T5={'LoRA' if use_lora else 'full'}={sum(p.numel() for p in t5_params):,}, "
          f"vision_BN={sum(p.numel() for p in vision_params):,}, "
          f"T5_lr={t5_lr})")

    steps_per_epoch = max(1, len(train_loader) // train_config.grad_accum)
    total_steps = steps_per_epoch * train_config.num_epochs
    warmup_steps = int(total_steps * train_config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Train] total steps: {total_steps} (warmup {warmup_steps})")

    model = model.to(device)

    best_score = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    trainable_params = [p for grp in optimizer.param_groups for p in grp["params"]]

    for epoch in range(train_config.num_epochs):
        model.train()
        model.vision_encoder.eval()

        running = {"loss": 0.0, "entropy": 0.0, "cls_loss": 0.0, "hint_rate": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config.num_epochs}")

        for step, batch in enumerate(pbar):
            frontal = batch["frontal"].to(device)
            lateral = batch["lateral"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=train_config.use_bf16):
                outputs = model(
                    frontal, lateral,
                    batch["reports"], training=True,
                    disease_labels=batch["disease_labels"],
                    current_epoch=epoch,
                )
                loss = outputs.gen_loss / train_config.grad_accum

            loss.backward()

            if (step + 1) % train_config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, train_config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running["loss"] += outputs.gen_loss.item()
            running["entropy"] += outputs.entropy.item()
            running["cls_loss"] += outputs.cls_loss.item()
            running["hint_rate"] += outputs.hint_positive_rate.item()
            n = step + 1
            lr = scheduler.get_last_lr()[0]

            pbar.set_postfix({
                "loss": f"{running['loss']/n:.3f}",
                "cls":  f"{running['cls_loss']/n:.4f}",
                "ent":  f"{running['entropy']/n:.3f}",
                "hint": f"{running['hint_rate']/n:.2%}",
                "lr":   f"{lr:.1e}",
            })

        print(f"\n[Train] Epoch {epoch+1}: loss={running['loss']/n:.3f}, "
              f"cls_loss={running['cls_loss']/n:.4f}, "
              f"entropy={running['entropy']/n:.4f}, "
              f"hint_rate={running['hint_rate']/n:.1%}, "
              f"gate={torch.tanh(model.visual_gate).item():.4f}")

        if (epoch + 1) % train_config.eval_every == 0:
            from evaluate import evaluate_model
            print(f"\n[Eval] Validation after epoch {epoch+1}...")
            val_metrics = evaluate_model(
                model, val_dataset,
                train_config, device,
                save_name=f"val_epoch{epoch+1}.json",
            )
            print(f"[Eval] Epoch {epoch+1} val: {val_metrics}")
            entry = {
                "epoch": epoch + 1,
                "train_loss":    running["loss"] / n,
                "train_cls_loss": running["cls_loss"] / n,
                "train_entropy": running["entropy"] / n,
                **val_metrics,
            }
            history.append(entry)

            # Early stopping on composite score: micro-F1(5) + BLEU.
            # Micro-F1(5) is more stable than macro (rare classes dominate macro noise).
            # BLEU term naturally penalises low-fluency epochs; floor blocks degenerate output.
            micro5 = val_metrics.get("ClinicalF1_chexbert_micro_F_5", 0.0)
            macro5 = val_metrics.get("ClinicalF1_chexbert_macro_F", 0.0)
            bleu = val_metrics.get("R2Gen_BLEU_4")
            if bleu is None:
                bleu = val_metrics.get("BLEU-4", 0.0)
            current_score = (micro5 + bleu) if bleu >= 8.0 else 0.0

            if best_score is None or current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_checkpoint(model, train_config.output_dir, "best.pt", epoch+1, val_metrics)
                print(f"[Save] New best = {current_score:.2f} "
                      f"(micro5={micro5:.2f}, macro5={macro5:.2f}, BLEU-4={bleu:.2f}) (epoch {epoch+1})")
            else:
                epochs_without_improvement += 1
                print(f"[EarlyStop] no improvement "
                      f"({current_score:.2f} <= {best_score:.2f}, "
                      f"micro5={micro5:.2f}, BLEU-4={bleu:.2f}). "
                      f"Patience: {epochs_without_improvement}/{train_config.early_stop_patience}")

            if epochs_without_improvement >= train_config.early_stop_patience:
                print(f"[EarlyStop] Stopping at epoch {epoch+1} "
                      f"(best macro-F1={best_score:.2f} at epoch {best_epoch})")
                break

        if (epoch + 1) % train_config.save_every == 0:
            save_checkpoint(model, train_config.output_dir, f"epoch{epoch+1}.pt", epoch+1, None)

    with open(os.path.join(train_config.output_dir, "history.json"), "w") as f:
        clean = []
        for e in history:
            clean.append({k: (float(v) if hasattr(v, "item") else v) for k, v in e.items()})
        json.dump(clean, f, indent=2)

    return history


def save_checkpoint(model, output_dir: str, name: str, epoch: int, metrics):
    path = os.path.join(output_dir, name)
    torch.save({
        "epoch": epoch,
        "trainable_state": model.trainable_state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(ckpt["trainable_state"])
    print(f"[Load] Loaded checkpoint from {path} (epoch={ckpt.get('epoch')})")
    return ckpt
