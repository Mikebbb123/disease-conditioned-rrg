"""
Decoding parameter sweep — find optimal length_penalty, min_length, num_beams.
Runs on val set, reports BLEU + length ratio + CheXbert F1.

Usage:
  python sweep_decode.py --checkpoint best.pt [--device cuda] [--max_trials 20]
"""
import os, argparse, itertools, json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import DataConfig, ModelConfig, RetrievalConfig, TrainConfig
from data import load_r2gen_data, DualViewRAGDataset, XRayTransform, rag_collate_fn
from retrieval import (load_vision_encoder, build_retrieval_index,
                       extract_features, aggregate_dual_view, retrieve_top_k)
from disease_t5 import DiseaseT5


def eval_one_config(model, dataset, index_features, index_reports, aggregation,
                    train_config, device, num_beams, length_penalty, min_length,
                    no_repeat_ngram_size):
    """Run generation + compute BLEU + length ratio. Fast, no CheXbert."""
    model.eval()
    loader = DataLoader(dataset, batch_size=train_config.eval_batch_size,
                        shuffle=False, collate_fn=rag_collate_fn,
                        num_workers=train_config.num_workers)
    index_features = index_features.to(device)

    preds, refs = [], []
    total_pred_len, total_ref_len = 0, 0

    for batch in tqdm(loader, desc=f"  b={num_beams} lp={length_penalty} ml={min_length}", leave=False):
        frontal = batch["frontal"].to(device)
        lateral = batch["lateral"].to(device)
        sample_indices = batch["idx"].tolist()

        f_feats = extract_features(model.vision_encoder, frontal)
        l_feats = extract_features(model.vision_encoder, lateral)
        query = F.normalize(aggregate_dual_view(f_feats, l_feats, aggregation), dim=1)
        retrieved_lists = retrieve_top_k(
            query, index_features, index_reports,
            k=getattr(train_config, "retrieval_k", 1),
            exclude_indices=sample_indices,
        )

        gen = model.generate(
            frontal, lateral, retrieved_lists,
            num_beams=num_beams,
            max_length=train_config.max_target_length,
            length_penalty=length_penalty,
            min_length=min_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        for p, r in zip(gen, batch["reports"]):
            preds.append(p)
            refs.append(r)
            total_pred_len += len(p.split())
            total_ref_len += len(r.split())

    # BLEU
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [p] for i, p in enumerate(preds)}
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        bleu4 = round(bleu_scores[3] * 100, 2)
    except ImportError:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sm = SmoothingFunction().method1
        refs_tok = [[r.split()] for r in refs]
        pred_tok = [p.split() for p in preds]
        bleu4 = round(corpus_bleu(refs_tok, pred_tok,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=sm) * 100, 2)

    len_ratio = total_pred_len / max(total_ref_len, 1)
    return bleu4, len_ratio, preds, refs


def compute_f1_fast(preds, refs, labeler, device):
    """CheXbert F1 on a subset (first 200 samples) for speed."""
    from chexbert_eval import compute_clinical_f1_chexbert
    n = min(200, len(preds))
    m = compute_clinical_f1_chexbert(preds[:n], refs[:n], labeler)
    return m.get("ClinicalF1_chexbert_macro_F", 0.0), m.get("ClinicalF1_chexbert_micro_F_5", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_trials", type=int, default=30,
                    help="max param combinations (randomly sampled)")
    ap.add_argument("--full_f1", action="store_true",
                    help="run CheXbert F1 on every trial (slow)")
    args = ap.parse_args()

    device = args.device
    data_cfg  = DataConfig()
    model_cfg = ModelConfig()
    retr_cfg  = RetrievalConfig()
    train_cfg = TrainConfig()

    # ---- Load data ----
    train_raw, val_raw, _ = load_r2gen_data(data_cfg.annotation_file)
    xform = XRayTransform()
    chex_dir = retr_cfg.cache_dir
    val_ds = DualViewRAGDataset(val_raw, data_cfg.images_dir, xform,
                                 data_cfg.min_report_len, is_train=False,
                                 chexbert_cache_dir=chex_dir)

    # ---- Build retrieval index (train set) ----
    vision = load_vision_encoder(model_cfg.vision_weights,
                                  getattr(model_cfg, "vision_model", "resnet"))
    cache_path = os.path.join(retr_cfg.cache_dir, "train_index_concat.pt")
    index_ds = DualViewRAGDataset(train_raw, data_cfg.images_dir, xform,
                                   data_cfg.min_report_len, is_train=False,
                                   chexbert_cache_dir=chex_dir)
    train_index = build_retrieval_index(index_ds, vision, batch_size=32,
                                         device=device, aggregation="concat",
                                         cache_path=cache_path,
                                         num_workers=train_cfg.num_workers)
    del vision
    torch.cuda.empty_cache()

    # ---- Load model ----
    model = DiseaseT5(model_cfg, train_cfg).to(device)
    model.vision_encoder = model.vision_encoder.to(device)
    from train import load_checkpoint
    load_checkpoint(model, args.checkpoint)

    index_features = train_index["features"]
    index_reports = train_index["reports"]
    aggregation = train_index["aggregation"]

    # ---- CheXbert labeler (if running full F1) ----
    labeler = None
    if args.full_f1:
        from chexbert_eval import get_chexbert_labeler
        labeler = get_chexbert_labeler(device=device)

    # ---- Parameter grid ----
    param_grid = {
        "num_beams":             [4, 6, 8, 10],
        "length_penalty":        [1.0, 1.5, 2.0, 2.5, 3.0],
        "min_length":            [40, 45, 50, 55, 60],
        "no_repeat_ngram_size":  [2, 3, 4],
    }

    # Generate all combinations, subsample if too many
    all_combos = list(itertools.product(
        param_grid["num_beams"],
        param_grid["length_penalty"],
        param_grid["min_length"],
        param_grid["no_repeat_ngram_size"],
    ))
    import random
    random.seed(42)
    if len(all_combos) > args.max_trials:
        combos = random.sample(all_combos, args.max_trials)
    else:
        combos = all_combos

    # Always include baseline
    baseline = (8, 1.5, 40, 3)
    if baseline not in combos:
        combos.insert(0, baseline)

    print(f"\n{'='*70}")
    print(f"Sweeping {len(combos)} configs on val set")
    print(f"Baseline: num_beams=8, lp=1.5, min_len=40, ngram=3")
    print(f"{'='*70}\n")

    results = []
    for nb, lp, ml, ng in combos:
        tag = f"b={nb} lp={lp} ml={ml} ng={ng}"
        bleu4, len_ratio, preds, refs = eval_one_config(
            model, val_ds, index_features, index_reports, aggregation,
            train_cfg, device, num_beams=nb, length_penalty=lp,
            min_length=ml, no_repeat_ngram_size=ng,
        )

        f1_macro, f1_micro = 0.0, 0.0
        if args.full_f1 and labeler is not None:
            f1_macro, f1_micro = compute_f1_fast(preds, refs, labeler, device)

        results.append({
            "num_beams": nb, "length_penalty": lp, "min_length": ml,
            "no_repeat_ngram": ng,
            "BLEU-4": bleu4, "len_ratio": round(len_ratio, 3),
            "macro_F": round(f1_macro, 2), "micro_F": round(f1_micro, 2),
        })

        print(f"  {tag:40s}  BLEU={bleu4:5.1f}  len_ratio={len_ratio:.3f}"
              + (f"  macro_F={f1_macro:.1f}  micro_F={f1_micro:.1f}" if args.full_f1 else ""))

    # ---- Sort and print best ----
    print(f"\n{'='*70}")
    print("Top 10 by BLEU-4:")
    print(f"{'='*70}")
    sorted_by_bleu = sorted(results, key=lambda r: -r["BLEU-4"])
    print(f"{'#':>3s} {'b':>2s} {'lp':>4s} {'ml':>3s} {'ng':>2s} {'BLEU-4':>7s} {'len_r':>6s} {'mF1':>6s} {'uF1':>6s}")
    print("-" * 50)
    for i, r in enumerate(sorted_by_bleu[:15]):
        print(f"{i+1:3d} {r['num_beams']:2d} {r['length_penalty']:4.1f} "
              f"{r['min_length']:3d} {r['no_repeat_ngram']:2d} "
              f"{r['BLEU-4']:7.2f} {r['len_ratio']:6.3f} "
              f"{r['macro_F']:6.1f} {r['micro_F']:6.1f}")

    # Mark baseline
    bl = [r for r in results if r["num_beams"]==8 and r["length_penalty"]==1.5
          and r["min_length"]==40 and r["no_repeat_ngram"]==3]
    if bl:
        b = bl[0]
        print(f"\n  BASELINE (b=8 lp=1.5 ml=40 ng=3): BLEU={b['BLEU-4']}, "
              f"len_ratio={b['len_ratio']}, macro_F={b['macro_F']}, micro_F={b['micro_F']}")

    # Also show best by len_ratio (closest to 1.0)
    print(f"\n{'='*70}")
    print("Closest to len_ratio=1.0:")
    print(f"{'='*70}")
    sorted_by_len = sorted(results, key=lambda r: -abs(r["len_ratio"] - 1.0), reverse=True)
    for i, r in enumerate(sorted_by_len[:5]):
        print(f"  {r['num_beams']:2d} {r['length_penalty']:4.1f} {r['min_length']:3d} "
              f"{r['no_repeat_ngram']:2d}  BLEU={r['BLEU-4']:.2f}  "
              f"len_ratio={r['len_ratio']:.3f}  macro_F={r['macro_F']:.1f}")

    # Save full results
    out_path = os.path.join(os.path.dirname(args.checkpoint), "decode_sweep.json")
    with open(out_path, "w") as f:
        json.dump(sorted_by_bleu, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
