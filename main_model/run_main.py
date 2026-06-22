"""
Main entry point for Resampler-RAG — CheXbert labels + scheduled sampling.
"""
import os
import argparse

from config import DataConfig, ModelConfig, RetrievalConfig, TrainConfig
from data import load_r2gen_data, DualViewRAGDataset, XRayTransform
from train import train, load_checkpoint, set_seed

def get_model(model_cfg, train_cfg):
    from disease_t5 import DiseaseT5
    return DiseaseT5(model_cfg, train_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle hint eval: use GT findings instead of xrv predictions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed; output_dir gets a _seed{N} suffix "
                             "so multi-seed runs don't overwrite each other.")
    parser.add_argument("--no_hint", action="store_true",
                        help="A3: remove disease hints from prompt (tests text hint channel)")
    parser.add_argument("--no_cls_loss", action="store_true",
                        help="A4: remove auxiliary cls loss (tests perceiver training signal)")
    args = parser.parse_args()

    data_cfg  = DataConfig()
    model_cfg = ModelConfig()
    retr_cfg  = RetrievalConfig()
    train_cfg = TrainConfig()

    # ---- Seed: control the FULL pipeline (incl. model init) + per-seed output dir ----
    # set_seed BEFORE building datasets/model so parameter init is seeded too;
    # train() re-seeds before the data loop, so data order is also seed-determined.
    # output_dir gets tags (ablation markers) + _seed{N} suffix so each configuration
    # writes into its own isolated folder.
    # NOTE: the train undersampling RNG stays fixed (seed=42 in data.py) on purpose —
    # the training SUBSET is identical across seeds, so the variance you measure is
    # init+training noise, not data-resampling noise.

    # ---- Ablation flags ----
    tags = ""
    if args.no_hint:
        train_cfg.use_disease_hint = False
        tags += "_noHint"
    if args.no_cls_loss:
        train_cfg.use_cls_loss = False
        tags += "_noCls"

    if args.seed is not None:
        train_cfg.seed = args.seed
    train_cfg.output_dir = f"{train_cfg.output_dir}{tags}_seed{train_cfg.seed}"
    set_seed(train_cfg.seed)

    if args.smoke_test:
        train_cfg.num_epochs = 2
        train_cfg.batch_size = 4
        train_cfg.eval_every = 1

    print(f"\n{'='*60}\n  Output: {train_cfg.output_dir}\n{'='*60}")

    # =========================================================
    # 1. Load data — with CheXbert labels if cache available
    # =========================================================
    train_raw, val_raw, test_raw = load_r2gen_data(data_cfg.annotation_file)

    if args.smoke_test:
        train_raw = train_raw[:100]
        val_raw   = val_raw[:20]
        test_raw  = test_raw[:20]

    xform = XRayTransform()
    chex_dir = retr_cfg.cache_dir  # where chexbert_labels.pt lives
    train_ds = DualViewRAGDataset(train_raw, data_cfg.images_dir, xform, data_cfg.min_report_len,
                                   is_train=True,
                                   abnormal_normal_ratio=data_cfg.abnormal_normal_ratio,
                                   chexbert_cache_dir=chex_dir)
    val_ds   = DualViewRAGDataset(val_raw,   data_cfg.images_dir, xform, data_cfg.min_report_len, is_train=False, chexbert_cache_dir=chex_dir)
    test_ds  = DualViewRAGDataset(test_raw,  data_cfg.images_dir, xform, data_cfg.min_report_len, is_train=False, chexbert_cache_dir=chex_dir)

    # =========================================================
    # 2. Build model
    # =========================================================
    print("\n" + "="*60 + "\nStep 2: Build model\n" + "="*60)
    model = get_model(model_cfg, train_cfg)
    model = model.to(args.device)
    if model.vision_encoder is not None:
        model.vision_encoder = model.vision_encoder.to(args.device)

    if hasattr(model, 'set_disease_class_weights'):
        model.set_disease_class_weights(train_ds)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    # =========================================================
    # 3. Train
    # =========================================================
    if not args.eval_only:
        print("\n" + "="*60 + "\nStep 4: Train\n" + "="*60)
        history = train(model, train_ds, val_ds, train_cfg, args.device)
        best_path = os.path.join(train_cfg.output_dir, "best.pt")
        if os.path.exists(best_path):
            load_checkpoint(model, best_path)

    if hasattr(model, 'diagnose_disease_head'):
        print("\n" + "="*60 + "\nDisease head diagnosis (post-training)\n" + "="*60)
        model.diagnose_disease_head(val_ds, device=args.device)

    if hasattr(model, 'diagnose_xrv_hints'):
        print("\n" + "="*60 + "\nxrv hint diagnosis (val set)\n" + "="*60)
        model.diagnose_xrv_hints(val_ds, device=args.device)

    # =========================================================
    # 4. Final test evaluation
    # =========================================================
    print("\n" + "="*60 + "\nStep 5: Final test evaluation\n" + "="*60)
    from evaluate import evaluate_model
    oracle_tag = "_oracle" if args.oracle else ""
    test_metrics = evaluate_model(
        model, test_ds,
        train_cfg, args.device,
        save_name=f"test_final{oracle_tag}.json",
        cache_dir=retr_cfg.cache_dir,
        oracle_hint=args.oracle,
    )
    print("\n" + "="*60)
    label = "FINAL TEST METRICS [ORACLE]" if args.oracle else "FINAL TEST METRICS"
    print(label)
    print("="*60)

    m = test_metrics

    # ---- R2Gen official protocol (USE THESE for comparison vs published work) ----
    print("\n  --- NLG (R2Gen official protocol) ---")
    for k in ["R2Gen_BLEU_1", "R2Gen_BLEU_2", "R2Gen_BLEU_3", "R2Gen_BLEU_4",
              "R2Gen_METEOR", "R2Gen_ROUGE_L"]:
        if k in m:
            print(f"  {k:28s} {m[k]}")

    # ---- Clinical efficacy (CheXbert) ----
    print("\n  --- Clinical efficacy (CheXbert) ---")
    for k in ["ClinicalF1_chexbert_micro_F_14", "ClinicalF1_chexbert_macro_F_14",
              "ClinicalF1_chexbert_micro_F_5", "ClinicalF1_chexbert_macro_F_5",
              "ClinicalF1_chexbert_accuracy",
              "CheXbert5_Cardiomegaly_F1", "CheXbert5_Edema_F1",
              "CheXbert5_Consolidation_F1", "CheXbert5_Atelectasis_F1",
              "CheXbert5_Pleural Effusion_F1"]:
        if k in m:
            print(f"  {k:28s} {m[k]}")

    # ---- RadGraph ----
    print("\n  --- RadGraph (factual consistency) ---")
    for k in ["RadGraph_Simple", "RadGraph_Partial", "RadGraph_Complete"]:
        if k in m:
            print(f"  {k:28s} {m[k]}")


if __name__ == "__main__":
    main()
