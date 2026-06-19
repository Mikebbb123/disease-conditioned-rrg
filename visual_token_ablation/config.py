"""
Clean config for Resampler-RAG: Perceiver visual resampler + SciFive T5.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    annotation_file: str = "/content/drive/MyDrive/iu_xray/annotation.json"
    images_dir: str = "/content/drive/MyDrive/iu_xray/images"
    min_report_len: int = 15
    abnormal_normal_ratio: float = 1.0    # mild: ~50% abnormal (batch hint_rate target ~50%)


@dataclass
class ModelConfig:
    model_name: str = "razent/SciFive-base-Pubmed_PMC"
    tokenizer_name: str = "razent/SciFive-base-Pubmed_PMC"

    vision_model: str = "resnet"
    vision_weights: str = "resnet50-res512-all"
    visual_dim: int = 2048

    USE_LORA: bool = True

    # LoRA hyperparams
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    t5_hidden: int = 768
    dropout_rate: float = 0.3
    num_diseases: int = 5
    # [Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion]
    xrv_thresholds: tuple = (0.044, 0.052, 0.104, 0.136, 0.190)

    # Ensemble (xrv + disease_head) thresholds — tuned on val per-class F1
    # cardio 0.46 / edema 0.60* / consol 0.60* / atel 0.43 / eff 0.26
    # * frozen: edema val n=2, consol val n=1 — sweep values are noise
    ensemble_thresholds: tuple = (0.46, 0.60, 0.60, 0.43, 0.26)

    # Perceiver visual resampler
    num_query_tokens: int = 32
    perceiver_heads: int = 8
    perceiver_layers: int = 2

    # Visual aligner
    visual_aligner_layers: int = 2


@dataclass
class RetrievalConfig:
    top_k: int = 3
    aggregation: str = "concat"
    cache_dir: str = "/content/drive/MyDrive/iu_xray_rag/cache"


@dataclass
class TrainConfig:
    output_dir: str = "/content/drive/MyDrive/iu_xray_rag/output_visual_token"

    batch_size: int = 16
    eval_batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 5e-5
    lora_lr: float = 5e-5
    weight_decay: float = 0.2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    num_epochs: int = 50

    max_input_length: int = 512
    max_target_length: int = 200

    label_smoothing: float = 0.1

    unfreeze_vision_bn: bool = False
    vision_lr: float = 1e-5

    disease_loss_weight: float = 0.3     # Reduced from 1.0 now that cls_loss flows into perceiver

    finding_token_weight: float = 5.0   # weight for tokens in finding sentences (sweep 3-8)

    # ---- Hint scheduled sampling: 训练喂 GT findings, 逐步混入 xrv 预测 ----
    hint_gt_prob_start: float = 1.0    # epoch 0: 100% GT
    hint_gt_prob_end:   float = 0.5    # 末期: 50% GT / 50% xrv 预测
    hint_gt_decay_epochs: int = 20     # 线性衰减步数
    hint_flip_prob: float = 0.0        # 可选: 对 hint 加翻转噪声 (0=关)

    retrieval_k: int = 3

    # Visual tokens — enabled with zero-init gate (warm start safe).
    use_visual_tokens: bool = True

    num_beams: int = 8
    length_penalty: float = 1.5
    min_length: int = 40
    no_repeat_ngram_size: int = 3
    use_sampling: bool = False

    early_stop_patience: int = 10

    eval_every: int = 1
    save_every: int = 5

    seed: int = 42
    use_bf16: bool = True
    num_workers: int = 2
