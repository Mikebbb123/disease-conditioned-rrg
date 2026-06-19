"""
Disease-conditioned T5: Perceiver visual resampler + SciFive T5.
Ensemble (xrv + disease_head) for inference hints; scheduled-sampled
GT findings for training. Finding-token-weighted CE prevents template collapse.
"""
import re
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer


# ============================================================
# Prompt -- hints only
# ============================================================

PROMPT_TEMPLATE = (
    "Key findings to check: {disease_hints}\n"
    "Write a radiology report for this chest X-ray."
)

DISEASE_NAMES = ["cardiomegaly", "pulmonary edema", "consolidation", "atelectasis", "pleural effusion"]


def _build_disease_text(disease_labels: torch.Tensor, threshold: float = 0.5) -> str:
    """Convert disease prediction vector to a text prompt."""
    if disease_labels is None:
        return "none identified"
    probs = disease_labels.float().cpu()
    found = [DISEASE_NAMES[i] for i, p in enumerate(probs) if p > threshold]
    return ", ".join(found) if found else "no acute findings suspected"


def build_input_prompts(disease_labels: torch.Tensor = None) -> List[str]:
    """Build prompts from disease labels only -- no retrieval text injected."""
    prompts = []
    for i in range(disease_labels.size(0) if disease_labels is not None else 1):
        d_text = _build_disease_text(disease_labels[i]) if disease_labels is not None else "none identified"
        prompts.append(PROMPT_TEMPLATE.format(disease_hints=d_text))
    return prompts


@dataclass
class DiseaseT5Outputs:
    gen_loss: torch.Tensor
    entropy: torch.Tensor
    cls_loss: torch.Tensor
    hint_positive_rate: torch.Tensor  # fraction of samples with >=1 positive hint


# ============================================================
# Perceiver Visual Resampler
# ============================================================

class PerceiverResampler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_queries: int = 32,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, output_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())
        self.cross_attn_layers = nn.ModuleList([
            _ResamplerLayer(output_dim, num_heads, dropout) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        B = spatial.size(0)
        spatial_flat = spatial.flatten(2).transpose(1, 2)
        spatial_flat = self.input_proj(spatial_flat)
        queries = self.query_tokens.expand(B, -1, -1)
        for layer in self.cross_attn_layers:
            queries = layer(queries, spatial_flat)
        return self.output_norm(queries)


class _ResamplerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, context):
        attn_out, _ = self.cross_attn(queries, context, context)
        queries = self.norm1(queries + self.dropout(attn_out))
        queries = self.norm2(queries + self.dropout(self.ffn(queries)))
        return queries


# ============================================================
# Main model
# ============================================================

class DiseaseT5(nn.Module):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        import torchxrayvision as xrv
        vision_model = getattr(model_config, "vision_model", "resnet")
        if vision_model == "resnet":
            self.vision_encoder = xrv.models.ResNet(weights=model_config.vision_weights)
            self._vision_type = "resnet"
        else:
            self.vision_encoder = xrv.models.DenseNet(weights=model_config.vision_weights)
            self._vision_type = "densenet"

        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        n_vision = sum(p.numel() for p in self.vision_encoder.parameters())

        if getattr(train_config, "unfreeze_vision_bn", True) and vision_model == "resnet":
            n_bn = 0
            for name, p in self.vision_encoder.model.layer4.named_parameters():
                if "bn" in name or "downsample" in name:
                    p.requires_grad = True
                    n_bn += p.numel()
            print(f"[Model] ResNet layer4 BN unfrozen: {n_bn:,} params")
        self.vision_encoder.eval()
        print(f"[Model] Vision encoder ({vision_model}): {n_vision:,} params, "
              f"layer4 BN unfrozen={getattr(train_config, 'unfreeze_vision_bn', True)}")

        # xrv hint source
        TARGET_PATH = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
        ALT_NAMES = {"Edema": "Pulmonary Edema", "Pleural Effusion": "Effusion"}
        xrv_paths = self.vision_encoder.pathologies
        self._xrv_indices = []
        for name in TARGET_PATH:
            if name in xrv_paths:
                self._xrv_indices.append(xrv_paths.index(name))
            elif name in ALT_NAMES and ALT_NAMES[name] in xrv_paths:
                self._xrv_indices.append(xrv_paths.index(ALT_NAMES[name]))
            else:
                raise RuntimeError(f"xrv pathology '{name}' not found")
        self._xrv_indices = torch.tensor(self._xrv_indices, dtype=torch.long)
        print(f"[Model] xrv hint source -- indices: {self._xrv_indices.tolist()}")

        # Perceiver
        n_queries = getattr(model_config, "num_query_tokens", 32)
        self.perceiver = PerceiverResampler(
            input_dim=model_config.visual_dim, output_dim=model_config.t5_hidden,
            num_queries=n_queries,
            num_heads=getattr(model_config, "perceiver_heads", 8),
            num_layers=getattr(model_config, "perceiver_layers", 2), dropout=0.2)
        print(f"[Model] Perceiver: {n_queries} queries")

        # Visual aligner
        n_align = getattr(model_config, "visual_aligner_layers", 2)
        self.visual_aligner = None
        if n_align > 0:
            self.visual_aligner = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=model_config.t5_hidden, nhead=8,
                    dim_feedforward=2048, dropout=0.2, batch_first=True, activation='gelu'),
                num_layers=n_align)

        # Zero-init gate: tanh(0)=0 -> visual tokens initially silent (warm start safe)
        self.visual_gate = nn.Parameter(torch.zeros(1))

        self.modality_emb = nn.Embedding(2, model_config.t5_hidden)
        nn.init.normal_(self.modality_emb.weight, std=0.02)

        # T5
        model_name = getattr(model_config, "model_name", "razent/SciFive-base-Pubmed_PMC")
        print(f"[Model] Loading T5: {model_name} ...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.t5_hidden = self.t5.config.d_model

        use_lora = getattr(model_config, "USE_LORA", False)
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=getattr(model_config, "lora_r", 16),
                lora_alpha=getattr(model_config, "lora_alpha", 32),
                lora_dropout=getattr(model_config, "lora_dropout", 0.1),
                target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
                task_type="SEQ_2_SEQ_LM")
            self.t5 = get_peft_model(self.t5, lora_cfg)
            n_lora = sum(p.numel() for p in self.t5.parameters() if p.requires_grad)
            print(f"[Model] T5 + LoRA: {n_lora:,} trainable")
        else:
            for p in self.t5.encoder.parameters():
                p.requires_grad = True
            n_t5 = sum(p.numel() for p in self.t5.parameters())
            print(f"[Model] T5: {n_t5:,} ALL UNFROZEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)

        # ---- Auxiliary disease classification head (2-layer MLP) ----
        # Input: pooled visual_tokens from perceiver/aligner output [B, t5_hidden].
        # cls_loss gradients flow back into perceiver, making visual tokens
        # disease-discriminative without adding any new readable channel to T5.
        num_cls = model_config.num_diseases
        hidden_cls = model_config.t5_hidden // 2  # 768 -> 384
        self.disease_head = nn.Sequential(
            nn.LayerNorm(model_config.t5_hidden),   # normalize perceiver output scale
            nn.Linear(model_config.t5_hidden, hidden_cls),
            nn.GELU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(hidden_cls, num_cls),
        )
        nn.init.xavier_uniform_(self.disease_head[-1].weight)
        nn.init.zeros_(self.disease_head[-1].bias)
        self._disease_loss_w = getattr(train_config, "disease_loss_weight", 1.0)
        # Hint scheduled-sampling params
        self._hint_gt_p0 = getattr(train_config, "hint_gt_prob_start", 1.0)
        self._hint_gt_p1 = getattr(train_config, "hint_gt_prob_end", 0.5)
        self._hint_gt_decay = getattr(train_config, "hint_gt_decay_epochs", train_config.num_epochs)
        self._hint_flip_p = getattr(train_config, "hint_flip_prob", 0.0)

        xrv_thr = getattr(model_config, "xrv_thresholds", (0.044, 0.052, 0.104, 0.136, 0.190))
        self._xrv_thresholds = torch.tensor(xrv_thr, dtype=torch.float)
        self._head_diag_threshold = 0.5
        # Ensemble thresholds -- tuned on val per-class F1
        # cardio/atel/eff from sweep; edema/consol frozen high (val n<3 = noise)
        ens_thr = getattr(model_config, "ensemble_thresholds",
                          (0.46, 0.60, 0.60, 0.43, 0.26))
        self._ensemble_thresholds = torch.tensor(ens_thr, dtype=torch.float)
        self.register_buffer("_pos_weight", None)
        print(f"[Model] Disease head: {model_config.t5_hidden} -> {hidden_cls} -> {num_cls} "
              f"(2-layer MLP, input=perceiver tokens), lambda={self._disease_loss_w}")

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total trainable: {n_trainable:,} params")

    def encode_visual(self, frontal, lateral):
        with torch.autocast(device_type="cuda", enabled=False):
            if next(self.vision_encoder.parameters()).device != frontal.device:
                self.vision_encoder = self.vision_encoder.to(frontal.device)
            if self._vision_type == "resnet":
                m = self.vision_encoder.model
                f_1 = m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(frontal.float())))))
                l_1 = m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(lateral.float())))))
                f_2 = m.layer2(f_1)
                l_2 = m.layer2(l_1)
                f_3 = m.layer3(f_2)
                l_3 = m.layer3(l_2)
                f_4 = m.layer4(f_3)
                l_4 = m.layer4(l_3)
                return (f_4 + l_4) / 2
            f_feats = self.vision_encoder.features(frontal.float())
            l_feats = self.vision_encoder.features(lateral.float())
            return (f_feats + l_feats) / 2 if f_feats.dim() == 4 else None

    def _get_visual_tokens(self, spatial, frontal=None, lateral=None):
        if spatial is None:
            if frontal is not None and lateral is not None and self._vision_type == "resnet":
                spatial = self.encode_visual(frontal, lateral)
            else:
                return None
        tokens = self.perceiver(spatial)
        if self.visual_aligner is not None:
            tokens = self.visual_aligner(tokens)
        tokens = tokens + self.modality_emb(torch.zeros(1, dtype=torch.long, device=tokens.device))
        return tokens

    def _build_t5_inputs_with_visual(self, visual_tokens, text_input_ids, text_attention_mask):
        B, N_text = text_input_ids.shape
        device = text_input_ids.device
        text_emb = self.t5.encoder.embed_tokens(text_input_ids)
        text_emb = text_emb + self.modality_emb(torch.ones(1, dtype=torch.long, device=device))
        gated_visual = torch.tanh(self.visual_gate) * visual_tokens
        inputs_embeds = torch.cat([gated_visual, text_emb], dim=1)
        N_vis = visual_tokens.size(1)
        visual_mask = torch.ones(B, N_vis, dtype=text_attention_mask.dtype, device=device)
        attention_mask = torch.cat([visual_mask, text_attention_mask], dim=1)
        return inputs_embeds, attention_mask

    def prepare_labels(self, target_reports, device):
        label_inputs = self.tokenizer(target_reports, return_tensors="pt", padding=True,
                                       truncation=True, max_length=self.train_config.max_target_length).to(device)
        labels = label_inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def prepare_labels_and_weights(self, target_reports, device, finding_w=4.0):
        """Tokenize reports and produce (labels, token_weights) where tokens
        inside sentences that contain positive disease findings are upweighted.
        This prevents the model from ignoring finding tokens to minimise CE."""
        from data_utils_compat import extract_disease_labels

        enc = self.tokenizer(
            target_reports, return_tensors="pt", padding=True, truncation=True,
            max_length=self.train_config.max_target_length,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        offsets   = enc["offset_mapping"]      # [B, T, 2]
        attn      = enc["attention_mask"]      # [B, T]

        labels = input_ids.clone()
        labels[attn == 0] = -100

        B, T = input_ids.shape
        weights = torch.ones(B, T, dtype=torch.float)

        for b, report in enumerate(target_reports):
            # Find character spans of sentences that contain positive findings
            spans, pos = [], 0
            for sent in re.split(r'(?<=[.!?])\s+', report):
                if not sent:
                    continue
                start = report.find(sent, pos)
                if start < 0:
                    start = pos
                end = start + len(sent)
                pos = end
                if extract_disease_labels(sent.lower()):
                    spans.append((start, end))
            if not spans:
                continue
            # Weight tokens whose offsets fall inside a finding-sentence span
            for t in range(T):
                if attn[b, t] == 0:
                    continue
                cs, ce = int(offsets[b, t, 0]), int(offsets[b, t, 1])
                if cs == ce:          # special token offset is (0,0)
                    continue
                if any(cs >= s and ce <= e for (s, e) in spans):
                    weights[b, t] = finding_w

        return labels.to(device), weights.to(device)

    @torch.no_grad()
    def _xrv_disease_probs(self, frontal, lateral):
        if self._vision_type != "resnet":
            raise RuntimeError("xrv hints require ResNet")
        with torch.autocast(device_type="cuda", enabled=False):
            pf = self.vision_encoder(frontal.float())
            pl = self.vision_encoder(lateral.float())
        p = (pf + pl) / 2
        return p[:, self._xrv_indices.to(p.device)]

    def _get_disease_predictions(self, visual_tokens):
        """Extract disease probabilities from perceiver visual tokens via the MLP head.

        visual_tokens: [B, num_query_tokens, t5_hidden] from perceiver/aligner.
        Returns [B, num_diseases] float tensor of probabilities.
        """
        pooled = visual_tokens.mean(dim=1)  # [B, t5_hidden]
        logits = self.disease_head(pooled)  # [B, num_diseases]
        return torch.sigmoid(logits)

    @torch.no_grad()
    def set_disease_class_weights(self, train_dataset):
        all_labels = torch.stack([s["disease_labels"] for s in train_dataset.samples])
        pos = all_labels.sum(dim=0)
        neg = len(all_labels) - pos
        pos_weight = (neg / pos.clamp(min=1)).clamp(max=10.0).to(self.disease_head[-1].weight.device)
        self._pos_weight = pos_weight
        prevalence = pos / len(all_labels)
        print(f"[Model] Disease class weights set from {len(all_labels)} samples:")
        for i, name in enumerate(DISEASE_NAMES):
            print(f"  {name:20s}  pos={int(pos[i]):4d}  neg={int(neg[i]):4d}  "
                  f"pos_weight={pos_weight[i]:.1f}  prevalence={prevalence[i]:.2%}")

    @torch.no_grad()
    def diagnose_disease_head(self, dataset, device="cuda", max_samples=300):
        import numpy as np
        from torch.utils.data import DataLoader
        from data import rag_collate_fn
        loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=rag_collate_fn, num_workers=0)
        all_probs, all_xrv, all_labels = [], [], []
        for batch in loader:
            frontal = batch["frontal"].to(device)
            lateral = batch["lateral"].to(device)
            spatial = self.encode_visual(frontal, lateral)
            if spatial is not None:
                visual_tokens = self._get_visual_tokens(spatial, frontal, lateral)
                pooled = visual_tokens.mean(dim=1)
                probs = torch.sigmoid(self.disease_head(pooled))
                all_probs.append(probs.cpu())
                # Also collect xrv probs for ensemble sweep
                xp = self._xrv_disease_probs(frontal, lateral).cpu()
                all_xrv.append(xp)
                all_labels.append(batch["disease_labels"].cpu())
            if len(all_probs) * frontal.size(0) >= max_samples:
                break
        probs = torch.cat(all_probs, dim=0)
        xrv_p = torch.cat(all_xrv, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = (probs > self._head_diag_threshold).float()

        print(f"\n[Diagnose] Disease head P/R/F1 (threshold={self._head_diag_threshold}, n={len(probs)}):")
        print(f"{'Disease':22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Pred+':>6s} {'True+':>6s}")
        print("-" * 58)
        per_class = {}
        for i, name in enumerate(DISEASE_NAMES):
            tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum().item()
            fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum().item()
            fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum().item()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            n_pred_pos = preds[:, i].sum().item()
            n_true_pos = labels[:, i].sum().item()
            per_class[name] = {"P": p, "R": r, "F1": f1}
            print(f"{name:22s} {p:6.1%} {r:6.1%} {f1:6.1%} {n_pred_pos:6.0f} {n_true_pos:6.0f}")

        # Macro
        p_vals = [v["P"] for v in per_class.values()]
        r_vals = [v["R"] for v in per_class.values()]
        f_vals = [v["F1"] for v in per_class.values()]
        print("-" * 58)
        print(f"{'MACRO (head)':22s} {sum(p_vals)/5:6.1%} {sum(r_vals)/5:6.1%} {sum(f_vals)/5:6.1%}")

        # ===== Per-class F1 optimal threshold sweep -- disease_head / xrv / ensemble =====
        P_np, X_np, Y_np = probs.numpy(), xrv_p.numpy(), labels.numpy()
        E_np = (P_np + X_np) / 2  # ensemble

        def _f1_at(pred, y):
            tp = float(((pred == 1) & (y == 1)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            P = tp / (tp + fp) if tp + fp > 0 else 0.0
            R = tp / (tp + fn) if tp + fn > 0 else 0.0
            return (2 * P * R / (P + R) if P + R > 0 else 0.0), P, R

        for label, M in [("disease_head", P_np), ("xrv", X_np), ("ensemble (head+xrv)", E_np)]:
            print(f"\n[Tune] {label} -- per-class F1 optimal threshold:")
            print(f"{'Disease':22s} {'n_pos':>5s} {'F1@.5':>6s} {'bestF1':>7s} {'thr':>5s} {'Prec':>6s} {'Rec':>6s}")
            print("-" * 66)
            tuned_thrs, best_f1s = [], []
            for i, name in enumerate(DISEASE_NAMES):
                p, y = M[:, i], Y_np[:, i]
                npos = int(y.sum())
                f1_05, _, _ = _f1_at(p > 0.5, y)
                bf1, bt = max((_f1_at(p > t, y)[0], t) for t in np.arange(0.05, 0.95, 0.01))
                _, bP, bR = _f1_at(p > bt, y)
                tuned_thrs.append(round(float(bt), 3))
                best_f1s.append(bf1)
                note = ""
                if npos < 10:
                    note = "  WARNING n<10 -- noisy"
                print(f"{name:22s} {npos:5d} {f1_05:6.1%} {bf1:7.1%} {bt:5.2f} {bP:6.1%} {bR:6.1%}{note}")
            print("-" * 66)
            print(f"{'MACRO best-F1':22s} {'':5s} {'':6s} {sum(best_f1s)/5:7.1%}")
            print(f"  thresholds = {tuned_thrs}")

        return per_class

    @torch.no_grad()
    def diagnose_xrv_hints(self, dataset, device="cuda", max_samples=500):
        """Print per-class xrv hint positive rate on a dataset.
        If xrv is the weak 'all-normal' predictor we suspect, most classes
        will show <5% positive rate -- confirming the inference bottleneck."""
        from torch.utils.data import DataLoader
        from data import rag_collate_fn

        loader = DataLoader(dataset, batch_size=16, shuffle=False,
                          collate_fn=rag_collate_fn, num_workers=0)

        all_xrv, all_gt = [], []
        for batch in loader:
            frontal = batch["frontal"].to(device)
            lateral = batch["lateral"].to(device)
            probs = self._xrv_disease_probs(frontal, lateral)
            labels = (probs > self._xrv_thresholds.to(device)).float()
            all_xrv.append(labels.cpu())
            all_gt.append(batch["disease_labels"].cpu())
            if len(all_xrv) * frontal.size(0) >= max_samples:
                break

        xrv  = torch.cat(all_xrv, dim=0)   # [N, 5]
        gt   = torch.cat(all_gt, dim=0)

        print(f"\n[Diagnose] xrv hint positive rate (thresholds={self._xrv_thresholds.tolist()}, n={len(xrv)}):")
        print(f"{'Disease':22s} {'xrv+%':>7s} {'GT+%':>7s} {'xrv+':>6s} {'GT+':>6s}")
        print("-" * 54)
        for i, name in enumerate(DISEASE_NAMES):
            xp = xrv[:, i].sum().item()
            gp = gt[:, i].sum().item()
            print(f"{name:22s} {100*xp/len(xrv):6.1f}% {100*gp/len(gt):6.1f}% "
                  f"{xp:6.0f} {gp:6.0f}")

        any_xrv = (xrv.any(dim=1).sum().item())
        any_gt  = (gt.any(dim=1).sum().item())
        print("-" * 54)
        print(f"{'ANY abnormal':22s} {100*any_xrv/len(xrv):6.1f}% {100*any_gt/len(gt):6.1f}% "
              f"{any_xrv:6.0f} {any_gt:6.0f}")
        print()

    def _hint_gt_prob(self, epoch: int) -> float:
        if self._hint_gt_decay <= 0:
            return self._hint_gt_p1
        frac = min(1.0, epoch / self._hint_gt_decay)
        return self._hint_gt_p0 + (self._hint_gt_p1 - self._hint_gt_p0) * frac

    def _perturb_hint(self, hint_labels):
        if self._hint_flip_p <= 0:
            return hint_labels
        flip = (torch.rand_like(hint_labels) < self._hint_flip_p).float()
        return hint_labels * (1 - flip) + (1 - hint_labels) * flip

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, frontal, lateral, target_reports,
                disease_labels=None, training=True, current_epoch=0):
        device = frontal.device
        spatial = self.encode_visual(frontal, lateral)
        visual_tokens = self._get_visual_tokens(spatial, frontal, lateral)

        # ---- Disease hints: train with GT findings (scheduled-sampled mix with xrv) ----
        cls_loss = torch.tensor(0.0, device=device)
        xrv_probs = self._xrv_disease_probs(frontal, lateral)                # [B, 5]
        xrv_labels = (xrv_probs > self._xrv_thresholds.to(device)).float()

        if training and disease_labels is not None:
            gt = disease_labels.float().to(device)
            p_gt = self._hint_gt_prob(current_epoch)
            use_gt = (torch.rand(gt.size(0), 1, device=device) < p_gt).float()
            hint_labels = use_gt * gt + (1.0 - use_gt) * xrv_labels
            hint_labels = self._perturb_hint(hint_labels)
        else:
            hint_labels = xrv_labels

        prompts = build_input_prompts(hint_labels)

        # Auxiliary classification loss -- pooled from perceiver visual tokens.
        if training and disease_labels is not None and visual_tokens is not None:
            pooled = visual_tokens.mean(dim=1)          # [B, 768]
            cls_logits = self.disease_head(pooled)
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits, disease_labels.float().to(device), pos_weight=self._pos_weight)

        text_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.train_config.max_input_length).to(device)

        # ---- Labels: weighted by finding sentences during training ----
        if training:
            labels, tok_w = self.prepare_labels_and_weights(
                target_reports, device,
                finding_w=getattr(self.train_config, "finding_token_weight", 4.0))
        else:
            labels = self.prepare_labels(target_reports, device)
            tok_w = None

        # ---- T5 input: gate visual tokens ----
        if getattr(self.train_config, "use_visual_tokens", False):
            inputs_embeds, attention_mask = self._build_t5_inputs_with_visual(
                visual_tokens, text_inputs.input_ids, text_inputs.attention_mask)
        else:
            inputs_embeds = self.t5.encoder.embed_tokens(text_inputs.input_ids)
            attention_mask = text_inputs.attention_mask

        outputs = self.t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          labels=labels, return_dict=True)
        logits = outputs.logits.float()

        ls = getattr(self.train_config, "label_smoothing", 0.1) if training else 0.0
        if training and tok_w is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=-100, label_smoothing=ls, reduction="none")
            wf = tok_w.view(-1)
            valid = (labels.view(-1) != -100).float()
            gen_loss = (ce * wf).sum() / (wf * valid).sum().clamp(min=1.0)
        else:
            gen_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                        ignore_index=-100, label_smoothing=ls)
        total_loss = gen_loss + self._disease_loss_w * cls_loss

        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            probs_out = torch.exp(log_probs)
            ent = -(probs_out * log_probs).sum(dim=-1)
            valid = (labels != -100).float()
            entropy = (ent * valid).sum() / valid.sum().clamp(min=1)

        hint_pos_rate = hint_labels.any(dim=1).float().mean().detach() if hint_labels is not None else torch.tensor(-1.0)
        return DiseaseT5Outputs(gen_loss=total_loss, entropy=entropy.detach(),
                          cls_loss=cls_loss.detach(), hint_positive_rate=hint_pos_rate)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, frontal, lateral,
                 num_beams=8, max_length=150, use_sampling=False,
                 length_penalty=None, min_length=None, no_repeat_ngram_size=3,
                 gt_labels=None):
        """Generate reports.  When gt_labels is provided (oracle mode), uses
        ground-truth findings as hints -- this isolates decoder quality from
        image->findings predictor quality."""
        device = frontal.device
        spatial = self.encode_visual(frontal, lateral)
        visual_tokens = self._get_visual_tokens(spatial, frontal, lateral)

        if gt_labels is not None:
            # Oracle: GT findings -> hint (isolates decoder from predictor)
            disease_labels = gt_labels.float().to(device)
        else:
            # Ensemble: xrv (under-predicts) + disease_head (over-predicts)
            # with per-class thresholds tuned on val F1
            xrv_probs  = self._xrv_disease_probs(frontal, lateral)
            head_probs = self._get_disease_predictions(visual_tokens)
            ens = 0.5 * (xrv_probs + head_probs)
            disease_labels = (ens > self._ensemble_thresholds.to(device)).float()
        prompts = build_input_prompts(disease_labels)

        text_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.train_config.max_input_length).to(device)

        if getattr(self.train_config, "use_visual_tokens", False):
            inputs_embeds, attention_mask = self._build_t5_inputs_with_visual(
                visual_tokens, text_inputs.input_ids, text_inputs.attention_mask)
        else:
            inputs_embeds = self.t5.encoder.embed_tokens(text_inputs.input_ids)
            attention_mask = text_inputs.attention_mask

        gen_kwargs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                       "max_length": max_length, "no_repeat_ngram_size": no_repeat_ngram_size}
        if length_penalty is not None:
            gen_kwargs["length_penalty"] = length_penalty
        if min_length is not None:
            gen_kwargs["min_length"] = min_length
        if use_sampling:
            gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.5})
        else:
            gen_kwargs.update({"num_beams": num_beams, "early_stopping": True})
        out = self.t5.generate(**gen_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def trainable_state_dict(self):
        state = {
            "perceiver": self.perceiver.state_dict(),
            "visual_aligner": self.visual_aligner.state_dict() if self.visual_aligner is not None else {},
            "modality_emb": self.modality_emb.state_dict(),
            "t5": self.t5.state_dict(),
            "disease_head": self.disease_head.state_dict(),
        }
        state["visual_gate"] = self.visual_gate.data
        if self._pos_weight is not None:
            state["_pos_weight"] = self._pos_weight.clone()
        return state

    def load_trainable_state_dict(self, state):
        if "perceiver" in state:
            self.perceiver.load_state_dict(state["perceiver"])
        if "visual_aligner" in state and state["visual_aligner"]:
            self.visual_aligner.load_state_dict(state["visual_aligner"])
        if "modality_emb" in state:
            self.modality_emb.load_state_dict(state["modality_emb"])
        if "t5" in state:
            self.t5.load_state_dict(state["t5"])
        if "disease_head" in state:
            try:
                self.disease_head.load_state_dict(state["disease_head"])
            except RuntimeError:
                print("[Load] disease_head shape mismatch (old 2048->1024 vs new "
                      "768->384) -- using fresh random init for disease_head.")
        if "visual_gate" in state:
            self.visual_gate.data.copy_(state["visual_gate"])
        # (if not in state, keep zero-init -- warm start from baseline)
        if "_pos_weight" in state:
            self._pos_weight = state["_pos_weight"].to(self.disease_head[-1].weight.device)
