

    #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token Compression Attention for GLUE — Production-Ready Version
================================================================
Fixes all critical bugs and adds:
1. Proper weight initialization from pretrained BERT
2. Correct value compression with orthogonal projection
3. Gradient checkpointing for memory efficiency
4. LayerNorm weight renaming for checkpoint compatibility
5. Model saving/loading with compression modules
6. Mixed precision training support
7. Early stopping and better hyperparameters
"""

import os
import math
import argparse
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"  # though we use local parquet, still good

# ---------- Offline settings ----------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, default_data_collator,
    EarlyStoppingCallback, get_cosine_schedule_with_warmup,
    set_seed, PreTrainedModel, PretrainedConfig
)
from datasets import Dataset, DatasetDict
import evaluate
os.environ["HF_HUB_OFFLINE"] = "1"

# ── Environment ─────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── 1. Compression Configuration ────────────────────────────
@dataclass
class CompressionConfig:
    # Linformer
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    progressive: bool = True
    use_orthogonal: bool = True  # (ignored in current impl)
    use_gate: bool = True
    gate_bias: float = 0.0
    max_seq_len: int = 128
    attn_score_clamp: float = 50.0
    dropout: float = 0.1
    ortho_reg_weight: float = 0.001

    # LEKA
    attn_type: str = "linformer"   # "linformer" or "leka"
    num_landmarks: int = 16
    feature_dim: int = 128
    use_spherical_norm: bool = True
    kernel: str = "elu"

    def get_k_for_layer(self, layer_idx: int) -> int:
        if not self.progressive:
            return self.k_mid
        if layer_idx < 4:
            return self.k_early
        elif layer_idx < 8:
            return self.k_mid
        else:
            return self.k_late

# ── 2. Data Loading (Local Parquet) ────────────────────────
TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
"""
LEKAv2 — Research‑backed Linear‑time Efficient Kernel Attention v2
====================================================================
A drop‑in replacement for DistilBERT attention, integrating:

  1. Performer FAVOR+ (ICLR 2021)     – orthogonal random features, learnable
  2. Softpick (arXiv, Apr 2025)       – rectified softmax, ~47 % sparse maps
  3. GPA Geometric Landmarks (2025)   – content‑aware landmark selection
  4. Hybrid‑Attention Analysis (2025) – local window gating, 3:1 ratio
  5. RALA Rank Augmentation (CVPR 25) – adaptive per‑head landmark budget

Paper references:
  • Performer / FAVOR+: Choromanski et al., ICLR 2021 (arXiv:2009.14794)
  • Softpick: Zuhri et al., arXiv 2504.20966 (Apr 2025)
  • GPA: On the Geometric Selection of Landmarks, OpenReview 2025
  • Hybrid analysis: Wang et al., arXiv 2507.06457 (Aug 2025)
  • RALA: Fan et al., CVPR 2025 (arXiv:2411.07635)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LEKAv2Attention(nn.Module):
    """
    LEKAv2: Linear‑time Efficient Kernel Attention with softpick,
            content‑aware landmarks, orthogonal features, and local gating.
    """
    def __init__(
        self,
        config,
        comp_config,
        layer_idx,
        total_layers=6,
        bert_mode=False,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.head_dim = self.attention_head_size
        self.bert_mode = bert_mode
        self.layer_idx = layer_idx

        # ── Landmark configuration ──
        self.num_landmarks = getattr(comp_config, 'num_landmarks', 16)
        self.feature_dim   = getattr(comp_config, 'feature_dim', 128)

        # ── Q / K / V projections ──
        self.query  = nn.Linear(config.hidden_size, self.all_head_size)
        self.key    = nn.Linear(config.hidden_size, self.all_head_size)
        self.value  = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(comp_config, 'dropout', 0.1))

        # ── Improvement 1: TWO learnable orthogonalised RFF projections ──
        # (FAVOR+ from Performer, ICLR 2021)
        self.rand_proj1 = nn.Parameter(
            torch.randn(self.head_dim, self.feature_dim) * 0.02
        )
        self.rand_proj2 = nn.Parameter(
            torch.randn(self.head_dim, self.feature_dim // 2) * 0.02
        )

        # ── Improvement 2: Content‑aware landmark scorer ──
        # (GPA Geometric Progressive Attention, 2025)
        # Total feature size after concatenating both kernel maps:
        #   f1: 2 * feature_dim
        #   f2: 2 * (feature_dim // 2)
        total_feat_dim = 2 * self.feature_dim + 2 * (self.feature_dim // 2)
        self.landmark_scorer = nn.Sequential(
            nn.Linear(total_feat_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, 1)
        )

        # ── Improvement 3: Per‑head landmark correction ──
        self.W_corr = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(
                self.num_attention_heads, 1, 1
            )
        )

        # ── Improvement 4: Local window gate ──
        # (Hybrid‑Attention systematic analysis, 2025)
        self.local_window_size = 5  # ±2 tokens
        self.local_gate = nn.Parameter(
            torch.zeros(self.num_attention_heads) + 0.1
        )

        # ── Improvement 5: Adaptive landmark budget ──
        # (RALA rank‑augmented, CVPR 2025)
        self.landmark_logits = nn.Parameter(
            torch.zeros(self.num_attention_heads) + math.log(self.num_landmarks / 2)
        )

        # ── Optional gate (from original LEKA) ──
        if getattr(comp_config, 'use_gate', False) and not bert_mode:
            self.gate = nn.Parameter(
                torch.tensor(getattr(comp_config, 'gate_bias', 0.0))
            )
        else:
            self.gate = None

    # ── Reshape helper ──────────────────────────────────────────────
    def transpose_for_scores(self, x):
        """(B, N, all_head) → (B, H, N, D)"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    # ── Multi‑kernel feature map (Performer FAVOR+, ICLR 2021) ─────
    def kernel_feature_map(self, x):
        """
        x: (B, H, N, D)
        Returns features from TWO concatenated orthogonalised projections.
        """
        W1 = self.rand_proj1.to(x.device)
        W2 = self.rand_proj2.to(x.device)

        # projection 1: cos + sin
        proj1 = torch.matmul(x, W1)
        f1 = torch.cat([torch.cos(proj1), torch.sin(proj1)], dim=-1)
        f1 = f1 / math.sqrt(self.feature_dim)

        # projection 2: cos + sin
        proj2 = torch.matmul(x, W2)
        f2 = torch.cat([torch.cos(proj2), torch.sin(proj2)], dim=-1)
        f2 = f2 / math.sqrt(self.feature_dim // 2)

        return torch.cat([f1, f2], dim=-1)  # (B, H, N, 2d1+2d2)

    # ── Content‑aware landmark selection ────────────────────────────
    def select_landmarks(self, K_features, V):
        """
        GPA‑inspired (2025): score tokens and select top‑L.
        Falls back to uniform sampling at initialisation.
        """
        B, H, N, feat_dim = K_features.shape
        L_max = self.num_landmarks

        if N <= L_max:
            return K_features, V

        # Score each token
        scores = self.landmark_scorer(K_features).squeeze(-1)  # (B, H, N)

        # Adaptive per‑head landmark budget (RALA, CVPR 2025)
        budget = torch.sigmoid(self.landmark_logits)  # (H,) ∈ (0, 1)
        L_per_head = torch.clamp(
            (budget * L_max).long(), min=4, max=L_max
        )

        # For simplicity (batch uniformity), use max budget
        L = L_max
        _, idx = scores.topk(L, dim=-1)  # (B, H, L)

        K_land = torch.gather(
            K_features, 2,
            idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        )
        V_land = torch.gather(
            V, 2,
            idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1])
        )
        return K_land, V_land

    # ── Softpick rectified softmax (arXiv 2504.20966, 2025) ────────
    @staticmethod
    def softpick(x, dim=-1, temperature=1.0):
        """~47 % sparse, no attention sink."""
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        denom = e - torch.exp(-m / temperature)
        p = F.relu(denom)
        return p / (p.sum(dim=dim, keepdim=True) + 1e-9)

    # ── Main forward pass ───────────────────────────────────────────
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, D = hidden_states.shape
        H = self.num_attention_heads
        L = self.num_landmarks
        head_dim = self.head_dim

        # 1. Linear projections
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        # 2. Multi‑kernel feature map (FAVOR+, Performer)
        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)

        # 3. Content‑aware landmark selection (GPA, 2025)
        K_land, V_land = self.select_landmarks(K_features, V)

        # 4. Landmark value correction
        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum('bhld,bhde->bhle', V_land, W)

        # 5. Kernel attention scores
        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))

        # ── Mask handling (unchanged from your working version) ──
        mask = attention_mask
        if mask is None:
            mask = kwargs.get('attn_mask', None)
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
                if N > L:
                    indices = torch.linspace(
                        0, N - 1, L, device=token_valid.device
                    ).long()
                    landmark_mask = token_valid[:, indices]
                else:
                    landmark_mask = token_valid
                landmark_mask = landmark_mask.unsqueeze(1).unsqueeze(2)
            else:
                if mask.dim() == 4:
                    mask_2d = mask.squeeze(1).squeeze(1) > -1e9
                elif mask.dim() == 3:
                    mask_2d = mask.squeeze(1).bool()
                elif mask.dim() == 2:
                    mask_2d = mask.bool()
                else:
                    raise ValueError(f"Unexpected mask dim: {mask.dim()}")
                if N > L:
                    indices = torch.linspace(
                        0, N - 1, L, device=mask_2d.device
                    ).long()
                    landmark_mask = mask_2d[:, indices]
                else:
                    landmark_mask = mask_2d
                landmark_mask = landmark_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(
                ~landmark_mask, -10000.0
            )

        # 6. Softpick rectified softmax
        attn_weights = self.softpick(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. LEKA context
        context_leka = torch.matmul(attn_weights, V_land_adj)

        # 8. Local window gating (hybrid attention, 2025)
        idx = torch.arange(N, device=hidden_states.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = (dist <= self.local_window_size // 2).unsqueeze(0).unsqueeze(0)

        scale = head_dim ** -0.5
        local_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~window_mask, -1e9)
        local_attn = F.softmax(local_scores, dim=-1)
        context_local = torch.matmul(local_attn, V)

        # Learnable blend
        gate_val = torch.sigmoid(self.local_gate).view(1, H, 1, 1)
        context = gate_val * context_leka + (1 - gate_val) * context_local

        # 9. Reshape + output
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_shape)

        return context, None

    def get_ortho_loss(self):
        return 0.0

# ── Injection helper ──────────────────────────────────────────────
def inject_leka_v2_into_distilbert(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    """Replace DistilBERT attention with LEKAv2."""
    new_attn = LEKAv2Attention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    # Copy pretrained weights
    new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
    new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
    new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
    new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
    new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
    new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
    new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
    new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)

    layer.attention = new_attn
    return new_attn



def load_glue_local(task_name: str) -> DatasetDict:
    """Load GLUE dataset from local parquet files (raw, no tokenization)."""
    base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        # split name: "train", "validation", etc. (works with both "train.parquet" and "train-00000-of-00001.parquet")
        split = fname.split("-")[0]
        file_path = os.path.join(task_path, fname)
        splits[split] = Dataset.from_parquet([file_path])

    # Handle MNLI special splits
    if task_name == "mnli":
        for split_name in ["validation_matched", "validation_mismatched"]:
            fname = f"{split_name}.parquet"
            file_path = os.path.join(task_path, fname)
            if os.path.exists(file_path):
                splits[split_name] = Dataset.from_parquet([file_path])

    return DatasetDict(splits)

def get_num_labels_and_is_regression(task_name: str):
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    if t == "cola":
        return 2, False
    glue_num_labels = {
        "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
        "qnli": 2, "rte": 2, "wnli": 2,
    }
    return glue_num_labels[t], False

GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

class LinformerProjection(nn.Module):
    """Learnable projection for keys and values (no orthogonal constraint)."""
    def __init__(self, n: int, k: int):
        super().__init__()
        # Initialise as truncated identity + small noise
        base = torch.eye(n)[:k, :]          # (k, n)
        self.E = nn.Parameter(base.clone())
        self.F = nn.Parameter(base.clone())
        with torch.no_grad():
            self.E += torch.randn_like(self.E) * 0.02
            self.F += torch.randn_like(self.F) * 0.02

    def forward(self, K, V, mask_2d=None):
        # K, V: (B, heads, N, d)
        B, H, N, d = K.shape
        if mask_2d is not None:
            mask = mask_2d.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
            K = K * mask
            V = V * mask

        # Combine batch and heads for batched matrix multiplication
        K_flat = K.reshape(B * H, N, d)     # (B*H, N, d)
        V_flat = V.reshape(B * H, N, d)

        E_exp = self.E.unsqueeze(0).expand(B * H, -1, -1)   # (B*H, k, N)
        F_exp = self.F.unsqueeze(0).expand(B * H, -1, -1)

        # Project: (B*H, k, N) @ (B*H, N, d) -> (B*H, k, d)
        K_comp = torch.bmm(E_exp, K_flat)
        V_comp = torch.bmm(F_exp, V_flat)

        # Reshape back to (B, heads, k, d)
        K_comp = K_comp.reshape(B, H, self.E.shape[0], d)
        V_comp = V_comp.reshape(B, H, self.F.shape[0], d)
        return K_comp, V_comp
class TokenCompressedAttention(nn.Module):
    def __init__(self, config, comp_config, layer_idx, bert_mode=False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.comp_config = comp_config
        self.layer_idx = layer_idx
        self.k = comp_config.get_k_for_layer(layer_idx)
        self.bert_mode = bert_mode

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(comp_config.dropout)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        # Linformer-style projection (free parameters)
        self.projection = LinformerProjection(comp_config.max_seq_len, self.k)

        if comp_config.use_gate and not self.bert_mode:
            self.gate = nn.Parameter(torch.tensor(comp_config.gate_bias))
        else:
            self.gate = None

        self._ortho_loss = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, _ = hidden_states.shape
        L = self.num_landmarks

        # --- Linear projections ------------------------------------------------
        Q = self.transpose_for_scores(self.query(hidden_states))   # (B, H, N, D)
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)   # (B, H, N, 2*feat_dim)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V)  # (B, H, L, ...)

        # Corrected values
        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)   # (B, H, D, D)
        V_land_adj = torch.einsum('bhld,bhde->bhle', V_land, W)  # (B, H, L, D)

        # Kernel attention scores
        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))  # (B, H, N, L)

        # --- Mask handling ----------------------------------------------------
        # DistilBERT passes mask as `attn_mask`; BERT sometimes as `attention_mask`.
        # Unify them here.
        mask = attention_mask
        if mask is None:
            mask = kwargs.get('attn_mask', None)

        if mask is not None:
            # Convert any mask to 2D booleans (True = keep this token)
            if mask.dim() == 2:                     # (B, N)
                mask_2d = mask.bool()
            elif mask.dim() == 3:                   # (B, 1, N)
                mask_2d = mask.squeeze(1).bool()
            elif mask.dim() == 4:                   # (B, 1, 1, N) e.g. additive masks
                mask_2d = mask.squeeze(1).squeeze(1) > -1e9
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Sub‑sample to get mask for landmarks
            if N > L:
                # Take L evenly spaced positions (you can also use k‑means++ later)
                indices = torch.linspace(0, N-1, L, device=mask_2d.device).long()
                landmark_mask = mask_2d[:, indices]            # (B, L)
            else:
                landmark_mask = mask_2d                        # (B, N)

            # Broadcast to (B, 1, 1, L) so it can be applied to attn_weights
            landmark_mask = landmark_mask.unsqueeze(1).unsqueeze(2)

            # Apply additive masking: set scores for invalid landmarks to a large negative number
            attn_weights = attn_weights.masked_fill(~landmark_mask, -10000.0)

        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output computation
        context_layer = torch.matmul(attn_weights, V_land_adj)  # (B, H, N, D)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        return context_layer, None


    def get_ortho_loss(self):
        return 0.0




class CompressedBertAttention(nn.Module):
    def __init__(self, original_attention, compressed_self_attn):
        super().__init__()
        self.compressed_self = compressed_self_attn   # TokenCompressedAttention (normal mode)
        self.output = original_attention.output        # BertSelfOutput (dense will be replaced with Identity)
        self.output_dropout = original_attention.output.dropout   # keep original dropout
        self.pruned_heads = getattr(original_attention, 'pruned_heads', set())

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        attn_result = self.compressed_self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        self.bert.config.gradient_checkpointing = False
        if hasattr(self.bert, 'gradient_checkpointing_disable'):
            self.bert.gradient_checkpointing_disable()

        if isinstance(attn_result, tuple):
            context, attn_weights = attn_result
        else:
            context, attn_weights = attn_result, None

        # Apply the same dropout that original BertSelfOutput used to have after the dense
        context = self.output_dropout(context)
        # Now call BertSelfOutput (its dense is now Identity, so it only does residual + LayerNorm)
        layer_output = self.output(context, hidden_states)

        return (layer_output, attn_weights)    

            # ── 5. Inject Compression into BERT ─────────────────────────
            
def inject_compression_into_layer(layer, comp_config, layer_idx, model_config, model_type="bert"):
    if comp_config.attn_type == "leka":
        new_attn = LEKAAttention(model_config, comp_config, layer_idx, bert_mode=False)
    else:
        new_attn = TokenCompressedAttention(model_config, comp_config, layer_idx, bert_mode=False)

    if model_type == 'bert':
        old_self = layer.attention.self
        new_attn.query.weight.data.copy_(old_self.query.weight.data)
        new_attn.query.bias.data.copy_(old_self.query.bias.data)
        new_attn.key.weight.data.copy_(old_self.key.weight.data)
        new_attn.key.bias.data.copy_(old_self.key.bias.data)
        new_attn.value.weight.data.copy_(old_self.value.weight.data)
        new_attn.value.bias.data.copy_(old_self.value.bias.data)
        # copy output weights
        if hasattr(old_self, 'output'):
            new_attn.out_proj.weight.data.copy_(old_self.output.weight.data)
            new_attn.out_proj.bias.data.copy_(old_self.output.bias.data)
        else:  # for BERT, output dense is in BertSelfOutput
            output_dense = layer.attention.output.dense
            new_attn.out_proj.weight.data.copy_(output_dense.weight.data)
            new_attn.out_proj.bias.data.copy_(output_dense.bias.data)
        # layer.attention.output.dense = nn.Identity()
        replacement = CompressedBertAttention(layer.attention, new_attn)
        replacement.output.dense = nn.Identity()  # ✓
        layer.attention = replacement

        replacement = CompressedBertAttention(layer.attention, new_attn)
        layer.attention = replacement
        return new_attn
    elif model_type == 'distilbert':
        if comp_config.attn_type == "leka":
            new_attn = LEKAAttention(model_config, comp_config, layer_idx, bert_mode=False)
            old_attn = layer.attention
            new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
            new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
            new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
            new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
            new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
            new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
            layer.attention = new_attn
            return new_attn

        elif comp_config.attn_type == "leka_v2":
            # استفاده از تزریق‌گر اختصاصی LEKAv2
            new_attn = inject_leka_v2_into_distilbert(
                layer, model_config, comp_config, layer_idx,
                total_layers=6,   # یا از تنظیمات مدل بخوانید
                old_attn=layer.attention
            )
            return new_attn       # خود تابع جایگزینی را انجام می‌دهد

        else:
            new_attn = TokenCompressedAttention(model_config, comp_config, layer_idx, bert_mode=False)
            old_attn = layer.attention
            new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
            new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
            new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
            new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
            new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
            new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
            layer.attention = new_attn
            return new_attn


    else:   
        raise NotImplementedError(f"Unsupported model type: {model_type}")
    


def fix_layernorm_naming(model: nn.Module):
    """
    Fix LayerNorm parameter names from checkpoint format (beta/gamma) 
    to PyTorch format (bias/weight).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # Check if parameters are named beta/gamma
            state_dict = dict(module.named_parameters())
            
            if 'beta' in state_dict and 'bias' not in state_dict:
                # Rename beta to bias
                bias_param = module._parameters.pop('beta')
                module.register_parameter('bias', bias_param)
            
            if 'gamma' in state_dict and 'weight' not in state_dict:
                # Rename gamma to weight
                weight_param = module._parameters.pop('gamma')
                module.register_parameter('weight', weight_param)

# ── 6. Main Model Wrapper ───────────────────────────────────
class BertWithTokenCompression(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        comp_config: Optional[CompressionConfig] = None,
        disable_compression: bool = False,
    ):
        super().__init__()
        self.comp_config = comp_config or CompressionConfig()
        self.disable_compression = disable_compression

        # Load config and model
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        fix_layernorm_naming(self.bert)

        # Identify base transformer layers
        model_type = config.model_type
        if model_type == 'bert':
            self.base_model = self.bert.bert
            layers = self.base_model.encoder.layer
        elif model_type == 'distilbert':
            self.base_model = self.bert.distilbert
            layers = self.base_model.transformer.layer
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Inject compression
        self.compression_modules = nn.ModuleList()
        if not disable_compression:
            for i, layer in enumerate(layers):
                new_attn = inject_compression_into_layer(
                    layer=layer,
                    comp_config=self.comp_config,
                    layer_idx=i,
                    model_config=config,
                    model_type=model_type,
                )
                self.compression_modules.append(new_attn)

        # Gradient checkpointing for memory
        # self.bert.gradient_checkpointing_enable()

        log.info(f"Created model with {len(self.compression_modules)} compressed layers "
                 f"(type={model_type})")
        log.info(f"Compression ratios: early={self.comp_config.k_early}, "
                 f"mid={self.comp_config.k_mid}, late={self.comp_config.k_late}")
        self.bert.config.gradient_checkpointing = False
        if hasattr(self.bert, 'gradient_checkpointing_disable'):
            self.bert.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.bert.gradient_checkpointing_disable()


    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.bert(**kwargs)
    
    def get_ortho_loss(self) -> torch.Tensor:
        """Get total orthogonality loss from all compression modules."""
        total_loss = 0.0
        for module in self.compression_modules:
            loss = module.get_ortho_loss()
            if loss is not None:
                total_loss = total_loss + loss
        return torch.tensor(total_loss, device=self.bert.device)
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.bert.save_pretrained(save_directory)
    
    @property
    def device(self):
        return next(self.bert.parameters()).device

# ── 7. Custom Trainer with Orthogonality Loss ───────────────
# ── Custom Trainer with Orthogonality Loss ──────────────
class CompressionTrainer(Trainer):
    def __init__(self, ortho_weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_weight = ortho_weight
        log.info("CompressionTrainer initialized")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # log.info("compute_loss called")
        outputs = model(**inputs)
        # log.info(f"Model forward pass complete, loss: {outputs.loss}")
        task_loss = outputs.loss
        ortho_loss = model.get_ortho_loss()
        # log.info(f"Ortho loss: {ortho_loss}")
        total_loss = task_loss + self.ortho_weight * ortho_loss
        return (total_loss, outputs) if return_outputs else total_loss
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

import evaluate

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr
def get_compute_metrics(task: str):
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"eval_accuracy": accuracy_score(labels, preds)}
    return compute   # works for sst2, mnli, qnli, rte, wnli




def main():
    parser = argparse.ArgumentParser(description="Token‑Compressed Attention for GLUE")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"])
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./compressed_results")
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_progressive", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_type", type=str, default="linformer", choices=["linformer", "leka","leka_v2"])
    parser.add_argument("--num_landmarks", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--use_spherical_norm", action="store_true", default=True)
    parser.add_argument("--kernel", type=str, default="elu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Compression config
    comp_config = CompressionConfig(
            attn_type=args.attn_type,
    num_landmarks=args.num_landmarks,
    feature_dim=args.feature_dim,
    use_spherical_norm=args.use_spherical_norm,
    kernel=args.kernel,

        k_early=args.k_early,
        k_mid=args.k_mid,
        k_late=args.k_late,
        progressive=not args.no_progressive,
        use_gate=not args.no_gate,
        max_seq_len=args.max_len,
    )
    log.info(f"Task: {args.task} | Progressive: {comp_config.progressive}")
    log.info(f"k early/mid/late = {comp_config.k_early}/{comp_config.k_mid}/{comp_config.k_late}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load raw dataset
    raw_datasets = load_glue_local(args.task)
    if 'train' not in raw_datasets:
        raise RuntimeError(f"No 'train' split in {args.task}. Found: {list(raw_datasets.keys())}")
    log.info(f"Raw dataset sizes: train={len(raw_datasets['train'])}, val={len(raw_datasets.get('validation', []))}")

    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    # Tokenization function (with explicit return)
    def preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(
                examples[sent1_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
        else:
            tokenized = tokenizer(
                examples[sent1_key],
                examples[sent2_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
        # Always add labels, even if it fails
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = list(examples["label"])

        return tokenized

    # Apply tokenization
    # dataset = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)
    dataset = raw_datasets.map(
        preprocess, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names  # Remove all original columns
    )

    # --- Sanity checks ---
    if len(dataset["train"]) == 0:
        raise RuntimeError("Training dataset became empty after tokenization!")
    # Check a sample to see if columns exist
    sample = dataset["train"][0]
    required = ["input_ids", "attention_mask", "labels"]
    missing = [col for col in required if col not in sample]
    if missing:
        raise RuntimeError(f"Missing columns after tokenization: {missing}")

    # Set format for PyTorch
    # columns = ["input_ids", "attention_mask", "labels"]
    # if args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
    #     columns.append("token_type_ids")
    # dataset.set_format(type="torch", columns=columns)
    # columns = ["input_ids", "attention_mask", "labels"]
    # # فقط برای BERT اضافه کنید
    # if "bert-base" in args.model and args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
    #     columns.append("token_type_ids")
    is_distilbert = "distilbert" in args.model.lower()
    columns = ["input_ids", "attention_mask", "labels"]
    if not is_distilbert and args.task in (...):
        columns.append("token_type_ids")

    # فقط برای BERT خالص token_type_ids اضافه کن
    if not is_distilbert and args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
        columns.append("token_type_ids")

    log.info(f"Tokenized dataset: train={len(dataset['train'])}, val={len(dataset.get('validation', []))}")

    # Build model
    model = BertWithTokenCompression(args.model, num_labels, comp_config)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")
    log.info("DEBUG: model device is %s", next(model.bert.parameters()).device)

# بعد از model.to(device)
    # model.eval()
    # test_batch = dataset["train"].select(range(8))  # 8 نمونه
    # inputs = tokenizer(test_batch[sent1_key], 
    #                 test_batch[sent2_key] if sent2_key else None,
    #                 truncation=True, padding="max_length", max_length=args.max_len,
    #                 return_tensors="pt")
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # inputs["labels"] = torch.tensor(test_batch["label"]).to(device)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     print("Test forward OK, loss:", outputs.loss.item())
    # exit()
    # Metric for best model (with 'eval_' prefix)
    if args.task == "cola":
        metric_name = "eval_matthews_correlation"
    elif args.task == "stsb":
        metric_name = "eval_pearsonr"
    elif args.task in ("mrpc", "qqp"):
        metric_name = "eval_f1"
    else:
        metric_name = "eval_accuracy"

    print("jjj")
    ortho_weight = comp_config.ortho_reg_weight if not comp_config.use_orthogonal else 0.0
    log.info(f"Train dataset columns: {dataset['train'].column_names}")
    log.info(f"Train dataset features: {dataset['train'].features}")
    log.info(f"First example: {dataset['train'][0]}")
# 🚀 Optimized TrainingArguments + Trainer for DistilBERT / LEKA on GLUE
# Based on official HuggingFace recommendations and empirical tuning for SST-2.

    total_steps = (len(dataset["train"]) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = int(0.06 * total_steps)  # ≈ 600 steps (official DistilBERT recommendation)

    training_args = TrainingArguments(
        # --- Paths ---
        output_dir=os.path.join(args.output_dir, args.task),

        # --- Training schedule ---
        num_train_epochs=5,                      # ↑ 3 → 5 (better convergence)
        per_device_train_batch_size=32,          # ↑ 16 → 32 (larger batch, stable gradients)
        per_device_eval_batch_size=64,           # ↑ 2x train batch
        gradient_accumulation_steps=1,           # ↓ no accumulation needed with batch=32

        # --- Optimizer & scheduling ---
        learning_rate=1e-5,                      # ↓ 2e-5 → 1e-5 (official DistilBERT best)
        weight_decay=0.01,                       # keep
        lr_scheduler_type="linear",              # ← cosine → linear (better for SST-2)
        warmup_steps=warmup_steps,               # explicit warmup steps (~600)
        max_grad_norm=1.0,                       # ↓ 10 → 1.0 (tighter clipping for stability)

        # --- Performance ---
        dataloader_num_workers=4,                # parallel data loading
        dataloader_pin_memory=True,              # faster CPU → GPU transfer
        fp16=False,                              # MPS does NOT support fp16
        gradient_checkpointing=True,             # trade compute for memory → larger batch

        # --- Logging & evaluation ---
        logging_steps=100,                       # ↓ 50 → 100 (reduce overhead)
        eval_strategy="steps",                   # ← "epoch" → "steps" (faster feedback)
        eval_steps=10000,                          # evaluate every 500 steps
        save_strategy="steps",
        save_steps=10000,                          # save checkpoints aligned with eval
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,       # e.g., "eval_accuracy"
        greater_is_better=True,

        # --- Misc ---
        remove_unused_columns=False,
        save_total_limit=2,
        report_to="none",
    )

    # Optional: early stopping (patience = 3 evaluations)
    early_stopping = EarlyStoppingCallback(
        
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )

    trainer = CompressionTrainer(
        ortho_weight=ortho_weight,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=default_data_collator,
        compute_metrics=get_compute_metrics(args.task),
        # callbacks=[early_stopping],              # ← add early stopping
    )
    # log.info("Testing first batch...")
    # try:
    #     test_batch = next(iter(trainer.get_train_dataloader()))
    #     test_batch = {k: v.to(device) for k, v in test_batch.items()}
    #     log.info(f"Batch keys: {test_batch.keys()}")
    #     log.info(f"Batch shapes: {[(k, v.shape) for k, v in test_batch.items()]}")
        
    #     model.train()
    #     with torch.amp.autocast('cpu'):  # or 'cuda' if using GPU
    #         outputs = model(**test_batch)
    #         loss = outputs.loss
    #         log.info(f"Test forward pass OK, loss: {loss.item()}")
    #         loss.backward()
    #         log.info("Test backward pass OK")
    # except Exception as e:
    #     log.error(f"Failed at first batch: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     exit(1)


    log.info("Starting training...")
    trainer.args.metric_for_best_model = None
    trainer.args.load_best_model_at_end = False

    trainer.train()
    eval_result = trainer.evaluate()
    log.info(f"Final eval: {eval_result}")
    trainer.save_model(os.path.join(args.output_dir, args.task))
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.task))
    log.info("Done.")
if __name__ == "__main__":
    main()



















# vocab_layer_norm.weight | UNEXPECTED | 
# vocab_layer_norm.bias   | UNEXPECTED | 
# classifier.weight       | MISSING    | 
# pre_classifier.weight   | MISSING    | 
# pre_classifier.bias     | MISSING    | 
# classifier.bias         | MISSING    | 

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 01:58:17 | INFO    | Created model with 6 compressed layers (type=distilbert)
# 01:58:17 | INFO    | Compression ratios: early=32, mid=16, late=12
# Using MPS (Metal) device
# 01:58:18 | INFO    | DEBUG: model device is mps:0
# [transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
# jjj
# 01:58:18 | INFO    | Train dataset columns: ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
# 01:58:18 | INFO    | Train dataset features: {'input_ids': List(Value('int32')), 'token_type_ids': List(Value('int8')), 'attention_mask': List(Value('int8')), 'labels': Value('int64')}
# 01:58:18 | INFO    | First example: {'input_ids': [101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': 0}
# 01:58:18 | INFO    | CompressionTrainer initialized
# 01:58:18 | INFO    | Starting training...
# {'loss': '1.389', 'grad_norm': '2.229', 'learning_rate': '9.307e-07', 'epoch': '0.02375'}                                                                          
# {'loss': '1.376', 'grad_norm': '4.12', 'learning_rate': '1.88e-06', 'epoch': '0.04751'}                                                                            
# {'loss': '1.366', 'grad_norm': '3.964', 'learning_rate': '2.83e-06', 'epoch': '0.07126'}                                                                           
# {'loss': '1.362', 'grad_norm': '5.464', 'learning_rate': '3.78e-06', 'epoch': '0.09501'}                                                                           
# {'loss': '1.361', 'grad_norm': '3.028', 'learning_rate': '4.729e-06', 'epoch': '0.1188'}                                                                           
# {'loss': '1.367', 'grad_norm': '4.236', 'learning_rate': '5.679e-06', 'epoch': '0.1425'}                                                                           
# {'loss': '1.378', 'grad_norm': '2.384', 'learning_rate': '6.629e-06', 'epoch': '0.1663'}                                                                           
# {'loss': '1.36', 'grad_norm': '3.828', 'learning_rate': '7.578e-06', 'epoch': '0.19'}                                                                              
# {'loss': '1.356', 'grad_norm': '3.347', 'learning_rate': '8.528e-06', 'epoch': '0.2138'}                                                                           
# {'loss': '1.349', 'grad_norm': '4.503', 'learning_rate': '9.478e-06', 'epoch': '0.2375'}                                                                           
# {'loss': '1.375', 'grad_norm': '2.597', 'learning_rate': '1.043e-05', 'epoch': '0.2613'}                                                                           
# {'loss': '1.357', 'grad_norm': '7.647', 'learning_rate': '1.138e-05', 'epoch': '0.285'}                                                                            
# {'loss': '1.347', 'grad_norm': '5.213', 'learning_rate': '1.233e-05', 'epoch': '0.3088'}                                                                           
# {'loss': '1.355', 'grad_norm': '2.564', 'learning_rate': '1.328e-05', 'epoch': '0.3325'}                                                                           
# {'loss': '1.339', 'grad_norm': '3.865', 'learning_rate': '1.423e-05', 'epoch': '0.3563'}                                                                           
# {'loss': '1.308', 'grad_norm': '8.488', 'learning_rate': '1.518e-05', 'epoch': '0.38'}                                                                             
# {'loss': '1.309', 'grad_norm': '9.319', 'learning_rate': '1.613e-05', 'epoch': '0.4038'}                                                                           
# {'loss': '1.324', 'grad_norm': '20.14', 'learning_rate': '1.708e-05', 'epoch': '0.4276'}                                                                           
# {'loss': '1.312', 'grad_norm': '13.66', 'learning_rate': '1.802e-05', 'epoch': '0.4513'}                                                                           
# {'loss': '1.317', 'grad_norm': '4.528', 'learning_rate': '1.897e-05', 'epoch': '0.4751'}                                                                           
# {'loss': '1.314', 'grad_norm': '11.47', 'learning_rate': '1.992e-05', 'epoch': '0.4988'}                                                                           
# {'loss': '1.339', 'grad_norm': '8.952', 'learning_rate': '2e-05', 'epoch': '0.5226'}                                                                               
# {'loss': '1.297', 'grad_norm': '6.602', 'learning_rate': '1.999e-05', 'epoch': '0.5463'}                                                                           
# {'loss': '1.32', 'grad_norm': '4.256', 'learning_rate': '1.999e-05', 'epoch': '0.5701'}                                                                            
# {'loss': '1.287', 'grad_norm': '5.295', 'learning_rate': '1.998e-05', 'epoch': '0.5938'}                                                                           
# {'loss': '1.319', 'grad_norm': '2.317', 'learning_rate': '1.997e-05', 'epoch': '0.6176'}                                                                           
# {'loss': '1.276', 'grad_norm': '3.974', 'learning_rate': '1.995e-05', 'epoch': '0.6413'}                                                                           
# {'loss': '1.293', 'grad_norm': '3.121', 'learning_rate': '1.993e-05', 'epoch': '0.6651'}                                                                           
# {'loss': '1.263', 'grad_norm': '5.827', 'learning_rate': '1.991e-05', 'epoch': '0.6888'}                                                                           
# {'loss': '1.291', 'grad_norm': '7.101', 'learning_rate': '1.989e-05', 'epoch': '0.7126'}                                                                           
# {'loss': '1.263', 'grad_norm': '7.552', 'learning_rate': '1.986e-05', 'epoch': '0.7363'}                                                                           
# {'loss': '1.276', 'grad_norm': '9.575', 'learning_rate': '1.984e-05', 'epoch': '0.7601'}                                                                           
# {'loss': '1.264', 'grad_norm': '7.142', 'learning_rate': '1.981e-05', 'epoch': '0.7838'}                                                                           
# {'loss': '1.281', 'grad_norm': '12.83', 'learning_rate': '1.977e-05', 'epoch': '0.8076'}                                                                           
# {'loss': '1.285', 'grad_norm': '6.523', 'learning_rate': '1.973e-05', 'epoch': '0.8314'}                                                                           
# {'loss': '1.286', 'grad_norm': '8.291', 'learning_rate': '1.97e-05', 'epoch': '0.8551'}                                                                            
# {'loss': '1.281', 'grad_norm': '18.49', 'learning_rate': '1.965e-05', 'epoch': '0.8789'}                                                                           
# {'loss': '1.262', 'grad_norm': '23.58', 'learning_rate': '1.961e-05', 'epoch': '0.9026'}                                                                           
# {'loss': '1.269', 'grad_norm': '3.058', 'learning_rate': '1.956e-05', 'epoch': '0.9264'}                                                                           
# {'loss': '1.23', 'grad_norm': '13.25', 'learning_rate': '1.951e-05', 'epoch': '0.9501'}                                                                            
# {'loss': '1.265', 'grad_norm': '13.11', 'learning_rate': '1.946e-05', 'epoch': '0.9739'}                                                                           
# {'loss': '1.236', 'grad_norm': '9.675', 'learning_rate': '1.94e-05', 'epoch': '0.9976'}                                                                            
# {'eval_runtime': '8.155', 'eval_samples_per_second': '106.9', 'eval_steps_per_second': '3.434', 'epoch': '1'}                                                      
#  20%|███████████████████████▊                                                                                               | 2105/10525 [28:53<2:02:11,  1.15it/sTraceback (most recent call last):                                                                                                                                  
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 3124, in _determine_best_metric
#     metric_value = metrics[metric_to_check]
#                    ~~~~~~~^^^^^^^^^^^^^^^^^
# KeyError: 'eval_accuracy'

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/Users/tahamajs/Documents/uni/Project/Codes/LEKAAttention.py", line 974, in <module>
#     main()
#     ~~~~^^
#   File "/Users/tahamajs/Documents/uni/Project/Codes/LEKAAttention.py", line 967, in main
#     trainer.train()
#     ~~~~~~~~~~~~~^^
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1427, in train
#     return inner_training_loop(
#         args=args,
#     ...<2 lines>...
#         ignore_keys_for_eval=ignore_keys_for_eval,
#     )
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1509, in _inner_training_loop
#     self._run_epoch(
#     ~~~~~~~~~~~~~~~^
#         model=model,
#         ^^^^^^^^^^^^
#     ...<9 lines>...
#         steps_trained_in_current_epoch=steps_trained_in_current_epoch,
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     )
#     ^
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1810, in _run_epoch
#     self._maybe_log_save_evaluate(
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
#         self._tr_loss,
#         ^^^^^^^^^^^^^^
#     ...<6 lines>...
#         learning_rate=learning_rate,
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     )
#     ^
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 2085, in _maybe_log_save_evaluate
#     is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 3126, in _determine_best_metric
#     raise KeyError(
#     ...<2 lines>...
#     ) from exc
# KeyError: "The `metric_for_best_model` training argument is set to 'eval_accuracy', which is not found in the evaluation metrics. The available evaluation metrics are: ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']. Consider changing the `metric_for_best_model` via the TrainingArguments."
#  20%|███████████████████████▊                                                                                               | 2105/10525 [28:55<1:55:40,  1.21it/s]
# (.venv) ➜  Codes git:(main) ✗ 








# (.venv) ➜  Codes git:(main) ✗ python LEKAAttention.py --attn_type leka_v2 --num_landmarks 16 --feature_dim 128 \
#   --max_len 64 --batch_size 32 --epochs 5 --lr 1e-5 --no_gate \
#   --model distilbert-base-uncased --task sst2
# 05:08:18 | INFO    | Task: sst2 | Progressive: True
# 05:08:18 | INFO    | k early/mid/late = 32/16/12
# 05:08:19 | INFO    | Raw dataset sizes: train=67349, val=872
# 05:08:19 | INFO    | Tokenized dataset: train=67349, val=872
# Loading weights: 100%|████████████████████████████████████| 100/100 [00:00<00:00, 5830.60it/s]
# [transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
# Key                     | Status     | 
# ------------------------+------------+-
# vocab_layer_norm.weight | UNEXPECTED | 
# vocab_layer_norm.bias   | UNEXPECTED | 
# vocab_projector.bias    | UNEXPECTED | 
# vocab_transform.weight  | UNEXPECTED | 
# vocab_transform.bias    | UNEXPECTED | 
# classifier.bias         | MISSING    | 
# pre_classifier.weight   | MISSING    | 
# pre_classifier.bias     | MISSING    | 
# classifier.weight       | MISSING    | 

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 05:08:19 | INFO    | Created model with 6 compressed layers (type=distilbert)
# 05:08:19 | INFO    | Compression ratios: early=32, mid=16, late=12
# Using MPS (Metal) device
# 05:08:20 | INFO    | DEBUG: model device is mps:0
# jjj
# 05:08:20 | INFO    | Train dataset columns: ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
# 05:08:20 | INFO    | Train dataset features: {'input_ids': List(Value('int32')), 'token_type_ids': List(Value('int8')), 'attention_mask': List(Value('int8')), 'labels': Value('int64')}
# 05:08:20 | INFO    | First example: {'input_ids': [101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': 0}
# 05:08:20 | INFO    | CompressionTrainer initialized
# 05:08:20 | INFO    | Starting training...
# {'loss': '0.6908', 'grad_norm': '1.548', 'learning_rate': '1.569e-06', 'epoch': '0.04751'}    
# {'loss': '0.6805', 'grad_norm': '2.738', 'learning_rate': '3.154e-06', 'epoch': '0.09501'}    
# {'loss': '0.6637', 'grad_norm': '3.305', 'learning_rate': '4.739e-06', 'epoch': '0.1425'}     
# {'loss': '0.638', 'grad_norm': '10.38', 'learning_rate': '6.323e-06', 'epoch': '0.19'}        
# {'loss': '0.5742', 'grad_norm': '14.15', 'learning_rate': '7.908e-06', 'epoch': '0.2375'}     
# {'loss': '0.5552', 'grad_norm': '11.11', 'learning_rate': '9.493e-06', 'epoch': '0.285'}      
# {'loss': '0.49', 'grad_norm': '33.62', 'learning_rate': '9.931e-06', 'epoch': '0.3325'}       
# {'loss': '0.457', 'grad_norm': '26.67', 'learning_rate': '9.83e-06', 'epoch': '0.38'}         
# {'loss': '0.4554', 'grad_norm': '14.28', 'learning_rate': '9.729e-06', 'epoch': '0.4276'}     
# {'loss': '0.436', 'grad_norm': '27.66', 'learning_rate': '9.628e-06', 'epoch': '0.4751'}      
# {'loss': '0.429', 'grad_norm': '9.52', 'learning_rate': '9.527e-06', 'epoch': '0.5226'}       
# {'loss': '0.4004', 'grad_norm': '8.422', 'learning_rate': '9.426e-06', 'epoch': '0.5701'}     
# {'loss': '0.3856', 'grad_norm': '7.425', 'learning_rate': '9.325e-06', 'epoch': '0.6176'}     
# {'loss': '0.3715', 'grad_norm': '16.23', 'learning_rate': '9.224e-06', 'epoch': '0.6651'}     
# {'loss': '0.3593', 'grad_norm': '12.69', 'learning_rate': '9.123e-06', 'epoch': '0.7126'}     
# {'loss': '0.3817', 'grad_norm': '11.79', 'learning_rate': '9.022e-06', 'epoch': '0.7601'}     
# {'loss': '0.3516', 'grad_norm': '20.56', 'learning_rate': '8.921e-06', 'epoch': '0.8076'}     
# {'loss': '0.3656', 'grad_norm': '8.041', 'learning_rate': '8.819e-06', 'epoch': '0.8551'}     
# {'loss': '0.3496', 'grad_norm': '11.13', 'learning_rate': '8.718e-06', 'epoch': '0.9026'}     
# {'loss': '0.3467', 'grad_norm': '18.02', 'learning_rate': '8.617e-06', 'epoch': '0.9501'}     
# {'loss': '0.3419', 'grad_norm': '14.04', 'learning_rate': '8.516e-06', 'epoch': '0.9976'}     
# {'loss': '0.3223', 'grad_norm': '5.301', 'learning_rate': '8.415e-06', 'epoch': '1.045'}      
# {'loss': '0.3185', 'grad_norm': '18.49', 'learning_rate': '8.314e-06', 'epoch': '1.093'}      
# {'loss': '0.315', 'grad_norm': '14.94', 'learning_rate': '8.213e-06', 'epoch': '1.14'}        
# {'loss': '0.3075', 'grad_norm': '10.32', 'learning_rate': '8.112e-06', 'epoch': '1.188'}      
# {'loss': '0.2933', 'grad_norm': '3.704', 'learning_rate': '8.011e-06', 'epoch': '1.235'}      
# {'loss': '0.3056', 'grad_norm': '14.1', 'learning_rate': '7.91e-06', 'epoch': '1.283'}        
# {'loss': '0.2821', 'grad_norm': '7.689', 'learning_rate': '7.809e-06', 'epoch': '1.33'}       
# {'loss': '0.2734', 'grad_norm': '5.206', 'learning_rate': '7.708e-06', 'epoch': '1.378'}      
# {'loss': '0.2803', 'grad_norm': '12.43', 'learning_rate': '7.607e-06', 'epoch': '1.425'}      
# {'loss': '0.2886', 'grad_norm': '9.118', 'learning_rate': '7.506e-06', 'epoch': '1.473'}      
# {'loss': '0.2794', 'grad_norm': '11.06', 'learning_rate': '7.404e-06', 'epoch': '1.52'}       
# {'loss': '0.2827', 'grad_norm': '3.273', 'learning_rate': '7.303e-06', 'epoch': '1.568'}      
#  32%|█████████████████████████████████████▏                                                                               | 3348/10525 [1:12:09<2:37:57,  1.32s/it]^CTraceback (most recent call last):
#   File "/Users/tahamajs/Documents/uni/Project/Codes/LEKAAttention.py", line 1116, in <module>
#     main()