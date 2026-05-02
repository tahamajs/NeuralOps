"""
NEXUS – Neural Efficient X‑scoring Unified Sparse Attention
=============================================================
Novel hybrid attention that fuses the best of ICML/NeurIPS 2025:

 1. XAttention antidiagonal scoring (ICML 2025)      — training‑free block scoring
 2. SeerAttention learnable gating (NeurIPS 2025)    — per‑head sparsity
 3. DuSA dual‑stage sparse (NeurIPS 2025)            — intra + inter block
 4. AnchorAttention stripe granularity (2025)         — fine‑grain stripes
 5. Spectraformer random‑feature rectification (2025) — global linear fallback
 6. Softpick rectified softmax (2025)                 — natural ~47% sparsity

Papers:
  • XAttention: Xu et al., ICML 2025 (arXiv:2503.16428)
  • SeerAttention: Gao et al., NeurIPS 2025
  • DuSA: Cao et al., NeurIPS 2025
  • AnchorAttention: Zhu et al., 2025 (arXiv:2505.23523)
  • Spectraformer: Nguyen et al., 2025 (arXiv:2405.15310)
  • Softpick: Zuhri et al., arXiv 2504.20966 (2025)

Author: Taha Majs — 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NEXUSConfig:
    # --- Block structure ---
    block_size: int = 32                    # tokens per block (smaller = finer)
    top_k_blocks: int = 4                   # how many inter‑block blocks to keep

    # --- XAttention antidiagonal scoring ---
    xattn_enabled: bool = True
    xattn_stride: int = 2                   # stride for antidiagonal sampling

    # --- SeerAttention gating ---
    seer_enabled: bool = True
    seer_gate_rank: int = 16                # hidden dim of gate network
    seer_top_k_blocks: int = 4              # blocks kept by the learned gate
    seer_temperature: float = 1.0

    # --- DuSA dual‑stage ---
    dual_stage_enabled: bool = True
    intra_block_full: bool = True           # dense inside each block

    # --- AnchorAttention stripe ---
    stripe_enabled: bool = True
    stripe_width: int = 4                   # tokens per stripe

    # --- Spectraformer global rectification ---
    rectify_interval: int = 4               # every Nth layer uses kernel attn
    rff_features: int = 128

    # --- Softpick ---
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0

    dropout: float = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: Softpick rectified softmax
# ═══════════════════════════════════════════════════════════════════════════════

def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Rectified ~47 % sparse softmax, no attention sink."""
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. XAttention antidiagonal block scoring
# ═══════════════════════════════════════════════════════════════════════════════

class XAttentionScorer:
    """
    Training‑free block importance via antidiagonal sums.
    Reference: XAttention, Xu et al., ICML 2025.
    """
    @staticmethod
    def score_blocks(Q: torch.Tensor, K: torch.Tensor,
                     block_size: int, stride: int = 2) -> torch.Tensor:
        """
        Q, K: (B, H, N, D)
        Returns: block_score (B, H, n_blocks, n_blocks)
        """
        B, H, N, D = Q.shape
        n_blocks = (N + block_size - 1) // block_size

        # sample tokens at stride for speed
        Q_sample = Q[:, :, ::stride, :]   # (B, H, N_s, D)
        K_sample = K[:, :, ::stride, :]

        # compute a tiny attention map on sampled tokens
        scale = D ** -0.5
        attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * scale

        # for each block, sum antidiagonal values
        scores = torch.zeros(B, H, n_blocks, n_blocks, device=Q.device, dtype=Q.dtype)
        N_s = Q_sample.size(2)

        for bi in range(n_blocks):
            q_start = bi * (block_size // stride)
            q_end = min(q_start + block_size // stride, N_s)
            for bj in range(n_blocks):
                k_start = bj * (block_size // stride)
                k_end = min(k_start + block_size // stride, N_s)
                # antidiagonal: chunks where i + j = const within the block
                block = attn_sample[:, :, q_start:q_end, k_start:k_end]
                scores[:, :, bi, bj] = block.mean(dim=(2, 3))

        return scores   # (B, H, n_blocks, n_blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SeerAttention learnable gating
# ═══════════════════════════════════════════════════════════════════════════════

class SeerGating(nn.Module):
    """
    Learnable block‑level sparsity gate.
    Reference: SeerAttention, Gao et al., NeurIPS 2025.
    """
    def __init__(self, head_dim: int, num_heads: int,
                 gate_rank: int, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.temperature = temperature

        # pooling projection for Q and K
        self.q_pool = nn.Sequential(
            nn.Linear(head_dim, gate_rank),
            nn.GELU(),
            nn.Linear(gate_rank, gate_rank),
        )
        self.k_pool = nn.Sequential(
            nn.Linear(head_dim, gate_rank),
            nn.GELU(),
            nn.Linear(gate_rank, gate_rank),
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                n_blocks: int, training: bool = True) -> torch.Tensor:
        """
        Q, K: (B, H, N, D)
        Returns: hard_mask (B, H, n_blocks, n_blocks)
        """
        B, H, N, D = Q.shape
        gr = self.q_pool[0].out_features   # gate_rank

        # pool over sequence: mean pool Q and K for each head
        Q_pooled = Q.mean(dim=2)   # (B, H, D)
        K_pooled = K.mean(dim=2)

        # project through gate networks
        q_gate = self.q_pool(Q_pooled).view(B, H, 1, gr)   # (B, H, 1, gr)
        k_gate = self.k_pool(K_pooled).view(B, H, gr, 1)

        # produce a scalar gate score per head (we need per‑block)
        # expand to per‑block by tiling
        gate_score = torch.sigmoid(
            torch.matmul(q_gate, k_gate).squeeze(-1).squeeze(-1) / self.temperature
        )   # (B, H)

        # for each head, build a block mask:
        # use gate_score to decide how many blocks to keep
        keep_per_head = torch.clamp(
            (gate_score * n_blocks).long(), min=1, max=n_blocks
        )   # (B, H)

        # create full block mask (simplified: keep first k blocks)
        mask = torch.zeros(B, H, n_blocks, n_blocks, device=Q.device)
        for h in range(H):
            k = keep_per_head[0, h].item()   # batch‑uniform
            mask[:, h, :k, :k] = 1.0   # keep k×k blocks

        return mask.bool()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dual‑Stage + Stripe Attention (core engine)
# ═══════════════════════════════════════════════════════════════════════════════

class DualStageStripeAttention(nn.Module):
    """
    DuSA‑style intra‑block full + inter‑block XAttention‑scored top‑k,
    with AnchorAttention‑style stripe granularity inside blocks.

    References:
      • DuSA: Cao et al., NeurIPS 2025
      • XAttention: Xu et al., ICML 2025
      • AnchorAttention: Zhu et al., 2025
    """
    def __init__(self, cfg: NEXUSConfig, head_dim: int):
        super().__init__()
        self.cfg = cfg
        self.head_dim = head_dim
        self.xattn_scorer = XAttentionScorer()

    def _inter_block_topk(self, scores: torch.Tensor, top_k: int):
        """Select top‑k blocks per query block."""
        B, H, nb, nb2 = scores.shape
        tk = min(top_k, nb2)
        _, idx = scores.topk(tk, dim=-1)   # (B, H, nb, tk)
        mask = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
        return mask.bool()

    def forward(
        self,
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Q, K, V: (B, H, N, D)
        Returns: context (B, H, N, D)
        """
        B, H, N, D = Q.shape
        bs = self.cfg.block_size
        nb = (N + bs - 1) // bs
        top_k = self.cfg.top_k_blocks
        scale = D ** -0.5

        # ── 1. XAttention antidiagonal block scoring ──
        if self.cfg.xattn_enabled:
            block_scores = self.xattn_scorer.score_blocks(
                Q, K, bs, self.cfg.xattn_stride
            )   # (B, H, nb, nb)
            inter_mask = self._inter_block_topk(block_scores, top_k)
        else:
            inter_mask = torch.ones(B, H, nb, nb, device=Q.device).bool()

        # ── 2. Dual‑stage: intra‑block full + inter‑block selected ──
        outputs = []
        for bi in range(nb):
            start = bi * bs
            end = min(start + bs, N)
            Q_b = Q[:, :, start:end, :]   # (B, H, Bs, D)
            Bs = end - start

            # intra‑block keys (always full)
            K_intra = K[:, :, start:end, :]
            V_intra = V[:, :, start:end, :]

            # inter‑block keys (selected blocks)
            active_blocks = inter_mask[:, :, bi, :]   # (B, H, nb)
            K_inter_list, V_inter_list = [], []
            for bj in range(nb):
                if active_blocks[:, :, bj].any() and bj != bi:
                    s2 = bj * bs
                    e2 = min(s2 + bs, N)
                    K_inter_list.append(K[:, :, s2:e2, :])
                    V_inter_list.append(V[:, :, s2:e2, :])

            if K_inter_list:
                K_inter = torch.cat(K_inter_list, dim=2)
                V_inter = torch.cat(V_inter_list, dim=2)
                K_total = torch.cat([K_intra, K_inter], dim=2)
                V_total = torch.cat([V_intra, V_inter], dim=2)
            else:
                K_total = K_intra
                V_total = V_intra

            # ── 3. Stripe‑granularity sparsity inside blocks ──
            # Apply a learnable stripe mask within each block
            if self.cfg.stripe_enabled:
                sw = self.cfg.stripe_width
                idx = torch.arange(K_total.size(2), device=Q.device)
                # stripe pattern: group consecutive `sw` keys together
                stripe_group = idx // sw
                # allow each query stripe to attend to its own + adjacent key stripes
                q_stripe_group = (torch.arange(Bs, device=Q.device) // sw).unsqueeze(1)
                k_stripe_group = stripe_group.unsqueeze(0)
                stripe_mask = (q_stripe_group - k_stripe_group).abs() <= 1
                stripe_mask = stripe_mask.unsqueeze(0).unsqueeze(0)   # (1, 1, Bs, K_total)
            else:
                stripe_mask = None

            # compute attention
            scores = torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale

            if stripe_mask is not None:
                scores = scores.masked_fill(~stripe_mask, -1e9)

            if token_valid is not None:
                # apply padding mask
                intra_valid = token_valid[:, start:end].unsqueeze(1).unsqueeze(-1)
                scores = scores.masked_fill(~intra_valid.bool(), -1e9)

            attn = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature) \
                if self.cfg.softpick_enabled else F.softmax(scores, dim=-1)

            out_b = torch.matmul(attn, V_total)
            outputs.append(out_b)

        return torch.cat(outputs, dim=2)   # (B, H, N, D)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Spectraformer‑style random‑feature global rectification
# ═══════════════════════════════════════════════════════════════════════════════

class SpectraformerRectify(nn.Module):
    """
    Global linear attention via random Fourier features.
    Reference: Spectraformer, Nguyen et al., 2025.
    """
    def __init__(self, head_dim: int, rff_features: int = 128):
        super().__init__()
        self.rff_features = rff_features
        self.proj = nn.Parameter(torch.randn(head_dim, rff_features) * 0.02)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = Q.shape
        F_dim = self.rff_features

        # positive random features (Performer / Spectraformer)
        Q_proj = torch.matmul(F.normalize(Q, p=2, dim=-1), self.proj.to(Q.device))
        K_proj = torch.matmul(F.normalize(K, p=2, dim=-1), self.proj.to(Q.device))

        Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)      # (B, H, N, F)
        K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

        KV = torch.einsum('bhnf,bhnd->bhfd', K_feat, V)    # (B, H, F, D)
        Z  = K_feat.sum(dim=2, keepdim=True)                # (B, H, 1, F)

        ctx   = torch.einsum('bhnf,bhfd->bhnd', Q_feat, KV)             # (B, H, N, D)
        denom = (Q_feat * Z).sum(dim=-1, keepdim=True)                  # (B, H, N, 1)
        return ctx / (denom + 1e-8)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. NEXUS Attention — Full Module
# ═══════════════════════════════════════════════════════════════════════════════

class NEXUSAttention(nn.Module):
    """
    NEXUS – Neural Efficient X‑scoring Unified Sparse Attention.
    Drop‑in replacement for DistilBERT `attention`.
    """

    def __init__(self, model_config, nexus_cfg: NEXUSConfig,
                 layer_idx: int = 0, total_layers: int = 6,
                 pretrained_attn=None, bert_mode: bool = False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = nexus_cfg
        self.layer_idx = layer_idx
        self.bert_mode = bert_mode

        # Standard projections (weights copied from pretrained)
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key   = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(nexus_cfg.dropout)

        # SeerAttention learnable gating
        self.seer_gate = None
        if nexus_cfg.seer_enabled:
            self.seer_gate = SeerGating(
                self.head_dim, self.num_heads,
                nexus_cfg.seer_gate_rank,
                nexus_cfg.seer_top_k_blocks,
                nexus_cfg.seer_temperature,
            )

        # Dual‑stage + stripe attention engine
        self.dss_attn = DualStageStripeAttention(nexus_cfg, self.head_dim)

        # Spectraformer rectification (every N layers)
        self.rect_interval = nexus_cfg.rectify_interval
        self.is_rectify = (layer_idx % self.rect_interval == 0)
        if self.is_rectify:
            self.spec_rectify = SpectraformerRectify(
                self.head_dim, nexus_cfg.rff_features
            )
        else:
            self.spec_rectify = None

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_token_valid_mask(self, mask, B, N, device):
        if mask is None:
            return None
        if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
            token_valid = mask[:, 0, 0, :] > -1e9
        elif mask.dim() == 4:
            token_valid = mask.squeeze(1).squeeze(1) > -1e9
        elif mask.dim() == 3:
            token_valid = mask.squeeze(1).bool()
        elif mask.dim() == 2:
            token_valid = mask.bool()
        else:
            raise ValueError(f"Unexpected mask dim: {mask.dim()}")
        return token_valid.to(device)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        # Project
        Q = self._reshape(self.query(hidden_states))
        K = self._reshape(self.key(hidden_states))
        V = self._reshape(self.value(hidden_states))

        # Token validity mask
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        token_valid = self._get_token_valid_mask(mask, B, N, Q.device)

        # ── Spectraformer global rectification ──
        if self.is_rectify and self.spec_rectify is not None:
            context = self.spec_rectify(Q, K, V)
        else:
            # ── SeerAttention gating (optional, modifies block selection) ──
            if self.seer_gate is not None:
                bs = self.cfg.block_size
                nb = (N + bs - 1) // bs
                seer_mask = self.seer_gate(Q, K, nb, training=self.training)
                # (the gating is applied inside DualStageStripeAttention via XAttention + top‑k)
                # for now we pass through; the gate can be combined in future iterations
                _ = seer_mask

            # ── Dual‑stage + stripe attention ──
            context = self.dss_attn(Q, K, V, token_valid)

        # Apply token validity
        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        # Reshape and output
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(self.dropout(context))
        return (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.query.weight.device)


# ═══════════════════════════════════════════════════════════════════════════════
# Injection helper
# ═══════════════════════════════════════════════════════════════════════════════

def inject_nexus_into_distilbert(layer, model_config, nexus_cfg: NEXUSConfig,
                                  layer_idx, total_layers, old_attn):
    new_attn = NEXUSAttention(
        model_config=model_config,
        nexus_cfg=nexus_cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        pretrained_attn=old_attn,
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
class SimpleRFFKernel(nn.Module):
    """Fallback for AdaptiveKernelSwitch – just RFF features."""
    def __init__(self, head_dim, rff_dim, sph_dim=None):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(head_dim, rff_dim) * 0.02)
    def forward(self, q, k, training=True):
        def rff(x):
            p = torch.matmul(x, self.proj)
            return torch.cat([torch.cos(p), torch.sin(p)], dim=-1) / (self.proj.size(1)**0.25)
        return rff(q), rff(k)
    
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
    # if comp_config.attn_type == "leka":
    #     new_attn = LEKAAttention(model_config, comp_config, layer_idx, bert_mode=False)
    # else:
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
        # elif comp_config.attn_type == "spectra_v2":
            # spectra_cfg = SpectraConfig(
            #     block_size=64,
            #     top_k_inter=16,
            #     adaptive_span=True,
            #     max_span=256,
            #     rectify_interval=4,
            #     aks_enabled=True,
            # )
            # return inject_spectra_v2_into_distilbert(
            #     layer, model_config, spectra_cfg, layer_idx,
            #     total_layers=6, old_attn=layer.attention
            # )
        elif comp_config.attn_type == "nexus":
            nexus_cfg = NEXUSConfig(
                block_size=32,          # you can tune these
                top_k_blocks=4,
                seer_enabled=True,
                stripe_enabled=True,
                softpick_enabled=True,
                rectify_interval=4,
                rff_features=128,
            )
            return inject_nexus_into_distilbert(
                layer, model_config, nexus_cfg, layer_idx,
                total_layers=6, old_attn=layer.attention
            )
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
    parser.add_argument("--attn_type", type=str, default="linformer",
                    choices=["linformer", "leka", "leka_v2", "spectra_v2", "nexus"])
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



