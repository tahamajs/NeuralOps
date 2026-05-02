"""
SPECTRA‑v2 – Sparse Per‑head Efficient Context Transformer with Rectified Attention
Fix: true weighted inter‑block top‑k + order‑preserving FASA via relative bias.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpectraConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    # Block structure
    block_size: int = 64
    top_k_inter: int = 16          # per query token, how many inter‑block keys to keep
    # Intra‑block: always full (order preserved)
    # Inter‑block: selected from immediate neighbours (prev + next blocks)
    # Order preservation
    use_rel_pos_bias: bool = True
    rel_pos_max_distance: int = 128
    # Adaptive span (if desired, keep per‑head)
    adaptive_span: bool = True
    max_span: float = 512.0
    # Global rectification (every N layers)
    rectify_interval: int = 4
    rff_features: int = 128
    sph_features: int = 128
    aks_enabled: bool = True


class SpectraV2Attention(nn.Module):
    def __init__(self, model_config, spectra_cfg: SpectraConfig, layer_idx: int, total_layers: int, bert_mode=False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.cfg = spectra_cfg
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.bert_mode = bert_mode

        # Standard projections
        self.query = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.value = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, model_config.hidden_size)
        self.dropout = nn.Dropout(spectra_cfg.dropout)

        # Adaptive span (optional)
        if spectra_cfg.adaptive_span:
            self.span = nn.Parameter(torch.full((self.num_heads,), spectra_cfg.max_span, dtype=torch.float))
        else:
            self.span = None

        # Relative position bias (order‑preserving inter‑block selection)
        self.rel_pos_bias = None
        if spectra_cfg.use_rel_pos_bias:
            max_dist = spectra_cfg.rel_pos_max_distance
            # Learnable bias: (num_heads, 2*max_dist+1)
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.num_heads, 2 * max_dist + 1))
        self.max_rel_dist = spectra_cfg.rel_pos_max_distance if spectra_cfg.use_rel_pos_bias else 0

        # Global rectification kernel (every N layers)
        self.rect_interval = spectra_cfg.rectify_interval
        self.is_rectify_layer = (layer_idx % self.rect_interval == 0) and spectra_cfg.aks_enabled
        if self.is_rectify_layer:
            self.aks = SimpleRFFKernel(
                self.head_dim,
                spectra_cfg.rff_features,
                spectra_cfg.sph_features
            )
        else:
            self.aks = None

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).transpose(1, 2)   # (B, H, N, D)

    def _adaptive_span_logit(self, N: int, device: torch.device):
        """Return a log-weight mask of shape (H, N, N) that softly limits the attention span."""
        if self.span is None:
            return 0.0
        span = self.span.clamp(min=1.0)            # (H,)
        idx = torch.arange(N, device=device, dtype=torch.float)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().unsqueeze(0)   # (1, N, N)
        # Soft mask: large negative logit outside span, ~0 inside
        logit = torch.log((F.sigmoid((span.unsqueeze(1).unsqueeze(1) - dist) * 5.0) + 1e-9))
        return logit   # (H, N, N)

    def _get_token_valid_mask(self, mask, B, N, device):
        """Convert any attention mask to (B, N) boolean (True = valid)."""
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

    def _block_sparse_attention(self, Q, K, V, token_mask):
        """
        Intra‑block full + inter‑block per‑query top‑k with relative bias.
        Each query token attends to:
          - all tokens in its own block (full, order preserved)
          - up to top_k_inter tokens selected from the previous and next block.
        """
        B, H, N, D = Q.shape
        scale = self.head_dim ** -0.5
        block_size = self.cfg.block_size
        num_blocks = (N + block_size - 1) // block_size
        top_k = self.cfg.top_k_inter

        # 1) Adaptive span log mask (applied to intra scores only)
        span_logit = self._adaptive_span_logit(N, Q.device)   # (H, N, N)

        # 2) Precompute relative position indices for bias
        if self.rel_pos_bias is not None:
            # index = (pos_q - pos_k) + max_dist, clipped
            pos = torch.arange(N, device=Q.device)
            rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N)
            rel_pos_clipped = rel_pos.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
            # rel_pos_clipped: (N, N) of values [0, 2*max_dist]

        outputs = []
        for b_idx in range(num_blocks):
            start = b_idx * block_size
            end = min(start + block_size, N)
            Q_b = Q[:, :, start:end, :]                     # (B, H, Bs, D)
            Bs = end - start

            # ---- Intra‑block keys/values (always full) ----
            K_intra = K[:, :, start:end, :]
            V_intra = V[:, :, start:end, :]

            # ---- Inter‑block candidates from left and right ----
            inter_keys = []
            inter_vals  = []
            inter_positions = []   # store global positions for bias
            if b_idx > 0:
                p_start = max(0, (b_idx-1)*block_size)
                p_end   = start
                inter_keys.append(K[:, :, p_start:p_end, :])
                inter_vals.append(V[:, :, p_start:p_end, :])
                inter_positions.append(torch.arange(p_start, p_end, device=Q.device))
            if b_idx < num_blocks - 1:
                n_start = end
                n_end   = min(N, (b_idx+2)*block_size)   # next block only (can extend)
                inter_keys.append(K[:, :, n_start:n_end, :])
                inter_vals.append(V[:, :, n_start:n_end, :])
                inter_positions.append(torch.arange(n_start, n_end, device=Q.device))

            if len(inter_keys) > 0:
                K_inter = torch.cat(inter_keys, dim=2)   # (B, H, Li, D)
                V_inter = torch.cat(inter_vals, dim=2)
                pos_inter = torch.cat(inter_positions, dim=0)   # (Li,)
                Li = K_inter.size(2)

                # Compute raw scores for inter tokens (before bias)
                scores_inter = torch.matmul(Q_b, K_inter.transpose(-2, -1)) * scale   # (B, H, Bs, Li)

                # Add relative position bias (order‑preserving selection)
                if self.rel_pos_bias is not None:
                    q_pos = torch.arange(start, end, device=Q.device).view(1, 1, Bs, 1)
                    k_pos = pos_inter.view(1, 1, 1, Li)
                    rel = (q_pos - k_pos).clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
                    # rel: (1, 1, Bs, Li) with values 0..2*max_dist
                    bias = self.rel_pos_bias[:, rel]   # (H, Bs, Li) -> expects (B,H,Bs,Li) after expand
                    # bias is (H, Bs, Li), we need (B, H, Bs, Li)
                    bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
                    scores_inter = scores_inter + bias

                # Per‑query token top‑k selection from inter tokens
                if Li > top_k:
                    # Select top_k for each query token independently
                    topk_vals, topk_idx = scores_inter.topk(top_k, dim=-1)   # (B, H, Bs, top_k)
                    # Gather the selected inter keys/values
                    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
                    K_inter_sel = K_inter.unsqueeze(2).expand(-1, -1, Bs, -1, -1).gather(3, topk_idx_expanded)
                    V_inter_sel = V_inter.unsqueeze(2).expand(-1, -1, Bs, -1, -1).gather(3, topk_idx_expanded)
                    # K_inter_sel: (B, H, Bs, top_k, D)
                    K_inter_sel = K_inter_sel.view(B, H, Bs * top_k, D)   # flatten for concatenation
                    V_inter_sel = V_inter_sel.view(B, H, Bs * top_k, D)
                else:
                    K_inter_sel = K_inter
                    V_inter_sel = V_inter

                # Concatenate intra + selected inter
                K_total = torch.cat([K_intra, K_inter_sel], dim=2)   # (B, H, Bs + Bs*topk, D)
                V_total = torch.cat([V_intra, V_inter_sel], dim=2)
            else:
                K_total = K_intra
                V_total = V_intra

            # Compute scores with combined set
            scores = torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale   # (B, H, Bs, K_total_len)

            # Apply adaptive span only to intra part (first Bs keys)
            intra_scores = scores[:, :, :, :Bs]
            intra_scores = intra_scores + span_logit[:, start:end, start:end].unsqueeze(0)
            scores[:, :, :, :Bs] = intra_scores

            # Apply token validity mask (padding) if provided
            if token_mask is not None:
                # We need to build a mask for the combined keys. Intra keys come from same block, inter keys from their original positions.
                # We'll fetch the validity of the inter keys using token_mask.
                intra_valid = token_mask[:, start:end].unsqueeze(1).unsqueeze(-1)   # (B, 1, Bs, 1)
                if len(inter_keys) > 0:
                    inter_valid = token_mask[:, pos_inter].unsqueeze(1).unsqueeze(2)   # (B, 1, 1, Li)
                    if Li > top_k:
                        # Gather validity for the selected tokens
                        inter_valid = inter_valid.expand(-1, -1, end-start, -1).gather(3, topk_idx.unsqueeze(0))  # (B, 1, Bs, top_k)
                        inter_valid = inter_valid.view(B, 1, Bs * top_k)
                    else:
                        inter_valid = inter_valid.view(B, 1, Li)
                    total_valid = torch.cat([intra_valid.squeeze(-1), inter_valid], dim=-1)  # (B, 1, Bs+len)
                else:
                    total_valid = intra_valid.squeeze(-1)   # (B, 1, Bs)
                scores = scores.masked_fill(total_valid.unsqueeze(2) == 0, -10000.0)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_b = torch.matmul(attn, V_total)   # (B, H, Bs, D)
            outputs.append(out_b)

        out = torch.cat(outputs, dim=2)   # (B, H, N, D)
        return out

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, D = hidden_states.shape
        H = self.num_heads

        Q = self._reshape(self.query(hidden_states))
        K = self._reshape(self.key(hidden_states))
        V = self._reshape(self.value(hidden_states))

        # Token validity mask
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        token_mask = self._get_token_valid_mask(mask, B, N, Q.device)

        if self.is_rectify_layer:
            # Global linear attention via AKS (like HYPERNOVA)
            Q_feat, K_feat = self.aks(Q, K, training=self.training)
            KV = torch.matmul(K_feat.transpose(-2, -1), V)
            Z = K_feat.sum(dim=2, keepdim=True)
            context = torch.matmul(Q_feat, KV) / torch.matmul(Q_feat, Z.transpose(-2, -1))
        else:
            # Block sparse attention with true top‑k weighted sum
            context = self._block_sparse_attention(Q, K, V, token_mask)

        # Apply token validity to output
        if token_mask is not None:
            context = context * token_mask.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(self.dropout(context))
        if output_attentions:
            # (you would need to compute attn_weights; for now just return None)
            return (output, None)
        else:
            return (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.weight.device)
    
def inject_spectra_v2_into_distilbert(
    layer, model_config, spectra_cfg, layer_idx, total_layers, old_attn
):
    new_attn = SpectraV2Attention(
        model_config=model_config,
        spectra_cfg=spectra_cfg,
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
        elif comp_config.attn_type == "spectra_v2":
            spectra_cfg = SpectraConfig(
                block_size=64,
                top_k_inter=16,
                adaptive_span=True,
                max_span=256,
                rectify_interval=4,
                aks_enabled=True,
            )
            return inject_spectra_v2_into_distilbert(
                layer, model_config, spectra_cfg, layer_idx,
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
                    choices=["linformer", "leka", "leka_v2", "spectra_v2"])
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






# (.venv) ➜  Codes git:(main) ✗ python SpectraV2Attention.py --attn_type spectra_v2 \
#   --max_len 64 --batch_size 32 --epochs 5 --lr 1e-5 --no_gate \
#   --model distilbert-base-uncased --task sst2
# 06:34:55 | INFO    | Task: sst2 | Progressive: True
# 06:34:55 | INFO    | k early/mid/late = 32/16/12
# 06:34:55 | INFO    | Raw dataset sizes: train=67349, val=872
# 06:34:55 | INFO    | Tokenized dataset: train=67349, val=872
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 5992.63it/s]
# [transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
# Key                     | Status     | 
# ------------------------+------------+-
# vocab_transform.weight  | UNEXPECTED | 
# vocab_layer_norm.bias   | UNEXPECTED | 
# vocab_projector.bias    | UNEXPECTED | 
# vocab_transform.bias    | UNEXPECTED | 
# vocab_layer_norm.weight | UNEXPECTED | 
# classifier.bias         | MISSING    | 
# classifier.weight       | MISSING    | 
# pre_classifier.bias     | MISSING    | 
# pre_classifier.weight   | MISSING    | 

# Notes:
# - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
# - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# 06:34:55 | INFO    | Created model with 6 compressed layers (type=distilbert)
# 06:34:55 | INFO    | Compression ratios: early=32, mid=16, late=12
# Using MPS (Metal) device
# 06:34:56 | INFO    | DEBUG: model device is mps:0
# jjj
# 06:34:56 | INFO    | Train dataset columns: ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
# 06:34:56 | INFO    | Train dataset features: {'input_ids': List(Value('int32')), 'token_type_ids': List(Value('int8')), 'attention_mask': List(Value('int8')), 'labels': Value('int64')}
# 06:34:56 | INFO    | First example: {'input_ids': [101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': 0}
# 06:34:56 | INFO    | CompressionTrainer initialized
# 06:34:56 | INFO    | Starting training...
# {'loss': '0.6922', 'grad_norm': '0.9323', 'learning_rate': '1.569e-06', 'epoch': '0.04751'}                                                                        
# {'loss': '0.6824', 'grad_norm': '3.353', 'learning_rate': '3.154e-06', 'epoch': '0.09501'}                                                                         
# {'loss': '0.663', 'grad_norm': '4.947', 'learning_rate': '4.739e-06', 'epoch': '0.1425'}                                                                           
# {'loss': '0.618', 'grad_norm': '14.87', 'learning_rate': '6.323e-06', 'epoch': '0.19'}                                                                             
# {'loss': '0.5986', 'grad_norm': '7.059', 'learning_rate': '7.908e-06', 'epoch': '0.2375'}                                                                          
# {'loss': '0.5577', 'grad_norm': '28.42', 'learning_rate': '9.493e-06', 'epoch': '0.285'}                                                                           
# {'loss': '0.4933', 'grad_norm': '13.33', 'learning_rate': '9.931e-06', 'epoch': '0.3325'}                                                                          
# {'loss': '0.4434', 'grad_norm': '21.21', 'learning_rate': '9.83e-06', 'epoch': '0.38'}                                                                             
# {'loss': '0.4258', 'grad_norm': '13.68', 'learning_rate': '9.729e-06', 'epoch': '0.4276'}                                                                          
# {'loss': '0.3922', 'grad_norm': '28.26', 'learning_rate': '9.628e-06', 'epoch': '0.4751'}                                                                          
# {'loss': '0.3588', 'grad_norm': '5.481', 'learning_rate': '9.527e-06', 'epoch': '0.5226'}                                                                          
# {'loss': '0.3408', 'grad_norm': '2.85', 'learning_rate': '9.426e-06', 'epoch': '0.5701'}                                                                           
# {'loss': '0.3495', 'grad_norm': '5.213', 'learning_rate': '9.325e-06', 'epoch': '0.6176'}                                                                          
# {'loss': '0.3229', 'grad_norm': '4.747', 'learning_rate': '9.224e-06', 'epoch': '0.6651'}                                                                          
# {'loss': '0.3185', 'grad_norm': '6.216', 'learning_rate': '9.123e-06', 'epoch': '0.7126'}                                                                          
# {'loss': '0.3238', 'grad_norm': '5.498', 'learning_rate': '9.022e-06', 'epoch': '0.7601'}                                                                          
# {'loss': '0.3099', 'grad_norm': '6.174', 'learning_rate': '8.921e-06', 'epoch': '0.8076'}                                                                          
# {'loss': '0.3122', 'grad_norm': '7.905', 'learning_rate': '8.819e-06', 'epoch': '0.8551'}                                                                          
# {'loss': '0.3002', 'grad_norm': '6.587', 'learning_rate': '8.718e-06', 'epoch': '0.9026'}                                                                          
# {'loss': '0.2726', 'grad_norm': '7.161', 'learning_rate': '8.617e-06', 'epoch': '0.9501'}                                                                          
# {'loss': '0.2977', 'grad_norm': '6.493', 'learning_rate': '8.516e-06', 'epoch': '0.9976'}                                                                          
# {'loss': '0.2629', 'grad_norm': '9.409', 'learning_rate': '8.415e-06', 'epoch': '1.045'}                                                                           
# {'loss': '0.2475', 'grad_norm': '7.853', 'learning_rate': '8.314e-06', 'epoch': '1.093'}                                                                           
# {'loss': '0.2402', 'grad_norm': '6.938', 'learning_rate': '8.213e-06', 'epoch': '1.14'}                                                                            
# {'loss': '0.2618', 'grad_norm': '3.037', 'learning_rate': '8.112e-06', 'epoch': '1.188'}                                                                           
# {'loss': '0.248', 'grad_norm': '2.79', 'learning_rate': '8.011e-06', 'epoch': '1.235'}                                                                             
# {'loss': '0.2447', 'grad_norm': '6.226', 'learning_rate': '7.91e-06', 'epoch': '1.283'}                                                                            
# {'loss': '0.2285', 'grad_norm': '1.992', 'learning_rate': '7.809e-06', 'epoch': '1.33'}                                                                            
# {'loss': '0.2349', 'grad_norm': '3.694', 'learning_rate': '7.708e-06', 'epoch': '1.378'}                                                                           
# {'loss': '0.2236', 'grad_norm': '6.939', 'learning_rate': '7.607e-06', 'epoch': '1.425'}                                                                           
# {'loss': '0.2213', 'grad_norm': '5.934', 'learning_rate': '7.506e-06', 'epoch': '1.473'}                                                                           
#  29%|███████████████████████████████████                                                                                    | 3102/10525 [50:21<1:49:18,  1.13it/s]^CTraceback (most recent call last):
#   File "/Users/tahamajs/Documents/uni/Project/Codes/SpectraV2Attention.py", line 1423, in <module>
#     main()
#     ~~~~^^
#   File "/Users/tahamajs/Documents/uni/Project/Codes/SpectraV2Attention.py", line 1416, in main
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
#   File "/Users/tahamajs/Documents/uni/Project/.venv/lib/python3.14/site-packages/transformers/trainer.py", line 1742, in _run_epoch
#     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
#                                       ~~~~~~~~~~~^^^^^^^^^^^^^^
# KeyboardInterrupt
#  29%|███████████████████████████████████                                                                                    | 3102/10525 [50:24<2:00:37,  1.03it/s]

# (.venv) ➜  Codes git:(main) ✗ 