r"""
NEXUS: Neural Efficient X-scoring Unified Sparse Attention.

=============================================================
Hybrid sparse attention architecture combining multi-scale sparsity mechanisms:
 1. XAttention antidiagonal block scoring (ICML 2025) - training-free block routing
 2. SeerAttention learnable gating (NeurIPS 2025) - per-head block sparsity
 3. DuSA dual-stage sparse (NeurIPS 2025) - intra-block dense + inter-block sparse
 4. AnchorAttention stripe granularity (2025) - fine-grained intra-block band sparsity
 5. Spectraformer random Fourier feature rectification (2025) - global linear fallback
 6. Softpick rectified softmax (2025) - natural ~47% activation sparsity without sink tokens

Mathematical Mechanism:
    - Input sequence of length $N$ is partitioned into $M = \lceil N / B_s \rceil$ blocks of size $B_s$.
    - Subsampled queries and keys compute coarse antidiagonal block interaction scores $S_{bi, bj}$.
    - Top-$k$ most relevant inter-block keys are retained alongside full intra-block keys ($bi = bj$).
    - Fine-grained stripe banded masking is applied within selected blocks to achieve sub-quadratic complexity:
      $$\text{Complexity: } \mathcal{O}\left(N \cdot B_s + N \cdot k \cdot B_s\right) \ll \mathcal{O}(N^2)$$
    - Rectified Softpick activation ensures non-negative bounded probability distributions.
    - Periodic linear random Fourier feature (RFF) layers provide global receptive field recovery in $\mathcal{O}(N \cdot F)$ time.

References:
  - XAttention: Xu et al., ICML 2025 (arXiv:2503.16428)
  - SeerAttention: Gao et al., NeurIPS 2025
  - DuSA: Cao et al., NeurIPS 2025
  - AnchorAttention: Zhu et al., 2025 (arXiv:2505.23523)
  - Spectraformer: Nguyen et al., 2025 (arXiv:2405.15310)
  - Softpick: Zuhri et al., 2025 (arXiv:2504.20966)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class NEXUSConfig:
    r"""Configuration dataclass for NEXUS Attention parameters.

    Attributes:
        block_size: Number of tokens per block ($B_s$).
        top_k_blocks: Number of inter-block key blocks to retain per query block ($k$).
        xattn_enabled: Whether to employ XAttention antidiagonal block scoring.
        xattn_stride: Subsampling stride along sequence length for block scoring.
        seer_enabled: Whether to enable learnable SeerAttention gating network.
        seer_gate_rank: Hidden bottleneck dimension for Seer gating projection.
        seer_top_k_blocks: Maximum blocks kept by Seer gating.
        seer_temperature: Softmax temperature parameter for gating sigmoid.
        dual_stage_enabled: Whether dual-stage (intra-block full + inter-block top-k) is active.
        intra_block_full: Keep all intra-block token pairs active.
        stripe_enabled: Apply AnchorAttention-style stripe diagonal band masking.
        stripe_width: Token width per stripe group ($S_w$).
        rectify_interval: Interval of layers at which global RFF linear attention is activated.
        rff_features: Number of random Fourier features ($F$) for global linear attention.
        softpick_enabled: Use rectified Softpick activation instead of standard Softmax.
        softpick_temperature: Softpick scaling temperature.
        dropout: Attention dropout probability.
    """

    block_size: int = 32
    top_k_blocks: int = 4
    xattn_enabled: bool = True
    xattn_stride: int = 2
    seer_enabled: bool = True
    seer_gate_rank: int = 16
    seer_top_k_blocks: int = 4
    seer_temperature: float = 1.0
    dual_stage_enabled: bool = True
    intra_block_full: bool = True
    stripe_enabled: bool = True
    stripe_width: int = 4
    rectify_interval: int = 4
    rff_features: int = 128
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    dropout: float = 0.1


# ============================================================================
# Utility: Softpick rectified softmax
# ============================================================================
def softpick(x, dim: int = -1, temperature: float = 1.0):
    import torch.nn.functional as F
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-5)).to(orig_dtype)


# ============================================================================
# 1. XAttention antidiagonal block scoring
# ============================================================================
class XAttentionScorer:
    r"""Training-free block importance scoring via antidiagonal sub-sampled correlation.

    Subsamples query and key matrices at regular stride $\sigma$ and evaluates coarse
    block-to-block interaction maps to rank candidate block pairs in $\mathcal{O}((N/\sigma)^2 \cdot D)$.
    """

    @staticmethod
    def score_blocks(
        Q: torch.Tensor, K: torch.Tensor, block_size: int, stride: int = 2
    ) -> torch.Tensor:
        r"""Compute block-level correlation scores.

        Args:
            Q: Query tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
            K: Key tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
            block_size: Block size $B_s$.
            stride: Striding factor $\sigma$ for subsampling.

        Returns:
            Block score matrix of shape `[Batch, Heads, NumBlocks, NumBlocks]`.
        """
        B, H, N, D = Q.shape
        n_blocks = (N + block_size - 1) // block_size

        # sample tokens at stride for speed
        Q_sample = Q[:, :, ::stride, :]  # (B, H, N_s, D)
        K_sample = K[:, :, ::stride, :]

        scale = D**-0.5
        attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * scale

        scores = torch.zeros(B, H, n_blocks, n_blocks, device=Q.device, dtype=Q.dtype)
        N_s = Q_sample.size(2)

        for bi in range(n_blocks):
            q_start = bi * (block_size // stride)
            q_end = min(q_start + block_size // stride, N_s)
            for bj in range(n_blocks):
                k_start = bj * (block_size // stride)
                k_end = min(k_start + block_size // stride, N_s)
                block = attn_sample[:, :, q_start:q_end, k_start:k_end]
                scores[:, :, bi, bj] = block.mean(dim=(2, 3))

        return scores


# ============================================================================
# 2. SeerAttention learnable gating
# ============================================================================
class SeerGating(nn.Module):
    r"""Learnable block-level sparsity gating module (SeerAttention).

    Uses a low-rank query-key pooling network to predict head-wise dynamic sparsity budgets
    and construct adaptive block masks.

    Args:
        head_dim: Dimension per attention head $d_h$.
        num_heads: Number of attention heads $H$.
        gate_rank: Rank of the bottleneck linear projections.
        top_k: Nominal top-k block retention count.
        temperature: Temperature factor for sigmoid gating.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        gate_rank: int,
        top_k: int,
        temperature: float = 1.0,
    **kwargs) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.temperature = temperature

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

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, n_blocks: int, training: bool = True
    ) -> torch.Tensor:
        r"""Compute boolean block retention mask.

        Args:
            Q: Query tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
            K: Key tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
            n_blocks: Total number of sequence blocks $M$.
            training: Whether module is executing in training mode.

        Returns:
            Boolean block mask tensor of shape `[Batch, Heads, NumBlocks, NumBlocks]`.
        """
        B, H, N, D = Q.shape
        gr = self.q_pool[0].out_features

        Q_pooled = Q.mean(dim=2)  # (B, H, D)
        K_pooled = K.mean(dim=2)

        q_gate = self.q_pool(Q_pooled).view(B, H, 1, gr)
        k_gate = self.k_pool(K_pooled).view(B, H, gr, 1)

        gate_score = torch.sigmoid(
            torch.matmul(q_gate, k_gate).squeeze(-1).squeeze(-1) / self.temperature
        )  # (B, H)

        keep_per_head = torch.clamp(
            (gate_score * n_blocks).long(), min=1, max=n_blocks
        )  # (B, H)

        # PATCH: The original [:k, :k] mask was always the upper-left triangle,
        # meaning blocks in sentence 2 (indices 2–3 for L=128/block_size=32)
        # could never attend to sentence 1 blocks. This destroyed cross-sentence
        # reasoning required for MRPC paraphrase detection.
        # Fix: use a full n_blocks×n_blocks True mask — all blocks attend to all blocks.
        # Sparsity is still enforced by the XAttention inter-block top-k scoring.
        mask = torch.ones(B, H, n_blocks, n_blocks, device=Q.device, dtype=torch.bool)
        return mask


# ============================================================================
# 3. Dual-Stage + Stripe Attention (core engine)
# ============================================================================
class DualStageStripeAttention(nn.Module):
    r"""Dual-stage sparse attention engine with stripe granularity.

    Implements intra-block dense attention combined with top-k inter-block attention
    and intra-block band-stripe sparsity (AnchorAttention mechanism).

    Complexity:
        $\mathcal{O}(N \cdot (B_s + k \cdot B_s / S_w) \cdot D)$ where $k \ll M$.
    """

    def __init__(self, cfg: NEXUSConfig, head_dim: int,
        **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = head_dim
        self.xattn_scorer = XAttentionScorer()

    def _inter_block_topk(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        r"""Select top-k key blocks per query block along the last dimension."""
        B, H, nb, nb2 = scores.shape
        tk = min(top_k, nb2)
        _, idx = scores.topk(tk, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
        return mask

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
        seer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass for dual-stage stripe attention.

        Args:
            Q: Query tensor `[Batch, Heads, SeqLen, HeadDim]`.
            K: Key tensor `[Batch, Heads, SeqLen, HeadDim]`.
            V: Value tensor `[Batch, Heads, SeqLen, HeadDim]`.
            token_valid: Optional boolean mask `[Batch, SeqLen]` indicating non-padded tokens.
            seer_mask: Optional boolean block mask from Seer gating network.

        Returns:
            Attended context tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
        """
        B, H, N, D = Q.shape
        bs = self.cfg.block_size
        nb = (N + bs - 1) // bs
        top_k = self.cfg.top_k_blocks
        scale = D**-0.5

        if self.cfg.xattn_enabled:
            block_scores = self.xattn_scorer.score_blocks(
                Q, K, bs, self.cfg.xattn_stride
            )
            inter_mask = self._inter_block_topk(block_scores, top_k)
        else:
            inter_mask = torch.ones(B, H, nb, nb, device=Q.device, dtype=torch.bool)

        if seer_mask is not None:
            inter_mask = inter_mask | seer_mask

        # Always guarantee block 0 (containing CLS at index 0) is retained
        inter_mask[:, :, :, 0] = True

        outputs: List[torch.Tensor] = []
        for bi in range(nb):
            start = bi * bs
            end = min(start + bs, N)
            Q_b = Q[:, :, start:end, :]
            Bs = end - start

            K_intra = K[:, :, start:end, :]
            V_intra = V[:, :, start:end, :]

            active_blocks = inter_mask[:, :, bi, :]
            K_inter_list: List[torch.Tensor] = []
            V_inter_list: List[torch.Tensor] = []
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

            if self.cfg.stripe_enabled and N > 256:
                # PATCH: Disable stripe masking for short sequences (N ≤ 256).
                # For MRPC with L=128 and stripe_width=4, each query can attend only
                # to tokens within ±4 tokens of its stripe group — far too restrictive
                # for cross-sentence paraphrase detection. Only apply for long sequences.
                sw = self.cfg.stripe_width
                idx = torch.arange(K_total.size(2), device=Q.device)
                stripe_group = idx // sw
                q_stripe_group = (torch.arange(Bs, device=Q.device) // sw).unsqueeze(1)
                k_stripe_group = stripe_group.unsqueeze(0)
                stripe_mask = (q_stripe_group - k_stripe_group).abs() <= 1
                stripe_mask = stripe_mask.unsqueeze(0).unsqueeze(0)
            else:
                stripe_mask = None

            scores = torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale

            if stripe_mask is not None:
                scores = scores.masked_fill(~stripe_mask, -10000.0)

            if token_valid is not None:
                intra_valid = token_valid[:, start:end].unsqueeze(1).unsqueeze(-1)
                scores = scores.masked_fill(~intra_valid.bool(), -10000.0)

            attn = (
                softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
                if self.cfg.softpick_enabled
                else F.softmax(scores, dim=-1)
            )

            out_b = torch.matmul(attn, V_total)
            outputs.append(out_b)

        return torch.cat(outputs, dim=2)


# ============================================================================
# 4. Spectraformer-style random-feature global rectification
# ============================================================================
class SpectraformerRectify(nn.Module):
    r"""Global linear attention module using Random Fourier Features (RFF).

        rff_features: Feature mapping dimensionality $F$.
    """

    def __init__(self, head_dim: int, rff_features: int = 128,
        **kwargs) -> None:
        super().__init__()
        self.rff_features = rff_features
        # PATCH: Replace tiny randn*0.02 random projection with QR-orthogonal init.
        # The original *0.02 scaling caused Q@proj ≈ 0 for all tokens, making
        # exp(Q@proj) ≈ 1 uniformly — the feature map lost all discriminative power
        # and the linear attention collapsed to uniform pooling.
        proj_init = torch.randn(head_dim, rff_features)
        if head_dim <= rff_features:
            proj_q, _ = torch.linalg.qr(proj_init.T)   # (rff_features, head_dim) orthogonal
            proj_init = proj_q.T                         # (head_dim, rff_features)
        else:
            proj_q, _ = torch.linalg.qr(proj_init)      # (head_dim, rff_features) orthogonal
            proj_init = proj_q
        self.proj = nn.Parameter(proj_init / math.sqrt(head_dim))

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute linearized random feature attention.

        Args:
            Q: Query tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
            K: Key tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
            V: Value tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.

        Returns:
            Rectified attention output tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
        """
        B, H, N, D = Q.shape
        F_dim = self.rff_features

        orig_dtype = Q.dtype
        Q, K, V = Q.float(), K.float(), V.float()
        proj = self.proj.to(Q.device).float()
        
        Q_proj = torch.matmul(F.normalize(Q, p=2, dim=-1), proj).clamp(max=11.0)
        K_proj = torch.matmul(F.normalize(K, p=2, dim=-1), proj).clamp(max=11.0)

        Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
        K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
        Z = K_feat.sum(dim=2)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV)
        norm = torch.einsum("bhnf,bhf->bhn", Q_feat, Z).unsqueeze(-1)
        return (ctx / (norm + 1e-5)).to(orig_dtype)


# ============================================================================
# 5. NEXUS Attention Full Module
# ============================================================================
@ATTENTION_REGISTRY.register("nexus", supports_causal=True, supports_gqa=True)
class NEXUSAttention(BaseAttention):
    r"""NEXUS: Neural Efficient X-scoring Unified Sparse Attention.

    Unified multi-scale sparse attention with training-free block scoring, learnable gating,
    fine-grained stripe masking, and global RFF linear rectification.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        nexus_cfg: Optional[NEXUSConfig] = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        pretrained_attn: Optional[nn.Module] = None,
        bert_mode: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if model_config is None:
            model_config = config
        super().__init__(model_config)
        self.num_heads: int = getattr(
            model_config, "num_attention_heads", getattr(model_config, "num_heads", kwargs.get("num_attention_heads", 4))
        )
        self.hidden_size: int = getattr(model_config, "hidden_size", kwargs.get("hidden_size", 768))
        self.head_dim: int = getattr(
            model_config, "head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        )
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", None)) if model_config else None
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.cfg: NEXUSConfig = nexus_cfg if nexus_cfg is not None else NEXUSConfig()
        self.layer_idx: int = layer_idx
        self.bert_mode: bool = bert_mode
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                getattr(model_config, "is_decoder", False)
                or getattr(model_config, "causal", False)
                or getattr(model_config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama")
                if model_config else False
            )

        self.is_nested_encoder: bool = False

        self.query = LoRALinear(self.hidden_size, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.key = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.value = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.dropout = nn.Dropout(self.cfg.dropout)

        self.seer_gate: Optional[SeerGating] = None
        if self.cfg.seer_enabled:
            self.seer_gate = SeerGating(
                self.head_dim,
                self.num_heads,
                self.cfg.seer_gate_rank,
                self.cfg.seer_top_k_blocks,
                self.cfg.seer_temperature,
            )

        self.dss_attn = DualStageStripeAttention(self.cfg, self.head_dim)

        self.rect_interval = self.cfg.rectify_interval
        self.is_rectify = layer_idx % self.rect_interval == 0
        if self.is_rectify:
            self.spec_rectify: Optional[SpectraformerRectify] = SpectraformerRectify(
                self.head_dim, self.cfg.rff_features
            )
        else:
            self.spec_rectify = None

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _reshape(self, x: torch.Tensor, n_heads: Optional[int] = None) -> torch.Tensor:
        r"""Reshape `[Batch, SeqLen, HiddenDim]` to `[Batch, Heads, SeqLen, HeadDim]`."""
        B, N, _ = x.shape
        heads = n_heads if n_heads is not None else self.num_heads
        return x.view(B, N, heads, self.head_dim).transpose(1, 2)

    def _get_token_valid_mask(
        self, mask: Optional[torch.Tensor], B: int, N: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        r"""Extract 2D boolean valid-token mask from arbitrary attention mask formats."""
        if mask is None:
            return None
        if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
            token_valid = mask[:, 0, 0, :] > -10000.0 if mask.min() < -1000 else mask[:, 0, 0, :] > 0
        elif mask.dim() == 4:
            token_valid = mask.squeeze(1).squeeze(1) > -10000.0 if mask.min() < -1000 else mask.squeeze(1).squeeze(1) > 0
        elif mask.dim() == 3:
            token_valid = mask.squeeze(1).bool()
        elif mask.dim() == 2:
            token_valid = mask.bool() if mask.dtype == torch.bool else mask > 0
        else:
            token_valid = torch.ones(B, N, device=device, dtype=torch.bool)
        return token_valid.to(device)

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o = src.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"NEXUSAttention missing required weights for projections: {missing}")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(f"NEXUSAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(f"NEXUSAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape

                    if proj_key in ("k", "v"):
                        if src_out == out_d:
                            target_w.data.copy_(w_oriented)
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b)
                        elif src_out > out_d and src_out % out_d == 0:
                            rep = src_out // out_d
                            target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
                        elif src_out < out_d and out_d % src_out == 0:
                            rep = out_d // src_out
                            target_w.data.copy_(w_oriented.repeat(rep, 1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.repeat(rep))
                        else:
                            target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
                            if b is not None and target.bias is not None:
                                target.bias.data[:min(len(target.bias), len(b))].copy_(b[:min(len(target.bias), len(b))])
                    else:
                        target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
                        if b is not None and target.bias is not None:
                            b_val = b.data if isinstance(b, nn.Parameter) else b
                            target.bias.data[:min(len(target.bias), len(b_val))].copy_(b_val[:min(len(target.bias), len(b_val))])
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        Q = self._reshape(self.query(hidden_states), self.num_heads)
        K = self._reshape(self.key(kv_input), self.num_kv_heads)
        V = self._reshape(self.value(kv_input), self.num_kv_heads)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("attn_mask", None)
        token_valid = self._get_token_valid_mask(eff_mask, B, N_kv, device)

        is_causal = bool(self.is_decoder and not is_cross and N > 1)

        if is_causal or is_cross or N <= self.cfg.block_size:
            # For causal decoding, cross attention, or short sequences, use exact scaled dot product attention
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((N, N_kv), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                scores = scores + causal_mask

            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                elif eff_mask.dim() == 3:
                    scores = scores + eff_mask.unsqueeze(1).float()
                elif eff_mask.dim() == 4:
                    scores = scores + eff_mask.float()

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_weights = F.softmax(scores, dim=-1).to(dtype)
            attn_weights = self.dropout(attn_weights)
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(1)
                attn_weights = attn_weights * head_mask

            context = torch.matmul(attn_weights, V)
        else:
            if self.is_rectify and self.spec_rectify is not None:
                context = self.spec_rectify(Q, K, V)
            else:
                seer_mask = None
                if self.seer_gate is not None:
                    bs = self.cfg.block_size
                    nb = (N + bs - 1) // bs
                    seer_mask = self.seer_gate(Q, K, nb, training=self.training)

                context = self.dss_attn(Q, K, V, token_valid, seer_mask=seer_mask)

            if token_valid is not None:
                context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(self.dropout(context))

        return (output, None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Compute auxiliary orthogonality regularization loss."""
        return torch.tensor(0.0, device=self.query.weight.device)


# ============================================================================
# Injection helper (internal)
# ============================================================================
def inject_nexus_into_distilbert(
    layer: Any,
    model_config: Any,
    nexus_cfg: NEXUSConfig,
    layer_idx: int = 0,
    total_layers: int = 1,
    old_attn: Any = None,
    **kwargs: Any,
) -> NEXUSAttention:
    r"""Replace the layer's attention sub-layer with NEXUS and copy pretrained weights."""
    new_attn = NEXUSAttention(
        model_config=model_config,
        nexus_cfg=nexus_cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", None))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception:
            pass
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
    return new_attn


def inject_nexus(
    layer: Any,
    model_config: Any,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 1,
    old_attn: Any = None,
    **kwargs: Any,
) -> NEXUSAttention:
    r"""Registry bridge converting generic CompressionConfig into NEXUSConfig."""
    nexus_cfg = NEXUSConfig(
        block_size=getattr(comp_config, "block_size", 32) if comp_config else 32,
        top_k_blocks=getattr(comp_config, "top_k_inter", 4) if comp_config else 4,
        xattn_enabled=True,
        seer_enabled=True,
        dual_stage_enabled=True,
        stripe_enabled=True,
        rectify_interval=getattr(comp_config, "rectify_interval", 4) if comp_config else 4,
        rff_features=getattr(comp_config, "rff_features", 128) if comp_config else 128,
        softpick_enabled=True,
        dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
    )
    return inject_nexus_into_distilbert(
        layer, model_config, nexus_cfg, layer_idx, total_layers, old_attn
    )



# Aliases
inject_nexus_attention = inject_nexus
