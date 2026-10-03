"""
NEXUS: Neural Efficient X-scoring Unified Sparse Attention.

=============================================================
Hybrid sparse attention architecture combining multi-scale sparsity mechanisms:
 1. XAttention antidiagonal block scoring (ICML 2025) - training-free block routing
 2. SeerAttention learnable gating (NeurIPS 2025) - per-head block sparsity
 3. DuSA dual-stage sparse (NeurIPS 2025) - intra-block dense + inter-block sparse
 4. AnchorAttention stripe granularity (2025) - fine-grained intra-block band sparsity
 5. Spectraformer random Fourier feature rectification (2025) - global linear fallback
 6. Softpick rectified softmax (2025) - natural ~47% activation sparsity

"""

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

# ===========================================================================
# RoPE
# ===========================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

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
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# LoRALinear (NX-10: no self-move on forward)
# ===========================================================================


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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
            (nn.zeros_(self.lora_B) if hasattr(nn, "zeros_") else torch.nn.init.zeros_(self.lora_B))
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
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(
                F.linear(x, self.lora_A.to(dtype=x.dtype)),
                self.lora_B.to(dtype=x.dtype),
            )
            out = out + lora_out * self.lora_scaling
        return out


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class NEXUSConfig:
    block_size: int = 16
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
    sparse_min_seq_len: int = 8
    enable_spectraformer: bool = False
    dropout: float = 0.1


# ===========================================================================
# Softpick rectified softmax
# ===========================================================================


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    temp = max(temperature, 1e-6)
    m = x_f.max(dim=dim, keepdim=True).values
    m = torch.where(m < -1e3, torch.zeros_like(m), m)
    z = x_f - m
    exp_z = torch.exp(z / temp)
    exp_neg_m = torch.exp((-m) / temp).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)
    zero_mask = denom <= 1e-8
    if zero_mask.any():
        fallback = F.softmax(x_f, dim=dim)
        p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
    else:
        p = p / denom
    return p.to(orig_dtype)


# ===========================================================================
# 1. XAttention antidiagonal block scoring (NX-06: vectorized)
# ===========================================================================


class XAttentionScorer:
    @staticmethod
    def score_blocks(
        Q: torch.Tensor, K: torch.Tensor, block_size: int, stride: int = 2
    ) -> torch.Tensor:
        B, H, N, D = Q.shape
        n_blocks = (N + block_size - 1) // block_size

        Q_sample = Q[:, :, ::stride, :]
        K_sample = K[:, :, ::stride, :]
        N_s = Q_sample.size(2)

        scale = D**-0.5
        attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1).contiguous()) * scale

        b_s = max(1, block_size // stride)
        target = n_blocks * b_s
        if N_s < target:
            pad = target - N_s
            attn_sample = F.pad(attn_sample, (0, pad, 0, pad), value=0.0)
        attn_sample = attn_sample[:, :, :target, :target]

        attn_blocks = attn_sample.view(B, H, n_blocks, b_s, n_blocks, b_s)
        scores = attn_blocks.mean(dim=(3, 5))
        return scores


# ===========================================================================
# 2. SeerAttention learnable gating
# ===========================================================================


class SeerGating(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        gate_rank: int,
        top_k: int,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
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
        B, H, N, D = Q.shape
        gr = self.q_pool[0].out_features

        Q_pooled = Q.mean(dim=2)
        K_pooled = K.mean(dim=2)

        q_gate = self.q_pool(Q_pooled).view(B, H, 1, gr)
        k_gate = self.k_pool(K_pooled).view(B, H, gr, 1)

        gate_score = torch.sigmoid(
            torch.matmul(q_gate, k_gate).squeeze(-1).squeeze(-1) / self.temperature
        )

        keep_per_head = torch.clamp((gate_score * n_blocks).long(), min=1, max=n_blocks)

        mask = torch.zeros(B, H, n_blocks, n_blocks, device=Q.device, dtype=torch.bool)
        diag_idx = torch.arange(n_blocks, device=Q.device)
        mask[:, :, diag_idx, diag_idx] = True
        mask[:, :, :, 0] = True

        for b in range(B):
            for h in range(H):
                k_keep = keep_per_head[b, h].item()
                if k_keep > 1 and n_blocks > 2:
                    extra_needed = min(k_keep - 1, n_blocks - 1)
                    if extra_needed > 0:
                        mask[b, h, :, 1 : 1 + extra_needed] = True
        return mask


# ===========================================================================
# 3. Dual-Stage + Stripe Attention (NX-01, NX-02, NX-04, NX-07)
# ===========================================================================


class DualStageStripeAttention(nn.Module):
    def __init__(
        self,
        cfg: NEXUSConfig,
        head_dim: int,
        num_heads: int = 1,
        is_decoder: bool = False,
        is_nested_encoder: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.is_decoder = is_decoder
        self.is_nested_encoder = is_nested_encoder
        self.xattn_scorer = XAttentionScorer()

    def _inter_block_topk(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        B, H, nb, nb2 = scores.shape
        tk = min(top_k, nb2)
        scores_inter = scores.clone()
        min_dim = min(nb, nb2)
        diag_idx = torch.arange(min_dim, device=scores.device)
        scores_inter[:, :, diag_idx, diag_idx] = float("-inf")
        _, idx = scores_inter.topk(tk, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
        mask[:, :, diag_idx, diag_idx] = True
        return mask

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None = None,
        seer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, N, D = Q.shape
        bs = self.cfg.block_size
        nb = (N + bs - 1) // bs
        top_k = self.cfg.top_k_blocks
        scale = D**-0.5

        if self.cfg.xattn_enabled:
            block_scores = self.xattn_scorer.score_blocks(Q, K, bs, self.cfg.xattn_stride)
            inter_mask = self._inter_block_topk(block_scores, top_k)
        else:
            inter_mask = torch.ones(B, H, nb, nb, device=Q.device, dtype=torch.bool)

        if seer_mask is not None:
            inter_mask = inter_mask | seer_mask

        inter_mask[:, :, :, 0] = True
        diag_idx = torch.arange(nb, device=Q.device)
        inter_mask[:, :, diag_idx, diag_idx] = True

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        expanded_mask = inter_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)[
            :, :, :N, :N
        ]
        scores = scores.masked_fill(~expanded_mask, -10000.0)

        if self.cfg.stripe_enabled and N > 256:
            sw = self.cfg.stripe_width
            idx = torch.arange(N, device=Q.device)
            stripe_group = idx // sw
            q_stripe = stripe_group.unsqueeze(1)
            k_stripe = stripe_group.unsqueeze(0)
            stripe_mask = (q_stripe - k_stripe).abs() <= 1
            scores = scores.masked_fill(~stripe_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        if self.is_decoder:
            causal = torch.tril(torch.ones(N, N, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -10000.0)

        if token_valid is not None:
            scores = scores.masked_fill((~token_valid.bool()).unsqueeze(1).unsqueeze(2), -10000.0)

        if self.cfg.softpick_enabled:
            attn = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature).to(Q.dtype)
        else:
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores, dim=-1).to(Q.dtype)

        context = torch.matmul(attn, V)
        return context


# ===========================================================================
# 4. Spectraformer-style random-feature global rectification (NX-03: clamp fixed)
# ===========================================================================


class SpectraformerRectify(nn.Module):
    def __init__(self, head_dim: int, rff_features: int = 128, **kwargs) -> None:
        super().__init__()
        self.rff_features = rff_features
        proj_init = torch.randn(head_dim, rff_features)
        if head_dim <= rff_features:
            proj_q, _ = torch.linalg.qr(proj_init.T)
            proj_init = proj_q.T
        else:
            proj_q, _ = torch.linalg.qr(proj_init)
            proj_init = proj_q
        self.proj = nn.Parameter(proj_init / math.sqrt(head_dim))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
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
        # NX-03: clamp norm, not (norm + eps)
        return (ctx / norm.clamp(min=1e-5)).to(orig_dtype)


# ===========================================================================
# 5. NEXUS Attention (full module)
# ===========================================================================


@ATTENTION_REGISTRY.register("nexus", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("nexus_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("NEXUSAttention", supports_causal=True, supports_gqa=True)
class NEXUSAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        model_config: Any | None = None,
        nexus_cfg: NEXUSConfig | None = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        pretrained_attn: nn.Module | None = None,
        bert_mode: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if model_config is None:
            model_config = config
        super().__init__(model_config)
        self.num_heads: int = getattr(
            model_config,
            "num_attention_heads",
            getattr(model_config, "num_heads", kwargs.get("num_attention_heads", 4)),
        )
        self.hidden_size: int = getattr(model_config, "hidden_size", kwargs.get("hidden_size", 768))
        cfg_head_dim = getattr(model_config, "head_dim", None) if model_config else None
        adapter_head_dim = (
            getattr(getattr(model_config, "adapter", None), "head_dim", None)
            if model_config
            else None
        )
        self.head_dim: int = int(
            cfg_head_dim
            or adapter_head_dim
            or kwargs.get("head_dim")
            or (self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        )
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = (
            getattr(
                model_config,
                "num_key_value_heads",
                getattr(model_config, "num_kv_heads", None),
            )
            if model_config
            else None
        )
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
            _mt = str(getattr(model_config, "model_type", "") or "").lower() if model_config else ""
            self.is_decoder = (
                bool(
                    getattr(model_config, "is_decoder", False)
                    or getattr(model_config, "causal", False)
                )
                or any(
                    _mt.startswith(p)
                    for p in (
                        "gpt",
                        "qwen",
                        "llama",
                        "mistral",
                        "falcon",
                        "bloom",
                        "opt",
                    )
                )
                if model_config
                else False
            )

        self.is_nested_encoder: bool = False

        src_p = next(pretrained_attn.parameters(), None) if pretrained_attn is not None else None
        target_dtype = kwargs.get("dtype", src_p.dtype if src_p is not None else torch.float32)
        target_device = kwargs.get("device", src_p.device if src_p is not None else None)

        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            device=target_device,
            dtype=target_dtype,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            device=target_device,
            dtype=target_dtype,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            device=target_device,
            dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            device=target_device,
            dtype=target_dtype,
        )
        self.dropout = nn.Dropout(self.cfg.dropout)

        # NX-04: pass is_decoder into DSS
        self.dss_attn = DualStageStripeAttention(
            self.cfg,
            self.head_dim,
            num_heads=self.num_heads,
            is_decoder=self.is_decoder,
            is_nested_encoder=self.is_nested_encoder,
        )

        self.seer_gate: SeerGating | None = None
        if self.cfg.seer_enabled:
            self.seer_gate = SeerGating(
                self.head_dim,
                self.num_heads,
                self.cfg.seer_gate_rank,
                self.cfg.seer_top_k_blocks,
                self.cfg.seer_temperature,
            )

        self.rect_interval = self.cfg.rectify_interval
        self.is_rectify = (
            (self.rect_interval > 0)
            and (layer_idx % self.rect_interval == 0)
            and ((pretrained_attn is None) or getattr(self.cfg, "enable_spectraformer", False))
        )
        if self.is_rectify:
            self.spec_rectify: SpectraformerRectify | None = SpectraformerRectify(
                self.head_dim, self.cfg.rff_features
            )
        else:
            self.spec_rectify = None

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        if target_device is not None and target_dtype is not None:
            self.to(device=target_device, dtype=target_dtype)
        elif target_device is not None:
            self.to(device=target_device)
        elif target_dtype is not None:
            self.to(target_dtype)

    @property
    def q_proj(self):
        return self.query

    @property
    def k_proj(self):
        return self.key

    @property
    def v_proj(self):
        return self.value

    @property
    def q_lin(self):
        return self.query

    @property
    def k_lin(self):
        return self.key

    @property
    def v_lin(self):
        return self.value

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    def _reshape(self, x: torch.Tensor, n_heads: int | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        heads = n_heads if n_heads is not None else self.num_heads
        return x.view(B, N, heads, self.head_dim).transpose(1, 2).contiguous()

    # NX-05: robust valid-mask extraction
    def _get_token_valid_mask(self, mask, B, N, device):
        if mask is None:
            return None
        if mask.dim() == 2:
            if mask.dtype == torch.bool:
                return mask.to(device)
            if mask.dtype.is_floating_point:
                return (mask > -1e4).to(device)
            return (mask > 0).to(device)
        if mask.dim() == 3:
            if mask.dtype == torch.bool:
                v = mask
            elif mask.dtype.is_floating_point:
                v = mask > -1e4
            else:
                v = mask > 0
            return v.any(dim=1).to(device)
        if mask.dim() == 4:
            if mask.dtype == torch.bool:
                v = mask
            elif mask.dtype.is_floating_point:
                v = mask > -1e4
            else:
                v = mask > 0
            return v.any(dim=-2).squeeze(1).to(device)
        return torch.ones(B, N, device=device, dtype=torch.bool)

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t().contiguous()
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
            if (
                w_o.dim() == 2
                and w_o.shape[0] != self.hidden_size
                and w_o.shape[1] == self.hidden_size
            ):
                w_o = w_o.t().contiguous()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

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
                            extracted[proj] = (
                                layer.weight.data,
                                getattr(layer, "bias", None),
                            )
                            break

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
            )
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)
        # NX-09: propagate to DSS
        if hasattr(self, "dss_attn"):
            self.dss_attn.is_nested_encoder = self.is_nested_encoder

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("Module copy_weights warning. Proceeding with initialized weights.")

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"NEXUSAttention extracted weight for {proj_key} near-zero norm")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning("NEXUSAttention output weight near-zero norm")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in [
                ("q", self.query),
                ("k", self.key),
                ("v", self.value),
                ("o", self.out_proj),
            ]:
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
                            target_w.data.zero_()
                            target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                                w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                            )
                            if b is not None and target.bias is not None:
                                target.bias.data.zero_()
                                target.bias.data[: min(len(target.bias), len(b))].copy_(
                                    b[: min(len(target.bias), len(b))]
                                )
                    else:
                        target_w.data.zero_()
                        target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                            w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                        )
                        if b is not None and target.bias is not None:
                            b_val = b.data if isinstance(b, nn.Parameter) else b
                            target.bias.data.zero_()
                            target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                                b_val[: min(len(target.bias), len(b_val))]
                            )
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            if hasattr(self, "dss_attn"):
                self.dss_attn.is_decoder = self.is_decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        # NX-08: RoPE with past_len offset
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len : past_len + N, :]
            sin = sin[..., past_len : past_len + N, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("attn_mask")
        token_valid = self._get_token_valid_mask(eff_mask, B, N_kv, device)

        is_causal = bool(self.is_decoder and not is_cross)
        min_seq_len = getattr(self.cfg, "sparse_min_seq_len", 8)
        use_dense = is_causal or is_cross or (min_seq_len >= N) or (self.dss_attn is None)

        if use_dense:
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
            if is_causal:
                causal_diag = 1 + max(0, N_kv - N)
                causal_mask = (
                    torch.triu(
                        torch.full((N, N_kv), float("-inf"), device=device, dtype=torch.float32),
                        diagonal=causal_diag,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
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
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask[:, :, :1].unsqueeze(-1)
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
        return torch.tensor(0.0, device=self.query.weight.device)


# ===========================================================================
# Injection helper
# ===========================================================================


@ATTENTION_REGISTRY.register("inject_nexus")
@ATTENTION_REGISTRY.register("nexus_inject")
@ATTENTION_REGISTRY.register("inject_nexus_attention")
@ATTENTION_REGISTRY.register("nexus_attention_inject")
def inject_nexus(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> NEXUSAttention | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_nexus: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_nexus(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src_attn = old_attn
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "self"):
                        src_attn = cand.self
                    else:
                        src_attn = cand
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        if isinstance(src_attn, NEXUSAttention) or "NEXUS" in src_attn.__class__.__name__:
            return None
        if hasattr(src_attn, "inner") and isinstance(src_attn.inner, NEXUSAttention):
            return None
        if hasattr(src_attn, "_wrapped_attention") and isinstance(
            src_attn._wrapped_attention, NEXUSAttention
        ):
            return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, NEXUSAttention):
                    return None
                if "NEXUS" in cand.__class__.__name__:
                    return None

        if src_attn is not None:
            c_name = src_attn.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        nexus_cfg = kwargs.get("nexus_cfg")
        if nexus_cfg is None:
            nexus_cfg = NEXUSConfig(
                block_size=(getattr(comp_config, "block_size", 16) if comp_config else 16),
                top_k_blocks=(getattr(comp_config, "top_k_inter", 4) if comp_config else 4),
                xattn_enabled=True,
                seer_enabled=True,
                dual_stage_enabled=True,
                stripe_enabled=True,
                rectify_interval=(
                    getattr(comp_config, "rectify_interval", 4) if comp_config else 4
                ),
                rff_features=(getattr(comp_config, "rff_features", 128) if comp_config else 128),
                softpick_enabled=True,
                sparse_min_seq_len=8,
                enable_spectraformer=(
                    getattr(comp_config, "enable_spectraformer", False) if comp_config else False
                ),
                dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
            )

        new_attn = NEXUSAttention(
            model_config=cfg,
            nexus_cfg=nexus_cfg,
            layer_idx=layer_idx,
            total_layers=total_layers,
            pretrained_attn=src_attn,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if target is not None:
            if hasattr(target, "attention") and hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                if hasattr(new_attn, "dss_attn"):
                    new_attn.dss_attn.is_nested_encoder = True
                target.attention.self = new_attn
                return new_attn
            if hasattr(target, "attention"):
                new_attn.is_nested_encoder = False
                if hasattr(new_attn, "dss_attn"):
                    new_attn.dss_attn.is_nested_encoder = False
                target.attention = new_attn
                return new_attn
            if hasattr(target, "self_attn"):
                new_attn.is_nested_encoder = False
                target.self_attn = new_attn
                return new_attn
            if hasattr(target, "attn"):
                new_attn.is_nested_encoder = False
                target.attn = new_attn
                return new_attn
            if hasattr(target, "self"):
                new_attn.is_nested_encoder = True
                target.self = new_attn
                return new_attn
            target.attention = new_attn
            return new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_nexus failed: %s", e)
        return None


inject_nexus_attention = inject_nexus
inject_nexus_into_distilbert = inject_nexus

__all__ = [
    "NEXUSConfig",
    "NEXUSAttention",
    "DualStageStripeAttention",
    "SeerGating",
    "SpectraformerRectify",
    "XAttentionScorer",
    "softpick",
    "apply_rotary_pos_emb",
    "inject_nexus",
    "inject_nexus_attention",
]
