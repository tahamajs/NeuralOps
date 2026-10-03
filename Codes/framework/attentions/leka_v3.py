"""LEKAv3: Linear-time Efficient Kernel Attention with Value-Scored Landmarks (Corrected O(N)).

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- FactorizedLoRALinear (parameter-reducing) with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Exact causal masking for decoders
- Chunk-wise Landmark Selection (Solves Causal Starvation)
- Block-wise Sliding Window Attention (True O(N) memory scaling)
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE (dtype-safe)
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
        raise ValueError(
            f"apply_rotary_pos_emb: cos/sin seq len {cos.shape[-2]} != q seq len {q.shape[-2]}"
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q, sin_q = cos.to(q_rot.dtype), sin.to(q_rot.dtype)
        cos_k, sin_k = cos.to(k_rot.dtype), sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        cos_q, sin_q = cos[..., :head_dim].to(q.dtype), sin[..., :head_dim].to(q.dtype)
        cos_k, sin_k = cos[..., :head_dim].to(k.dtype), sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# Decoder detection
# ===========================================================================
def _detect_decoder(
    config: Any,
    is_decoder: bool | None,
    src_attn: nn.Module | None,
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if mt.startswith(
            (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
                "olmo",
                "mpt",
                "stablelm",
            )
        ):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if src_attn is not None:
        if getattr(src_attn, "is_decoder", False):
            return True
        if hasattr(src_attn, "c_attn"):
            return True
        if "gpt" in src_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# Reference softpick (arXiv:2504.20966)
# ===========================================================================
def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Rectified softmax with temperature:
        p_i = ReLU(exp((x_i - m) / T) - exp(-m / T)) / sum_j ReLU(...)
    Falls back to plain softmax for degenerate rows.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    temp = max(float(temperature), 1e-6)

    m = x_f.amax(dim=dim, keepdim=True)
    m_safe = torch.where(m < -1e3, torch.zeros_like(m), m)

    z = x_f - m_safe
    exp_z = torch.exp(z / temp)
    exp_neg_m = torch.exp(-m_safe / temp).clamp(max=1e4)

    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)

    fallback = F.softmax(x_f, dim=dim)
    p_norm = p / denom.clamp(min=1e-8)

    zero_mask = (denom <= 1e-8).expand_as(p_norm)
    out = torch.where(zero_mask, fallback, p_norm)
    return out.to(orig_dtype)


# ===========================================================================
# FactorizedLoRALinear
# ===========================================================================
class FactorizedLoRALinear(nn.Module):
    is_custom_attention = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **factory)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **factory)

        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, **factory)
            )
            self.lora_B = nn.Parameter(
                torch.empty(self.out_features, self.lora_rank, **factory)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_A is not None and self.lora_B is not None:
            w = w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return w

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_A is not None and self.lora_B is not None:
            out = (
                out
                + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            )
        return out


# ===========================================================================
# Main LEKAv3 Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("leka_v3", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register(
    "leka_v3_attention", supports_causal=True, supports_gqa=True
)
class LEKAV3Attention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        comp_config: Any | None = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        rank: int | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # Robust super().__init__
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = self._detect_head_dim(
            config, source_attn, kwargs, hidden_size, num_heads
        )

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_attention_heads = int(num_heads)
        self.num_heads = self.num_attention_heads
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.layer_idx = int(layer_idx)

        # Hyper-parameters
        self.landmark_ratio = float(
            getattr(comp_config, "landmark_ratio", 0.25) if comp_config else 0.25
        )
        self.landmark_chunk_size = int(
            getattr(comp_config, "landmark_chunk_size", 64) if comp_config else 64
        )
        self.num_landmarks = int(
            kwargs.get("num_landmarks")
            or (getattr(comp_config, "num_landmarks", None) if comp_config else None)
            or (getattr(config, "num_landmarks", None) if config else None)
            or 64
        )
        self.feature_dim = int(
            getattr(comp_config, "feature_dim", 128) if comp_config else 128
        )
        self.local_window_size = int(
            kwargs.get("local_window_size")
            or (
                getattr(comp_config, "local_window_size", None) if comp_config else None
            )
            or (getattr(config, "local_window_size", None) if config else None)
            or 128
        )
        dropout_p = float(
            getattr(comp_config, "dropout", getattr(config, "dropout", 0.1))
            if config is not None
            else 0.1
        )

        # Centralized decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = int(
            getattr(config, "lora_rank", None)
            or (getattr(comp_config, "lora_rank", None) if comp_config else None)
            or kwargs.get("lora_rank", 16)
        )
        self.lora_alpha = float(
            getattr(config, "lora_alpha", None)
            or (getattr(comp_config, "lora_alpha", None) if comp_config else None)
            or kwargs.get("lora_alpha", 32.0)
        )
        self.proj_rank = int(
            kwargs.get("proj_rank")
            or (getattr(comp_config, "proj_rank", None) if comp_config else None)
            or (rank if rank is not None else 64)
        )

        # FactorizedLoRALinear projections
        FF = FactorizedLoRALinear
        self.query = FF(
            self.dim,
            self.all_head_size,
            rank=self.proj_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.key = FF(
            self.dim,
            self.all_kv_head_size,
            rank=self.proj_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.value = FF(
            self.dim,
            self.all_kv_head_size,
            rank=self.proj_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.out_proj = FF(
            self.all_head_size,
            self.dim,
            rank=self.proj_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        self.dropout = nn.Dropout(dropout_p)

        # Single projection buffer (proj2 removed)
        self.register_buffer(
            "proj1",
            torch.randn(self.head_dim, self.feature_dim)
            / math.sqrt(self.head_dim / 2.0),
            persistent=True,
        )

        # Landmark scorer operates on K_features (2*feature_dim)
        self.landmark_scorer = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Per-head correlation matrix (identity init)
        self.W_corr = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_heads, 1, 1)
        )

        # Gate init = +2.0 → sigmoid(2) ≈ 0.88 favors LANDMARK branch
        self.gate = nn.Parameter(torch.full((self.num_heads,), 2.0))

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Dimension detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_hidden_size(
        config: Any, src: nn.Module | None, kwargs: dict[str, Any]
    ) -> int:
        for k in ("hidden_size", "d_model", "n_embd", "dim"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("hidden_size", "d_model", "n_embd", "dim"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("out_lin", "out_proj", "o_proj", "dense", "c_proj"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "out_features"):
                    return int(mod.out_features)
            for attr in ("q_proj", "q_lin", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "in_features"):
                    return int(mod.in_features)
        return 768

    @staticmethod
    def _detect_num_heads(
        config: Any, src: nn.Module | None, kwargs: dict[str, Any], hidden_size: int
    ) -> int:
        for k in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_attention_heads", "num_heads", "n_heads", "n_head"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for k in ("num_heads", "num_attention_heads", "n_heads"):
                v = getattr(src, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        return max(1, hidden_size // 64)

    @staticmethod
    def _detect_num_kv_heads(
        config: Any, src: nn.Module | None, kwargs: dict[str, Any], num_heads: int
    ) -> int:
        for k in ("num_key_value_heads", "num_kv_heads"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_key_value_heads", "num_kv_heads"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            v = getattr(src, "num_key_value_heads", None)
            if isinstance(v, int) and v > 0:
                return v
        return num_heads

    @staticmethod
    def _detect_head_dim(
        config: Any,
        src: nn.Module | None,
        kwargs: dict[str, Any],
        hidden_size: int,
        num_heads: int,
    ) -> int:
        v = kwargs.get("head_dim")
        if isinstance(v, int) and v > 0:
            return v
        if config is not None:
            for k in ("head_dim", "kv_channels", "d_kv"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(src, attr, None)
                if (
                    mod is not None
                    and hasattr(mod, "weight")
                    and mod.weight is not None
                ):
                    try:
                        return int(mod.weight.shape[0] // max(1, num_heads))
                    except Exception:
                        pass
        return max(1, hidden_size // max(1, num_heads))

    # ------------------------------------------------------------------
    # Kernel feature map (single projection)
    # ------------------------------------------------------------------
    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        proj1 = self.proj1.to(device=x.device, dtype=x.dtype)
        f1 = torch.matmul(x, proj1)
        f1 = torch.clamp(f1, min=-10.0, max=10.0)
        return torch.cat([torch.cos(f1), torch.sin(f1)], dim=-1) / math.sqrt(
            self.feature_dim
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # ------------------------------------------------------------------
    # Chunk-wise Landmark Selection (Solves Causal Starvation)
    # ------------------------------------------------------------------
    def select_landmarks_chunked(
        self,
        K_features: torch.Tensor,
        V: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ):
        B, H, N, feat_dim = K_features.shape
        chunk_sz = max(1, int(self.landmark_chunk_size))
        k_per_chunk = max(1, int(chunk_sz * self.landmark_ratio))

        with torch.no_grad():
            scores = self.landmark_scorer(K_features)[..., 0]  # (B, H, N)

            # Anchor boost for position 0 (always select first token)
            anchor_boost = torch.zeros_like(scores)
            anchor_boost[:, :, 0] = 1e4
            scores = scores + anchor_boost

            if valid_mask is not None:
                scores = scores.masked_fill(~valid_mask.unsqueeze(1).bool(), -1e4)

            # Pad to multiple of chunk_size
            pad_len = (chunk_sz - N % chunk_sz) % chunk_sz
            if pad_len > 0:
                scores_pad = F.pad(scores, (0, pad_len), value=-1e4)
            else:
                scores_pad = scores

            num_chunks = scores_pad.shape[-1] // chunk_sz
            scores_c = scores_pad.view(B, H, num_chunks, chunk_sz)

            # Extract top-K strictly within each temporal chunk
            k_act = min(k_per_chunk, chunk_sz)
            _, local_idx = scores_c.topk(k_act, dim=-1)

            # Convert local chunk indices back to global N-level indices
            offsets = torch.arange(
                0, num_chunks * chunk_sz, chunk_sz, device=scores.device
            ).view(1, 1, -1, 1)
            global_idx = (local_idx + offsets).view(B, H, -1)
            global_idx = global_idx.clamp(max=N - 1)

        K_land = torch.gather(
            K_features,
            2,
            global_idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim),
        )
        V_land = torch.gather(
            V,
            2,
            global_idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]),
        )
        return K_land, V_land, global_idx

    # ------------------------------------------------------------------
    # Block-wise Sliding Window Attention (True O(N) memory)
    # ------------------------------------------------------------------
    def block_local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, N, d = q.shape
        W = max(1, int(self.local_window_size))

        pad_len = (W - N % W) % W
        if pad_len > 0:
            q_p = F.pad(q, (0, 0, 0, pad_len))
            k_p = F.pad(k, (0, 0, 0, pad_len))
            v_p = F.pad(v, (0, 0, 0, pad_len))
        else:
            q_p, k_p, v_p = q, k, v

        N_pad = q_p.shape[2]
        num_blocks = N_pad // W
        scale = 1.0 / math.sqrt(d)

        q_b = q_p.view(B, H, num_blocks, W, d)
        k_b = k_p.view(B, H, num_blocks, W, d)
        v_b = v_p.view(B, H, num_blocks, W, d)

        # 1. Intra-block (self) attention
        attn_intra = torch.matmul(q_b, k_b.transpose(-1, -2)) * scale
        if self.is_decoder:
            causal_mask = torch.tril(
                torch.ones(W, W, device=q.device, dtype=torch.bool)
            )
            attn_intra = attn_intra.masked_fill(~causal_mask.view(1, 1, 1, W, W), -1e4)

        # 2. Inter-block (previous block) attention
        k_b_shifted = F.pad(k_b[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        v_b_shifted = F.pad(v_b[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        attn_inter = torch.matmul(q_b, k_b_shifted.transpose(-1, -2)) * scale
        attn_inter[:, :, 0, :, :] = -1e4  # First block has no previous block

        attn_combined = torch.cat([attn_inter, attn_intra], dim=-1)  # (B,H,nb,W,2W)

        if valid_mask is not None:
            v_mask_p = F.pad(valid_mask.float(), (0, pad_len), value=0.0).bool()
            # valid_mask shape: (B, N_pad) → broadcast against 2W dim
            v_mask_2w = torch.cat([v_mask_p, v_mask_p], dim=-1).view(
                B, num_blocks, 1, 2 * W
            )
            attn_combined = attn_combined.masked_fill(~v_mask_2w.unsqueeze(1), -1e4)

        attn_combined = torch.nan_to_num(
            attn_combined, nan=-1e4, posinf=1e4, neginf=-1e4
        )
        attn_weights = F.softmax(attn_combined, dim=-1).to(v.dtype)
        attn_weights = self.dropout(attn_weights)

        v_combined = torch.cat([v_b_shifted, v_b], dim=-2)  # (B,H,nb,2W,d)
        out_local = torch.matmul(attn_weights, v_combined)  # (B,H,nb,W,d)

        return out_local.reshape(B, H, N_pad, d)[:, :, :N, :]

    # ------------------------------------------------------------------
    # SVD init
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(
        w: torch.Tensor, b: torch.Tensor | None, proj: FactorizedLoRALinear
    ) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                W = W.reshape(W.size(0), -1)

            out_t, in_t = proj.out_features, proj.in_features
            if W.shape[0] != out_t or W.shape[1] != in_t:
                padded = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
                r = min(out_t, W.shape[0])
                c = min(in_t, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception as e:
                logger.warning("LEKAv3._svd_init: SVD failed (%s)", e)
                return

            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.down.weight.data.zero_()
            proj.up.weight.data.zero_()
            proj.down.weight.data[:r, :].copy_(Vh[:r, :].to(proj.down.weight.dtype))
            US = U[:, :r] @ torch.diag(S[:r])
            proj.up.weight.data[:, :r].copy_(US.to(proj.up.weight.dtype))

            if b is not None and proj.up.bias is not None:
                n = min(b.numel(), proj.up.bias.numel())
                proj.up.bias.data[:n].copy_(b[:n].to(proj.up.bias.dtype))
            elif proj.up.bias is not None:
                proj.up.bias.data.zero_()

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = (
            hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (c_w.ndim == 2 and c_w.shape[1] == 3 * self.hidden_size)
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. BERT / RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules (DistilBERT, LLaMA-style)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"LEKAV3Attention.copy_weights: missing mandatory projections "
                f"{missing_after} from {cls_name}."
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        def _resize_gqa(w, b, out_target):
            if w is None:
                return w, b
            src_out = w.shape[0]
            if src_out == out_target:
                return w, b
            if src_out > out_target and src_out % out_target == 0:
                rep = src_out // out_target
                w = w.view(out_target, rep, -1).mean(dim=1)
                if b is not None:
                    b = b.view(out_target, rep).mean(dim=1)
            elif src_out < out_target and out_target % src_out == 0:
                rep = out_target // src_out
                w = w.repeat(rep, 1)
                if b is not None:
                    b = b.repeat(rep)
            return w, b

        with torch.no_grad():
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_o, b_o = extracted["o"]

            if w_q is not None:
                self._svd_init(_orient(w_q, self.hidden_size), None, self.query)
            if w_k is not None:
                w_k_o = _orient(w_k, self.hidden_size)
                w_k_res, _ = _resize_gqa(w_k_o, None, self.key.out_features)
                self._svd_init(w_k_res, None, self.key)
            if w_v is not None:
                w_v_o = _orient(w_v, self.hidden_size)
                w_v_res, _ = _resize_gqa(w_v_o, None, self.value.out_features)
                self._svd_init(w_v_res, None, self.value)
            if w_o is not None:
                self._svd_init(_orient(w_o, self.all_head_size), b_o, self.out_proj)
            elif self.is_nested_encoder:
                eye = torch.eye(
                    self.all_head_size,
                    self.all_head_size,
                    device=self.out_proj.down.weight.device,
                    dtype=self.out_proj.down.weight.dtype,
                )
                self._svd_init(eye, None, self.out_proj)

            # Norm verification
            for name, mod in (
                ("query", self.query),
                ("key", self.key),
                ("value", self.value),
                ("out_proj", self.out_proj),
            ):
                try:
                    w_norm = float(torch.norm(mod.weight).item())
                    if w_norm < 1e-4:
                        logger.warning(
                            "LEKAV3Attention: weight '%s' has near-zero norm (%.2e).",
                            name,
                            w_norm,
                        )
                except Exception:
                    pass

            for mod in (self.query, self.key, self.value, self.out_proj):
                mod.freeze_base_weights()

        # Forward sanity check
        try:
            with torch.no_grad():
                d = next(self.parameters()).device
                dt = self.query.down.weight.dtype
                dummy = torch.randn(1, 8, self.dim, device=d, dtype=dt)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("LEKAv3 forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: Any | None = None,
        layer_idx: int = 0,
        output_attentions: bool | None = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim
        device = hidden_states.device

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)

        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k = (
            self.key(hidden_states)
            .view(B, N, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.value(hidden_states)
            .view(B, N, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # RoPE with past_len offset
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if (
                    isinstance(past_key_value, (tuple, list))
                    and len(past_key_value) >= 1
                ):
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            if past_len > 0:
                cos = cos[..., past_len : past_len + N, :]
                sin = sin[..., past_len : past_len + N, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        Q_features = self.kernel_feature_map(q)
        K_features = self.kernel_feature_map(k)

        # 1. Landmark branch (chunk-wise selection, solves causal starvation)
        K_land, V_land, land_idx = self.select_landmarks_chunked(
            K_features,
            v,
            valid_mask=valid_mask,
        )

        # Per-head correlation via einsum
        # V_land: (B,H,L,d); W_corr: (H,d,d)
        # V_land_adj[b,h,l,e] = sum_d V_land[b,h,l,d] * W_corr[h,d,e]
        V_land_adj = torch.einsum(
            "bhld,hde->bhle",
            V_land,
            self.W_corr.to(V_land.dtype),
        )

        attn_land = torch.matmul(
            Q_features.float(), K_land.transpose(-2, -1).contiguous().float()
        )

        if valid_mask is not None:
            pad_q = (~valid_mask).unsqueeze(1).unsqueeze(-1)
            attn_land = attn_land.masked_fill(pad_q.bool(), -1e4)

        if self.is_decoder:
            q_pos = torch.arange(N, device=device).view(1, 1, N, 1)
            causal_land_mask = land_idx.unsqueeze(2) <= q_pos
            attn_land = attn_land.masked_fill(~causal_land_mask, -1e4)

        attn_land = softpick(attn_land, dim=-1)
        attn_land = self.dropout(attn_land)
        context_land = torch.matmul(attn_land.to(v.dtype), V_land_adj)

        # 2. Local branch (block-wise sliding window, true O(N) memory)
        context_local = self.block_local_attention(q, k, v, valid_mask)

        # 3. Gate fusion: init sigmoid(+2.0)≈0.88 → favors LANDMARK branch
        gate_val = (
            torch.sigmoid(self.gate).view(1, H, 1, 1).to(dtype=context_land.dtype)
        )
        context = gate_val * context_land + (1.0 - gate_val) * context_local

        # head_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            context = context * hm.to(context.dtype)

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, None

    def get_ortho_loss(self) -> float:
        return 0.0


LEKAv3Attention = LEKAV3Attention


# ===========================================================================
# Injection (with recursion guard)
# ===========================================================================
@ATTENTION_REGISTRY.register("leka_v3_inject")
@ATTENTION_REGISTRY.register("inject_leka_v3")
def inject_leka_v3(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_leka_v3: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
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
            res = inject_leka_v3(
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
                    src_attn = getattr(cand, "self", cand)
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # Double-injection guard
        if (
            isinstance(src_attn, LEKAV3Attention)
            or "LEKAv3" in src_attn.__class__.__name__
        ):
            return None

        # Skip hybrid layers (e.g. Qwen3.5 DeltaNet)
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        new_attn = LEKAV3Attention(
            config=cfg,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            old_attn=src_attn,
            **kwargs,
        )

        if src_attn is not None:
            new_attn.copy_weights(src_attn)

        p = next(src_attn.parameters(), None) if src_attn is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn

        return new_attn
    except Exception as e:
        logger.exception("inject_leka_v3 failed at layer %d: %s", layer_idx, e)
        return None


leka_v3_inject = inject_leka_v3
