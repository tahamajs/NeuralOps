"""
LEKAv3: Linear-time Efficient Kernel Attention with Value-Scored Landmarks.

"""

from __future__ import annotations

from neuralops.core.attention_utils import (
    safe_mask_value as _safe_mask_value,
    apply_rotary_pos_emb,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)




# ===========================================================================
# Decoder detection  — [FIX-14]
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
            ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
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
# Reference softpick  — [FIX-12]
# ===========================================================================
def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Rectified softmax with temperature.

    p_i = ReLU(exp((x_i - m)/T) - exp(-m/T)) / sum_j ReLU(...)
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
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

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
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


# ===========================================================================
# Main LEKAv3 Attention
# ===========================================================================
class RefinedLandmarkAttention(BaseAttention):
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

        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = self._detect_head_dim(config, source_attn, kwargs, hidden_size, num_heads)

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

        # ---- Hyper-parameters ----
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
            or (getattr(comp_config, "local_window_size", None) if comp_config else None)
            or (getattr(config, "local_window_size", None) if config else None)
            or 128
        )
        dropout_p = float(
            getattr(comp_config, "dropout", getattr(config, "dropout", 0.1))
            if config is not None else 0.1
        )

        # ---- [FIX-14] Decoder detection ----
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- LoRA ----
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

        # ---- Projections ----
        FF = FactorizedLoRALinear
        self.query = FF(
            self.dim, self.all_head_size,
            rank=self.proj_rank, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, bias=True,
        )
        self.key = FF(
            self.dim, self.all_kv_head_size,
            rank=self.proj_rank, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, bias=True,
        )
        self.value = FF(
            self.dim, self.all_kv_head_size,
            rank=self.proj_rank, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, bias=True,
        )
        self.out_proj = FF(
            self.all_head_size, self.dim,
            rank=self.proj_rank, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, bias=True,
        )

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        self.dropout = nn.Dropout(dropout_p)

        # ---- Kernel feature projection ----
        self.register_buffer(
            "proj1",
            torch.randn(self.head_dim, self.feature_dim)
            / math.sqrt(self.head_dim / 2.0),
            persistent=True,
        )

        # ---- [FIX-2] Landmark scorer with lora_ prefix ----
        self.lora_landmark_scorer = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Per-head correlation matrix (identity init)
        self.W_corr = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_heads, 1, 1)
        )

        # ---- [FIX-3] Gate with lora_ prefix ----
        # sigmoid(+2.0) ≈ 0.88 → favors LANDMARK branch
        self.lora_gate = nn.Parameter(torch.full((self.num_heads,), 2.0))

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Dimension detectors
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
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
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
        return torch.cat([torch.cos(f1), torch.sin(f1)], dim=-1) / math.sqrt(self.feature_dim)

    # ------------------------------------------------------------------
    # [FIX-4] Chunk-wise Landmark Selection (decode-safe)
    # ------------------------------------------------------------------
    def select_landmarks_chunked(
        self,
        K_features: torch.Tensor,
        V: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k landmarks per chunk.

        [FIX-4] Uses N_k (key length) not N_q — critical for decode mode.
        [FIX-15] Safe mask broadcasting.

        Args:
            K_features: (B, H, N_k, feat_dim)
            V: (B, H, N_k, d)
            valid_mask: (B, N_k) boolean

        Returns:
            K_land: (B, H, L, feat_dim)
            V_land: (B, H, L, d)
            global_idx: (B, H, L) — absolute positions
        """
        B, H, N_k, feat_dim = K_features.shape
        chunk_sz = max(1, int(self.landmark_chunk_size))
        k_per_chunk = max(1, int(chunk_sz * self.landmark_ratio))

        with torch.no_grad():
            # [FIX-2] use lora_-prefixed scorer
            scores = self.lora_landmark_scorer(K_features)[..., 0]  # (B, H, N_k)

            # Anchor boost for token 0
            anchor_boost = torch.zeros_like(scores)
            if N_k > 0:
                anchor_boost[:, :, 0] = 1e4
            scores = scores + anchor_boost

            # [FIX-15] Safe valid_mask handling
            if valid_mask is not None:
                vm = valid_mask
                while vm.dim() < 3:
                    vm = vm.unsqueeze(1)
                vm = vm.expand(-1, H, -1)  # (B, H, N_k)
                scores = scores.masked_fill(~vm.bool(), -1e4)

            # Pad to multiple of chunk_size
            pad_len = (chunk_sz - N_k % chunk_sz) % chunk_sz
            if pad_len > 0:
                scores_pad = F.pad(scores, (0, pad_len), value=-1e4)
            else:
                scores_pad = scores

            num_chunks = scores_pad.shape[-1] // chunk_sz
            scores_c = scores_pad.view(B, H, num_chunks, chunk_sz)

            k_act = min(k_per_chunk, chunk_sz)
            _, local_idx = scores_c.topk(k_act, dim=-1)

            offsets = torch.arange(
                0, num_chunks * chunk_sz, chunk_sz, device=scores.device
            ).view(1, 1, -1, 1)
            global_idx = (local_idx + offsets).view(B, H, -1)
            global_idx = global_idx.clamp(min=0, max=max(0, N_k - 1))

        K_land = torch.gather(
            K_features, 2,
            global_idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim),
        )
        V_land = torch.gather(
            V, 2,
            global_idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]),
        )
        return K_land, V_land, global_idx

    # ------------------------------------------------------------------
    # [FIX-5, FIX-16] Block-wise sliding window
    # ------------------------------------------------------------------
    def block_local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block-wise sliding window attention.

        Requires N_q == N_k. For decode mode, caller must use the exact
        fallback in `forward`.
        """
        B, H, N, d = q.shape
        N_k = k.shape[2]
        if N != N_k:
            raise ValueError(
                f"block_local_attention requires N_q == N_k, got {N} vs {N_k}"
            )

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

        # 1. Intra-block attention
        attn_intra = torch.matmul(q_b, k_b.transpose(-1, -2)) * scale
        if self.is_decoder:
            causal_mask = torch.tril(
                torch.ones(W, W, device=q.device, dtype=torch.bool)
            )
            attn_intra = attn_intra.masked_fill(
                ~causal_mask.view(1, 1, 1, W, W), -1e4
            )

        # 2. Inter-block (previous block) attention
        k_b_shifted = F.pad(k_b[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        v_b_shifted = F.pad(v_b[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        attn_inter = torch.matmul(q_b, k_b_shifted.transpose(-1, -2)) * scale
        # First block has no previous block
        if num_blocks > 0:
            attn_inter = attn_inter.clone()
            attn_inter[:, :, 0, :, :] = -1e4

        attn_combined = torch.cat([attn_inter, attn_intra], dim=-1)  # (B, H, nb, W, 2W)

        if valid_mask is not None:
            v_mask_p = F.pad(
                valid_mask.float(), (0, pad_len), value=0.0
            ).bool()  # (B, N_pad)
            v_mask_2w = torch.cat([v_mask_p, v_mask_p], dim=-1).view(
                B, num_blocks, 1, 2 * W
            )
            attn_combined = attn_combined.masked_fill(~v_mask_2w.unsqueeze(1), -1e4)

        attn_combined = torch.nan_to_num(
            attn_combined, nan=-1e4, posinf=1e4, neginf=-1e4
        )
        attn_weights = F.softmax(attn_combined, dim=-1).to(v.dtype)
        attn_weights = self.dropout(attn_weights)

        v_combined = torch.cat([v_b_shifted, v_b], dim=-2)  # (B, H, nb, 2W, d)
        out_local = torch.matmul(attn_weights, v_combined)   # (B, H, nb, W, d)

        return out_local.reshape(B, H, N_pad, d)[:, :, :N, :]

    # ------------------------------------------------------------------
    # [FIX-5] Exact decode fallback
    # ------------------------------------------------------------------
    def _exact_decode_local(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """[FIX-6] Exact sliding-window attention for decode (N_q < N_k)."""
        B, H, N_q, d = q.shape
        N_k = k.shape[2]
        W = max(1, int(self.local_window_size))
        scale = 1.0 / math.sqrt(d)
        device = q.device

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        q_abs = (torch.arange(N_q, device=device) + past_len).unsqueeze(1)
        k_abs = torch.arange(N_k, device=device).unsqueeze(0)

        window_mask = (q_abs - k_abs).abs() <= W
        if self.is_decoder:
            causal_mask = k_abs <= q_abs
            valid = window_mask & causal_mask
        else:
            valid = window_mask

        mask_val = _safe_mask_value(scores.dtype)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), mask_val)

        if valid_mask is not None:
            kv_valid = valid_mask.to(device=device, dtype=torch.bool)
            scores = scores.masked_fill(~kv_valid[:, None, None, :], mask_val)

        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v.float()).to(q.dtype)
        return out

    # ------------------------------------------------------------------
    # KV-cache helpers  — [FIX-1]
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(
        past_key_value: Any, layer_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        if past_key_value is None or past_key_value is False:
            return None, None, 0

        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            kc, vc = past_key_value.key_cache, past_key_value.value_cache
            if len(kc) == 0 or layer_idx >= len(kc):
                return None, None, 0
            k = kc[layer_idx]
            v = vc[layer_idx] if layer_idx < len(vc) else None
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]

        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]

        return None, None, 0

    @staticmethod
    def _write_past_kv(
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Any,
        layer_idx: int,
    ) -> Any:
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                return past_key_value
            except Exception:
                pass
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if layer_idx < len(past_key_value.value_cache):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                past_key_value.value_cache.append(v)
            return past_key_value
        return (k, v)

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
            (hasattr(src, "self") and hasattr(src.self, "query")
             and hasattr(src, "output") and hasattr(src.output, "dense"))
            or (cls_name in [
                "BertSelfAttention", "RobertaSelfAttention",
                "XLMRobertaSelfAttention", "ElectraSelfAttention",
                "DebertaSelfAttention",
            ] or (cls_name.endswith("SelfAttention")
                  and "distil" not in cls_name.lower()))
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
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
                    c_w[split_size:2 * split_size, :],
                    c_b[split_size:2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size:, :],
                    c_b[2 * split_size:] if c_b is not None else None,
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

        # 3. Flat modules
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
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"RefinedLandmarkAttention.copy_weights: missing mandatory projections "
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

            for name, mod in (
                ("query", self.query), ("key", self.key),
                ("value", self.value), ("out_proj", self.out_proj),
            ):
                try:
                    w_norm = float(torch.norm(mod.weight).item())
                    if w_norm < 1e-4:
                        logger.warning(
                            "RefinedLandmarkAttention: weight '%s' near-zero norm (%.2e).",
                            name, w_norm,
                        )
                except Exception:
                    pass

            for mod in (self.query, self.key, self.value, self.out_proj):
                mod.freeze_base_weights()

        # ---- Inherit decoder flag ----
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            self.is_causal = self.is_decoder

    # ------------------------------------------------------------------
    # Forward  — [FIX-1, FIX-5, FIX-6]
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
    ) -> tuple[torch.Tensor, Any]:
        # Unwrap tuple input
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ---- Projections ----
        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k_new = (
            self.key(hidden_states)
            .view(B, N, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        v_new = (
            self.value(hidden_states)
            .view(B, N, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- [FIX-1] Read past KV ----
        past_k, past_v, past_len = self._read_past_kv(past_key_value, self.layer_idx)

        # ---- RoPE with past offset ----
        pos_emb = (
            position_embeddings if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        pos_ids = kwargs.get("position_ids")

        if pos_emb is not None:
            cos, sin = pos_emb
            if past_len > 0:
                # Slice for absolute positions
                if cos.dim() == 4:  # (B, H, S, d)
                    cos_q = cos[..., past_len:past_len + N, :]
                    sin_q = sin[..., past_len:past_len + N, :]
                elif cos.dim() == 3:  # (B, S, d)
                    cos_q = cos[:, past_len:past_len + N, :].unsqueeze(1)
                    sin_q = sin[:, past_len:past_len + N, :].unsqueeze(1)
                else:
                    cos_q = cos[past_len:past_len + N, :].unsqueeze(0).unsqueeze(0)
                    sin_q = sin[past_len:past_len + N, :].unsqueeze(0).unsqueeze(0)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos_q, sin_q)
            else:
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                cos, sin = self.rotary_emb(k_new, position_ids=pos_ids)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
            except Exception as e:
                logger.debug("rotary_emb failed: %s", e)

        # ---- [FIX-1] Concat KV cache ----
        if past_k is not None:
            K = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
            V = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
        else:
            K, V = k_new, v_new

        # ---- [FIX-1] Write back cache ----
        present_kv: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache:
            present_kv = self._write_past_kv(K, V, past_key_value, self.layer_idx)
        else:
            present_kv = past_key_value

        N_kv = K.shape[-2]

        # ---- GQA repeat ----
        if self.kv_repeat > 1:
            K_full = repeat_kv(K, self.kv_repeat)
            V_full = repeat_kv(V, self.kv_repeat)
        else:
            K_full, V_full = K, V

        # ---- Kernel features ----
        Q_features = self.kernel_feature_map(q)
        K_features = self.kernel_feature_map(K_full)

        # ---- Valid mask ----
        valid_mask: torch.Tensor | None = None
        if attention_mask is not None:
            valid_mask = get_token_valid_mask(attention_mask, B, N_kv, device)

        # ============================================================
        # 1. Landmark branch  — [FIX-4, FIX-6]
        # ============================================================
        K_land, V_land, land_idx = self.select_landmarks_chunked(
            K_features, V_full, valid_mask=valid_mask,
        )

        # Per-head correlation
        V_land_adj = torch.einsum(
            "bhld,hde->bhle", V_land, self.W_corr.to(V_land.dtype)
        )

        attn_land = torch.matmul(
            Q_features.float(), K_land.transpose(-2, -1).contiguous().float()
        )

        # [FIX-6] Causal mask using absolute positions
        if self.is_decoder:
            q_abs = (
                torch.arange(past_len, past_len + N, device=device)
                .view(1, 1, N, 1)
            )
            # land_idx contains absolute positions
            causal_land_mask = land_idx.unsqueeze(2) <= q_abs
            attn_land = attn_land.masked_fill(~causal_land_mask, -1e4)

        attn_land = softpick(attn_land, dim=-1)
        attn_land = self.dropout(attn_land)
        context_land = torch.matmul(attn_land.to(V_full.dtype), V_land_adj)

        # ============================================================
        # 2. Local branch  — [FIX-5]
        # ============================================================
        if N == N_kv:
            # Prefill mode: block-wise valid
            context_local = self.block_local_attention(
                q, K_full, V_full, valid_mask
            )
        else:
            # Decode mode: exact fallback
            context_local = self._exact_decode_local(
                q, K_full, V_full, past_len, valid_mask
            )

        # ============================================================
        # 3. Gate fusion  — [FIX-3]
        # ============================================================
        gate_val = (
            torch.sigmoid(self.lora_gate)
            .view(1, H, 1, 1)
            .to(dtype=context_land.dtype)
        )
        context = gate_val * context_land + (1.0 - gate_val) * context_local

        # ---- Head mask ----
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

        # ---- Output projection ----
        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- [FIX-1, FIX-13] Return format ----
        if self.is_decoder:
            if output_attentions:
                return out.to(dtype), present_kv, attn_land
            return out.to(dtype), present_kv

        return out.to(dtype), None

    def get_ortho_loss(self) -> float:
        return 0.0


# Backward-compat aliases
LEKAv3Attention = RefinedLandmarkAttention
LEKAV3Attention = RefinedLandmarkAttention


# ===========================================================================
# [FIX-10, FIX-11] Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("inject_leka_v3")
def inject_refined_landmark(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_refined_landmark: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_refined_landmark(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None:
        return None

    # [FIX-11] Idempotency
    if (
        isinstance(src_attn, RefinedLandmarkAttention)
        or "RefinedLandmark" in src_attn.__class__.__name__
        or "LEKAv3" in src_attn.__class__.__name__
    ):
        return None

    # Skip hybrid layers
    c_name = src_attn.__class__.__name__
    if "DeltaNet" in c_name or "LinearAttention" in c_name:
        return None

    if cfg is None and hasattr(src_attn, "config"):
        cfg = src_attn.config

    try:
        new_attn = RefinedLandmarkAttention(
            config=cfg,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            old_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.exception("RefinedLandmarkAttention construction failed: %s", e)
        return None

    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error("copy_weights failed: %s", e)
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    # ---- Splice ----
    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True

    if not spliced:
        logger.error(
            "inject_refined_landmark: no writable attention attribute on %s",
            type(target).__name__,
        )
        return None

    return new_attn


# ===========================================================================
# [FIX-13] Exports
# ===========================================================================
__all__ = [
    "RefinedLandmarkAttention",
    "LEKAv3Attention",
    "LEKAV3Attention",
    "FactorizedLoRALinear",
    "inject_refined_landmark",
    "apply_rotary_pos_emb",
    "softpick",
]