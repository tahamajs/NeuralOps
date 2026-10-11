"""
XAttention: Block Sparse Attention with Coarse-to-Fine Routing.

Partitions sequence tokens into non-overlapping blocks, evaluates coarse
block-level routing, and computes unified normalized fine-grained attention
across top-k candidate blocks, with multi-architecture weight transfer,
trainable LoRA residual adapters, GQA, RoPE, and causal decoding.

Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.

Fixes applied:
    [XATTN-FIX-01]  lora_ prefix on output_scale (KD optimizer scope)
    [XATTN-FIX-02]  Property alias for backward compatibility
    [XATTN-FIX-03]  Full KV-cache / past_key_value support
    [XATTN-FIX-04]  RoPE with past_len slicing for incremental decode
    [XATTN-FIX-05]  Safe nn.Identity handling in copy_weights (nested encoder)
    [XATTN-FIX-06]  GQA-aware mean-pooling during weight copy
    [XATTN-FIX-07]  Mask-aware block-averaging for padding tokens
    [XATTN-FIX-08]  Causal mask with correct past_len offset
    [XATTN-FIX-09]  Idempotency + recursion guards in injector
    [XATTN-FIX-10]  Explicit NotImplementedError for cross-attention
    [XATTN-FIX-11]  Triton path with robust fallback
    [XATTN-FIX-12]  Frobenius norm verification on weight copy
    [XATTN-FIX-13]  Single-token decode fast path via dense attention
    [XATTN-FIX-14]  __all__ declaration
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

try:
    from neuralops.core.triton_sparse_attn import (
        triton_block_sparse_attention,
        HAS_TRITON,
    )
except Exception:
    triton_block_sparse_attention = None
    HAS_TRITON = False

logger = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _extract_valid_token_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return a boolean validity mask (B, seq_len) where True = valid token.

    Handles the many shapes HF emits: (B, L), (B, 1, 1, L), (B, 1, L, L).
    """
    if mask is None or not isinstance(mask, torch.Tensor):
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

    mask = mask.to(device=device)

    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            return mask
        if mask.dtype.is_floating_point:
            if (mask < -1.0).any():
                return mask >= -1.0
            return mask > 0.5
        return mask > 0

    if mask.dim() == 4:
        # (B, 1, 1, N) padding  or  (B, 1, N, N) attention bias
        if mask.size(2) == 1:
            m = mask[:, 0, 0, :]
        else:
            m = mask[:, 0, -1, :]
        if m.dtype == torch.bool:
            return m
        if m.dtype.is_floating_point:
            if (m < -1.0).any():
                return m >= -1.0
            if (m <= 0.0).all():
                return torch.ones(
                    (batch_size, seq_len), dtype=torch.bool, device=device
                )
            return m > 0.5
        return m > 0

    return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    past_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE with past_len slicing for incremental decoding.

    Handles partial-RoPE (rotary_dim < head_dim) and any cos/sin shape.
    """
    if cos is None or sin is None:
        return q, k

    L = q.shape[-2]

    # Slice the correct window when cos/sin cover a longer range
    if cos.shape[-2] > past_len + L:
        cos = cos[..., past_len : past_len + L, :]
        sin = sin[..., past_len : past_len + L, :]

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_cut = cos[..., :rotary_dim].to(q.dtype)
    sin_cut = sin[..., :rotary_dim].to(q.dtype)

    q_rot_e = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_e = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_e, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_e
    k = torch.cat([k_rot_e, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_e

    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

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
        fk = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (
            (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        )
        self.freeze_base = bool(freeze_base)

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **fk)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **fk)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **fk)
            )
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
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
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
        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            out = (
                out
                + F.linear(F.linear(x, self.lora_A), self.lora_B)
                * self.lora_scaling
            )
        return out


# ===========================================================================
# XAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "xattention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class XAttention(BaseAttention):
    """
    Block-sparse attention with coarse-to-fine routing.

    Pipeline:
        1. Project Q, K, V (with LoRA)
        2. Apply RoPE (with past_len slicing)
        3. Concatenate KV-cache
        4. If N_k is small or keep_ratio == 1 → dense attention
           Otherwise → block-level coarse routing → top-k blocks
        5. Fine-grained attention within selected blocks
        6. Optional Triton kernel acceleration
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "sparse_attn"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        block_size: int = 32,
        sparsity: float = 0.5,
        lora_rank: int = 16,
        pretrained_attn: nn.Module | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- Resolve dims ----
        d_model = (
            kwargs.get("hidden_size")
            or _cfg_get(config, "hidden_size")
            or _cfg_get(config, "d_model")
            or _cfg_get(config, "dim")
            or 768
        )
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        num_heads = (
            kwargs.get("num_heads")
            or _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "num_heads")
            or _cfg_get(config, "n_heads")
            or 12
        )
        num_kv_heads = (
            kwargs.get("num_kv_heads")
            or _cfg_get(config, "num_key_value_heads")
            or _cfg_get(config, "num_kv_heads")
            or num_heads
        )

        self.hidden_size = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))
        self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- Decoder detection ----
        mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _dec_pfx = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        is_dec = (
            bool(_cfg_get(config, "is_decoder", False))
            or bool(_cfg_get(config, "is_causal", False))
            or any(mt.startswith(p) for p in _dec_pfx)
            or kwargs.get("is_decoder", False)
            or kwargs.get("is_causal", False)
        )
        if pretrained_attn is not None:
            if getattr(pretrained_attn, "is_decoder", False) or hasattr(
                pretrained_attn, "c_attn"
            ):
                is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_nested_encoder: bool = False
        self.layer_idx = int(
            layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0)
        )

        # ---- Block config ----
        self.block_size: int = max(
            0,
            int(
                kwargs.get("block_size", _cfg_get(config, "xattn_block_size", block_size))
            ),
        )
        raw_keep = kwargs.get("keep_ratio", _cfg_get(config, "xattn_keep_ratio", None))
        if raw_keep is not None:
            self.keep_ratio: float = float(raw_keep)
            self.sparsity: float = 1.0 - self.keep_ratio
        else:
            self.sparsity = float(
                kwargs.get("sparsity", _cfg_get(config, "xattn_sparsity", sparsity))
            )
            self.keep_ratio = 1.0 - self.sparsity
        self.keep_ratio = max(0.0, min(1.0, self.keep_ratio))

        # ---- LoRA config ----
        self.lora_rank = int(
            kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        )

        # ---- Target dtype ----
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and pretrained_attn is not None:
            try:
                target_dtype = next(pretrained_attn.parameters()).dtype
            except StopIteration:
                target_dtype = torch.float32
        if target_dtype is None:
            target_dtype = torch.float32

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.out_proj: nn.Module = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )

        # ---- [XATTN-FIX-01] KD-compatible output scale ----
        self.lora_output_scale = nn.Parameter(
            torch.ones(1, dtype=target_dtype)
        )

        # ---- Dropout ----
        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)
        self.triton_kwargs = {"block_size": self.block_size, "eps": 1e-6}

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        self.to(target_dtype)

    # ------------------------------------------------------------------
    # [XATTN-FIX-02] Backward-compat aliases
    # ------------------------------------------------------------------
    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    @property
    def W_q(self): return self.q_proj
    @property
    def W_k(self): return self.k_proj
    @property
    def W_v(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # [XATTN-FIX-05, XATTN-FIX-06, XATTN-FIX-12] Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        """Extract Q/K/V/O weights across multiple architectures."""
        if src is None:
            return

        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # ---- 1. UniversalAdapter extraction ----
        try:
            adapter = UniversalAdapter()
            for proj in ["q", "k", "v", "o"]:
                try:
                    w, b = adapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted[proj] = (w, b)
                except Exception as e:
                    logger.debug("UniversalAdapter '%s' extraction failed: %s", proj, e)
        except Exception as e:
            logger.debug("UniversalAdapter init failed: %s", e)

        # ---- 2. GPT-2 fused c_attn (Conv1D) ----
        if hasattr(src, "c_attn") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            c_attn = getattr(src.c_attn, "base_layer", src.c_attn)
            weight = c_attn.weight.data
            bias = getattr(c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None

            is_conv1d = "Conv1D" in c_attn.__class__.__name__ or hasattr(c_attn, "nf")
            if is_conv1d or (
                weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size
            ):
                weight = weight.t().contiguous()

            split_size = weight.shape[0] // 3
            extracted["q"] = (
                weight[:split_size, :],
                bias_data[:split_size] if bias_data is not None else None,
            )
            extracted["k"] = (
                weight[split_size : 2 * split_size, :],
                bias_data[split_size : 2 * split_size] if bias_data is not None else None,
            )
            extracted["v"] = (
                weight[2 * split_size :, :],
                bias_data[2 * split_size :] if bias_data is not None else None,
            )

        # ---- 3. GPT-2 c_proj ----
        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj = getattr(src.c_proj, "base_layer", src.c_proj)
            w_o = c_proj.weight.data
            is_conv1d = "Conv1D" in c_proj.__class__.__name__ or hasattr(c_proj, "nf")
            if is_conv1d or (
                w_o.dim() == 2
                and w_o.shape[0] != self.hidden_size
                and w_o.shape[1] == self.hidden_size
            ):
                w_o = w_o.t().contiguous()
            b_o = getattr(c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # ---- 4. BERT / RoBERTa nested encoder ----
        if hasattr(src, "self") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            s = src.self
            for proj_key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                if hasattr(s, sub):
                    mod = getattr(s, sub)
                    base = getattr(mod, "base_layer", mod)
                    if hasattr(base, "weight"):
                        b = getattr(base, "bias", None)
                        extracted[proj_key] = (
                            base.weight.data,
                            b.data if b is not None else None,
                        )

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = getattr(src.output.dense, "base_layer", src.output.dense)
            b = getattr(dense, "bias", None)
            extracted["o"] = (dense.weight.data, b.data if b is not None else None)

        # ---- 5. Flat projections ----
        named_map = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        layer = getattr(layer, "base_layer", layer)
                        if hasattr(layer, "weight"):
                            b = getattr(layer, "bias", None)
                            extracted[proj] = (
                                layer.weight.data,
                                b.data if b is not None else None,
                            )
                            break

        # ---- Detect nested encoder ----
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense", "out_proj", "c_proj", "out_lin", "o_proj",
                "wo", "o", "proj", "out",
            )
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        # ---- [XATTN-FIX-06] GQA-aware mean-pooling ----
        def _align_kv(
            w: torch.Tensor,
            b: torch.Tensor | None,
            target_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if w.shape[0] == target_size:
                return w, b
            if w.shape[0] > target_size and w.shape[0] % target_size == 0:
                rep = w.shape[0] // target_size
                w_pooled = (
                    w.view(target_size, rep, -1).mean(dim=1).contiguous()
                )
                b_pooled = (
                    b.view(target_size, rep).mean(dim=1).contiguous()
                    if b is not None
                    else None
                )
                return w_pooled, b_pooled
            return w[:target_size, :], (b[:target_size] if b is not None else None)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # ---- Copy into projections ----
        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                # [XATTN-FIX-05] Skip output projection for nested encoders
                if proj_key == "o" and self.is_nested_encoder:
                    self.out_proj = nn.Identity()
                    continue

                w, b = extracted[proj_key]
                if w is None:
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                in_dim = (
                    self.all_head_size if proj_key == "o" else self.hidden_size
                )
                w = _orient(w, in_dim)

                # GQA mean-pooling for K and V
                if proj_key in ("k", "v"):
                    w, b = _align_kv(w, b, target.weight.size(0))

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))

                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        device=target.weight.device,
                        dtype=target.weight.dtype,
                    )
                )

                if b is not None and target.bias is not None:
                    b_data = b.data if isinstance(b, nn.Parameter) else b
                    mb = min(target.bias.numel(), b_data.numel())
                    target.bias.data[:mb].copy_(
                        b_data[:mb].to(
                            device=target.bias.device,
                            dtype=target.bias.dtype,
                        )
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                # [XATTN-FIX-12] Frobenius verification
                w_norm = float(
                    torch.linalg.norm(target.weight.data.float()).item()
                )
                if w_norm < 1e-4:
                    logger.warning(
                        "XAttention copy_weights: '%s' near-zero norm (%.2e)",
                        proj_key,
                        w_norm,
                    )

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        # [XATTN-FIX-10] Explicit rejection of cross-attention
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        if key_value_states is not None:
            raise NotImplementedError(
                "XAttention does not support cross-attention."
            )

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N_q, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ---- [XATTN-FIX-03] Read past KV cache ----
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        past_len = 0

        if past_key_value is not None and past_key_value is not False:
            if isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 2 and isinstance(
                    past_key_value[0], torch.Tensor
                ):
                    past_k, past_v = past_key_value[0], past_key_value[1]
                    past_len = past_k.shape[-2] if past_k is not None else 0
            elif hasattr(past_key_value, "key_cache") and hasattr(
                past_key_value, "value_cache"
            ):
                if len(past_key_value.key_cache) > self.layer_idx:
                    past_k = past_key_value.key_cache[self.layer_idx]
                    past_v = past_key_value.value_cache[self.layer_idx]
                    past_len = past_k.shape[-2] if past_k is not None else 0

        # ---- Projections ----
        Q = (
            self.q_proj(hidden_states)
            .view(B, N_q, H, d)
            .transpose(1, 2)
            .contiguous()
        )
        K_new = (
            self.k_proj(hidden_states)
            .view(B, N_q, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V_new = (
            self.v_proj(hidden_states)
            .view(B, N_q, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- [XATTN-FIX-04] RoPE with past_len ----
        pos_emb = (
            kwargs.get("position_embeddings")
            if "position_embeddings" in kwargs
            else None
        )
        if (
            pos_emb is not None
            and isinstance(pos_emb, (tuple, list))
            and len(pos_emb) >= 2
        ):
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K_new = apply_rotary_pos_emb(
                Q, K_new, cos, sin,
                position_ids=kwargs.get("position_ids"),
                past_len=past_len,
            )
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                position_ids = kwargs.get("position_ids")
                if position_ids is None and past_len > 0:
                    position_ids = (
                        torch.arange(
                            past_len, past_len + N_q, device=device
                        )
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                cos, sin = self.rotary_emb(K_new, seq_len=past_len + N_q)
                Q, K_new = apply_rotary_pos_emb(
                    Q, K_new, cos, sin,
                    position_ids=position_ids,
                    past_len=past_len,
                )
            except Exception as e:
                logger.debug("XAttention rotary_emb failed: %s", e)

        # ---- Concatenate with cache ----
        if past_k is not None and past_v is not None:
            K_cat = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V_cat = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K_cat, V_cat = K_new, V_new

        # ---- Update cache ----
        new_past_key_value: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache:
            if hasattr(past_key_value, "update") and past_key_value is not None:
                try:
                    past_key_value.update(K_new, V_new, self.layer_idx)
                    new_past_key_value = past_key_value
                except Exception:
                    new_past_key_value = (K_cat, V_cat)
            else:
                new_past_key_value = (K_cat, V_cat)

        N_k = K_cat.shape[-2]

        # ---- GQA expansion ----
        if self.kv_repeat > 1:
            K_full = repeat_kv(K_cat, self.kv_repeat)
            V_full = repeat_kv(V_cat, self.kv_repeat)
        else:
            K_full, V_full = K_cat, V_cat

        scale = d**-0.5
        attn: torch.Tensor | None = None
        context: torch.Tensor | None = None

        # ---- Decide path: dense vs block-sparse ----
        use_dense = (
            self.block_size <= 0
            or self.keep_ratio >= 1.0
            or N_k <= self.block_size
            or N_q <= self.block_size
            or N_q == 1  # generation fast path
        )

        # ---- Fine-grained validity mask ----
        valid_m = _extract_valid_token_mask(
            attention_mask, B, N_k, device
        ) if attention_mask is not None else None

        if use_dense:
            # =============================================================
            # DENSE PATH (short sequences / generation)
            # =============================================================
            scores = torch.matmul(Q, K_full.transpose(-2, -1).contiguous()) * scale

            if self.is_decoder and N_q > 1:
                q_pos = (
                    past_len + torch.arange(N_q, device=device)
                ).unsqueeze(1)
                k_pos = torch.arange(N_k, device=device).unsqueeze(0)
                causal = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~causal, -1e4)

            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    scores = scores + attention_mask.to(scores.dtype)
                elif valid_m is not None:
                    scores = scores.masked_fill(
                        ~valid_m.unsqueeze(1).unsqueeze(2), -1e4
                    )

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores.float(), dim=-1).to(dtype)
            attn = self.dropout(attn)
            context = torch.matmul(attn, V_full)

        else:
            # =============================================================
            # BLOCK-SPARSE PATH (long sequences)
            # =============================================================
            scores = torch.matmul(Q, K_full.transpose(-2, -1).contiguous()) * scale

            bsz = self.block_size
            pad_q = (bsz - (N_q % bsz)) % bsz
            pad_k = (bsz - (N_k % bsz)) % bsz

            Q_pad = F.pad(Q, (0, 0, 0, pad_q)) if pad_q > 0 else Q
            K_pad = F.pad(K_full, (0, 0, 0, pad_k)) if pad_k > 0 else K_full

            total_q = Q_pad.shape[-2]
            total_k = K_pad.shape[-2]
            num_q_blocks = total_q // bsz
            num_k_blocks = total_k // bsz

            # ---- [XATTN-FIX-07] Padding-aware block averaging ----
            q_valid = torch.zeros((B, 1, total_q, 1), device=device, dtype=Q.dtype)
            q_valid[:, :, :N_q, :] = 1.0
            k_valid = torch.zeros((B, 1, total_k, 1), device=device, dtype=K_full.dtype)
            k_valid[:, :, :N_k, :] = 1.0

            if valid_m is not None:
                k_valid[:, :, :N_k, 0] = (
                    k_valid[:, :, :N_k, 0] * valid_m.unsqueeze(1).to(K_full.dtype)
                )

            q_valid_blk = q_valid.view(B, 1, num_q_blocks, bsz, 1)
            Q_blocks = (
                Q_pad.view(B, H, num_q_blocks, bsz, d) * q_valid_blk
            ).sum(dim=3) / q_valid_blk.sum(dim=3).clamp(min=1.0)

            k_valid_blk = k_valid.view(B, 1, num_k_blocks, bsz, 1)
            K_blocks = (
                K_pad.view(B, H, num_k_blocks, bsz, d) * k_valid_blk
            ).sum(dim=3) / k_valid_blk.sum(dim=3).clamp(min=1.0)

            coarse_scores = (
                torch.matmul(Q_blocks, K_blocks.transpose(-1, -2).contiguous())
                * scale
            )

            if self.is_decoder:
                q_blk = (
                    torch.arange(num_q_blocks, device=device).unsqueeze(1)
                    + (num_k_blocks - num_q_blocks)
                )
                k_blk = torch.arange(num_k_blocks, device=device).unsqueeze(0)
                causal_blk = (k_blk <= q_blk).unsqueeze(0).unsqueeze(0)
                coarse_scores = coarse_scores.masked_fill(~causal_blk, -1e4)
            else:
                causal_blk = None

            k_blocks_keep = max(
                1,
                min(num_k_blocks, int(math.ceil(num_k_blocks * self.keep_ratio))),
            )
            _, topk_idx = torch.topk(coarse_scores, k=k_blocks_keep, dim=-1)

            block_mask = torch.zeros(
                (B, H, num_q_blocks, num_k_blocks), device=device, dtype=torch.bool
            )
            block_mask.scatter_(-1, topk_idx, True)

            if self.is_decoder and causal_blk is not None:
                block_mask = block_mask & causal_blk
                # Always keep the diagonal (self-attention block)
                diag_k = min(num_q_blocks, num_k_blocks)
                diag_idx = torch.arange(diag_k, device=device)
                block_mask[:, :, diag_idx, diag_idx] = True

            # ---- [XATTN-FIX-11] Triton fast path ----
            if (
                HAS_TRITON
                and Q.is_cuda
                and triton_block_sparse_attention is not None
            ):
                try:
                    token_mask = valid_m if valid_m is not None else None
                    context = triton_block_sparse_attention(
                        q=Q,
                        k=K_full,
                        v=V_full,
                        block_mask=block_mask,
                        block_size=self.block_size,
                        token_mask=token_mask,
                        is_causal=self.is_decoder,
                    )
                except Exception as e:
                    logger.warning(
                        "Triton block-sparse attention failed (%s); falling back to PyTorch.",
                        e,
                    )
                    context = None
                    attn = None
            else:
                context = None

            # ---- PyTorch fallback ----
            if context is None:
                if pad_q > 0 or pad_k > 0:
                    scores_p = F.pad(
                        scores,
                        (0, pad_k, 0, pad_q),
                        value=-1e4,
                    )
                else:
                    scores_p = scores

                scores_p = scores_p.view(
                    B, H, num_q_blocks, bsz, num_k_blocks, bsz
                )
                scores_p = scores_p.masked_fill(
                    ~block_mask.unsqueeze(3).unsqueeze(5), -1e4
                )
                scores = scores_p.reshape(B, H, total_q, total_k)[
                    :, :, :N_q, :N_k
                ]

                if self.is_decoder:
                    q_pos = (
                        past_len + torch.arange(N_q, device=device)
                    ).unsqueeze(1)
                    k_pos = torch.arange(N_k, device=device).unsqueeze(0)
                    causal = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                    scores = scores.masked_fill(~causal, -1e4)

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        scores = scores + attention_mask.to(scores.dtype)
                    elif valid_m is not None:
                        scores = scores.masked_fill(
                            ~valid_m.unsqueeze(1).unsqueeze(2), -1e4
                        )

                scores = torch.nan_to_num(
                    scores, nan=-1e4, posinf=1e4, neginf=-1e4
                )
                attn = F.softmax(scores.float(), dim=-1).to(dtype)
                attn = self.dropout(attn)
                context = torch.matmul(attn, V_full)

        # ---- head_mask ----
        if head_mask is not None and attn is not None:
            hm = head_mask.to(attn.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            attn = attn * hm
            context = torch.matmul(attn, V_full)

        # ---- Reshape + output projection ----
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(B, N_q, self.all_head_size)
        )

        if self.is_nested_encoder:
            output = context * self.lora_output_scale
        else:
            output = self.out_proj(context) * self.lora_output_scale

        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Mask output on padding query tokens ----
        if attention_mask is not None:
            v_m = _extract_valid_token_mask(attention_mask, B, N_q, device)
            output = output * v_m.unsqueeze(-1).to(output.dtype)

        if output_attentions:
            return output.to(dtype), (
                new_past_key_value if self.is_decoder else None
            ), attn
        return output.to(dtype), (
            new_past_key_value if self.is_decoder else None
        )


# ===========================================================================
# Injection
# ===========================================================================
def inject_xattention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    # [XATTN-FIX-09] Recursion guard
    if _depth > 8:
        logger.warning("inject_xattention: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = (
        model_config
        if model_config is not None
        else getattr(target, "config", None)
    )

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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_xattention(
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

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None:
        logger.warning(
            "inject_xattention: no source attention found on %s",
            type(target).__name__,
        )
        return None

    # [XATTN-FIX-09] Idempotency guard
    if isinstance(src_attn, XAttention) or "XAttention" in src_attn.__class__.__name__:
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    # ---- Resolve dims ----
    hidden_size = (
        _cfg_get(cfg, "hidden_size")
        or _cfg_get(cfg, "d_model")
        or _cfg_get(cfg, "dim")
        or getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj")
            and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin")
            and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (
            src_attn.query.in_features
            if hasattr(src_attn, "query")
            and hasattr(src_attn.query, "in_features")
            else None
        )
        or 768
    )
    num_heads = (
        _cfg_get(cfg, "num_attention_heads")
        or _cfg_get(cfg, "num_heads")
        or _cfg_get(cfg, "n_heads")
        or getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or 12
    )

    # ---- Fallback config ----
    if cfg is None:
        import types

        cfg = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(
                src_attn, "num_key_value_heads", num_heads
            ),
            is_decoder=getattr(src_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop(
        "lora_rank", _cfg_get(comp_config, "lora_rank", 16)
    )
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))
    block_size = kwargs.pop(
        "block_size", _cfg_get(comp_config, "block_size", 32)
    )
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "sparsity", 0.5))

    try:
        new_attn = XAttention(
            config=cfg,
            rank=rank,
            block_size=block_size,
            sparsity=sparsity,
            lora_rank=lora_rank,
            pretrained_attn=src_attn,
            layer_idx=layer_idx,
            **kwargs,
        )

        # ---- Move to source device/dtype ----
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        # ---- Splice ----
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True

        if not spliced:
            logger.error(
                "inject_xattention: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        return new_attn

    except Exception as e:
        logger.exception(
            "inject_xattention failed at layer %d: %s", layer_idx, e
        )
        return None


__all__ = [
    "XAttention",
    "LoRALinear",
    "inject_xattention",
    "apply_rotary_pos_emb",
]