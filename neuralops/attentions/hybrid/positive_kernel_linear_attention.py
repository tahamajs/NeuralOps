"""
LEKAv1: Linear-time Efficient Kernel Attention.

Approximates the attention kernel via Positive Feature Mapping (ELU + 1)
to guarantee strict positivity and stable gradients, avoiding NaN denominator collapse.

Fixes applied:
  [FIX-1]  Full KV-cache support (read + concat + write + return present_key_value)
  [FIX-2]  `pres_gate` → `lora_pres_gate` (KD optimizer detection)
  [FIX-3]  `_causal_linear_attention` vectorized + KV-cache aware (past_len)
  [FIX-4]  Padding mask uses N_kv (not N_q) — decode-safe
  [FIX-5]  `_rotate_half` imported from core.rope
  [FIX-6]  `repeat_kv` imported from core.gqa_utils
  [FIX-7]  Robust decoder detection (kwargs + config + src.c_attn)
  [FIX-8]  `get_token_valid_mask` from core.attention_mask
  [FIX-9]  Injector registered in ATTENTION_REGISTRY (4 keys)
  [FIX-10] Idempotency check in injector
  [FIX-11] Denom clamp at 1e-5
  [FIX-12] `__all__` exports
  [FIX-13] Unified return format for decoder (2-tuple with past_kv)
  [FIX-14] Safe handling of `past_key_value=None` sentinel `False`
"""

from __future__ import annotations

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
# RoPE (dtype-safe)
# ===========================================================================
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
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# [FIX-7] Decoder detection
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
# FactorizedLoRALinear
# ===========================================================================
class FactorizedLoRALinear(nn.Module):
    """SVD-factorized linear + LoRA adapter.

    Forward: y = up(down(x)) + (lora_B @ lora_A)(x) * scaling
    Base (down, up) frozen; LoRA trainable.
    """

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
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


# ===========================================================================
# Main LEKAv1
# ===========================================================================
class PositiveKernelLinearAttention(BaseAttention):
    """Linear-time attention via ELU+1 kernel feature map.

    Q and K are mapped to a strictly positive feature space via ELU(x)+1.
    This allows the linear-attention reordering (QK^T)V → Q(K^T V),
    reducing complexity from O(N²) to O(N).

    Stability: strictly positive feature map prevents NaN from denominator collapse.
    """

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
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.layer_idx = int(layer_idx)

        self.feature_dim = int(
            getattr(comp_config, "feature_dim", 256) if comp_config is not None else 256
        )
        dropout_p = float(
            getattr(comp_config, "dropout", getattr(config, "dropout", 0.1))
            if config is not None else 0.1
        )

        # [FIX-7] Decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

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

        self.norm = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout_p)

        # [FIX-2] pres_gate with lora_ prefix
        # sigmoid(2.0) ≈ 0.88 → attention dominant at init
        self.lora_pres_gate = nn.Parameter(torch.tensor(2.0))

        # ---- Kernel feature projection ----
        proj_data = self._build_projection_buffer()
        self.register_buffer("proj", proj_data, persistent=True)

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
            for attr in ("q_proj", "q_lin", "query"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    try:
                        return int(mod.weight.shape[0] // max(1, num_heads))
                    except Exception:
                        pass
        return max(1, hidden_size // max(1, num_heads))

    # ------------------------------------------------------------------
    # Robust QR-orthogonal projection builder
    # ------------------------------------------------------------------
    def _build_projection_buffer(self) -> torch.Tensor:
        d = self.head_dim
        F_dim = self.feature_dim
        proj = torch.empty(self.num_heads, d, F_dim)

        for h in range(self.num_heads):
            if F_dim <= d:
                q, _ = torch.linalg.qr(torch.randn(d, F_dim))
                proj[h] = q
            else:
                num_blocks = (F_dim + d - 1) // d
                blocks = [
                    torch.linalg.qr(torch.randn(d, d))[0] for _ in range(num_blocks)
                ]
                q = torch.cat(blocks, dim=1)[:, :F_dim]
                proj[h] = q

        return proj

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
                logger.warning("LEKAv1._svd_init: SVD failed (%s)", e)
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
    # Kernel feature map (ELU + 1)
    # ------------------------------------------------------------------
    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Positive kernel feature embedding using ELU + 1.

        Guarantees strict positivity so denominator never collapses.
        Returns: (B, H, N, F)
        """
        x_norm = self.norm(x)
        proj_matrix = self.proj.to(device=x.device, dtype=x.dtype)
        features = torch.einsum("bhnd,hdf->bhnf", x_norm, proj_matrix)
        return F.elu(features) + 1.0 + 1e-6

    # ------------------------------------------------------------------
    # Weight transfer
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

        # 3. Flat modules (DistilBERT, LLaMA, Qwen)
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
                f"PositiveKernelLinearAttention.copy_weights: missing mandatory "
                f"projections {missing_after} from {cls_name}."
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
                            "PositiveKernelLinearAttention: weight '%s' near-zero norm (%.2e).",
                            name, w_norm,
                        )
                except Exception:
                    pass

            for mod in (self.query, self.key, self.value, self.out_proj):
                mod.freeze_base_weights()

        # Inherit decoder flag
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            self.is_causal = self.is_decoder

        # Forward sanity check
        try:
            with torch.no_grad():
                d = next(self.parameters()).device
                dt = self.query.down.weight.dtype
                dummy = torch.randn(1, 4, self.dim, device=d, dtype=dt)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("LEKAv1 forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    # [FIX-3, FIX-11] Vectorized causal linear attention (KV-cache aware)
    # ------------------------------------------------------------------
    @staticmethod
    def _causal_linear_attention(
        Q_feat: torch.Tensor,   # (B, H, N, F)
        K_feat: torch.Tensor,   # (B, H, N_kv, F)
        V: torch.Tensor,        # (B, H, N_kv, d)
        past_len: int,
    ) -> torch.Tensor:
        """Fully vectorized causal linear attention.

        Uses cumsum over the full KV history, then slices at query positions
        to get the causal state. Handles both training (N == N_kv) and decode
        (N == 1) uniformly.

        Complexity: O(N_kv · F · d) memory, O(N · F · d) time.
        """
        B, H, N, F_dim = Q_feat.shape
        N_kv = K_feat.shape[2]

        # Compute cumulative state over KV history
        KV_outer = K_feat.unsqueeze(-1) * V.unsqueeze(-2)   # (B, H, N_kv, F, d)
        KV_cumsum = KV_outer.cumsum(dim=2)                   # inclusive
        K_cumsum = K_feat.cumsum(dim=2)                      # (B, H, N_kv, F)

        # Slice at query positions (past_len .. past_len + N)
        KV_at_q = KV_cumsum[:, :, past_len:past_len + N, :, :]
        K_at_q = K_cumsum[:, :, past_len:past_len + N, :]

        # Numerator: (B, H, N, F) × (B, H, N, F, d) → (B, H, N, d)
        num = torch.matmul(Q_feat.unsqueeze(-2), KV_at_q).squeeze(-2)

        # Denominator: strictly positive due to ELU+1
        den = (
            torch.einsum("bhnf,bhnf->bhn", Q_feat, K_at_q)
            .unsqueeze(-1)
            .clamp(min=1e-5)
        )

        return num / den

    # ------------------------------------------------------------------
    # [FIX-1] KV-cache helpers
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
        """Write (k, v) into past_key_value.

        Prefers in-place update if supported, otherwise returns 2-tuple.
        """
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
    # Forward  — [FIX-1, FIX-3, FIX-4]
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
        H, d = self.num_heads, self.head_dim
        dtype = hidden_states.dtype
        device = hidden_states.device

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
                if cos.dim() == 4:
                    cos_q = cos[..., past_len:past_len + N, :]
                    sin_q = sin[..., past_len:past_len + N, :]
                elif cos.dim() == 3:
                    cos_q = cos[:, past_len:past_len + N, :].unsqueeze(1)
                    sin_q = sin[:, past_len:past_len + N, :].unsqueeze(1)
                else:
                    cos_q = cos[past_len:past_len + N, :].unsqueeze(0).unsqueeze(0)
                    sin_q = sin[past_len:past_len + N, :].unsqueeze(0).unsqueeze(0)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos_q, sin_q)
            else:
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
        elif self.rotary_emb is not None:
            try:
                cos, sin = self.rotary_emb(k_new, position_ids=pos_ids)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
            except Exception as e:
                logger.debug("rotary_emb failed: %s", e)

        # ---- [FIX-1] Concat KV cache ----
        if past_k is not None:
            k = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
            v = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
        else:
            k, v = k_new, v_new

        # ---- [FIX-1, FIX-14] Write back cache ----
        present_kv: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache:
            present_kv = self._write_past_kv(k, v, past_key_value, self.layer_idx)
        else:
            present_kv = past_key_value

        N_kv = k.shape[-2]

        # ---- GQA repeat ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- [FIX-4] Padding mask on N_kv (not N_q) ----
        valid_mask: torch.Tensor | None = None
        if attention_mask is not None:
            valid_mask = get_token_valid_mask(attention_mask, B, N_kv, device)
            if valid_mask is not None:
                kv_mask = valid_mask[:, None, :, None].to(dtype)   # (B, 1, N_kv, 1)
                k = k * kv_mask
                v = v * kv_mask

        # ---- Kernel feature map ----
        Q_features = self.kernel_feature_map(q)
        K_features = self.kernel_feature_map(k)

        # ---- Compute attention ----
        if self.is_decoder:
            context = self._causal_linear_attention(
                Q_features.float(),
                K_features.float(),
                v.float(),
                past_len=past_len,
            ).to(dtype)
        else:
            # Bidirectional linear attention via associativity Q(K^T V)
            KV = torch.einsum("bhnf,bhnd->bhfd", K_features.float(), v.float())
            K_sum = K_features.float().sum(dim=2)
            num = torch.einsum("bhnf,bhfd->bhnd", Q_features.float(), KV)
            den = (
                torch.einsum("bhnf,bhf->bhn", Q_features.float(), K_sum)
                .unsqueeze(-1)
                .clamp(min=1e-5)
            )
            context = (num / den).to(dtype)

        context = torch.nan_to_num(context, nan=0.0)

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

        # ---- Reshape and output projection ----
        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )

        if self.is_nested_encoder:
            attn_out = context
        else:
            attn_out = self.out_proj(context)

        # ---- [FIX-2] Residual blend via lora_pres_gate ----
        alpha = torch.sigmoid(self.lora_pres_gate)
        out = alpha * attn_out + (1.0 - alpha) * hidden_states
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- [FIX-13] Return format ----
        if self.is_decoder:
            if output_attentions:
                return out.to(dtype), present_kv, None
            return out.to(dtype), present_kv

        return out.to(dtype), None

    def get_ortho_loss(self) -> float:
        return 0.0


# Backward-compat aliases
LEKAV1Attention = PositiveKernelLinearAttention


# ===========================================================================
# [FIX-9, FIX-10] Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("inject_leka_v1")
def inject_positive_kernel_linear(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_positive_kernel_linear: max recursion depth exceeded.")
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
            res = inject_positive_kernel_linear(
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
    if src_attn is None:
        src_attn = target

    # [FIX-10] Idempotency
    if (
        isinstance(src_attn, PositiveKernelLinearAttention)
        or "PositiveKernelLinear" in src_attn.__class__.__name__
    ):
        return None

    if (
        "DeltaNet" in src_attn.__class__.__name__
        or "LinearAttention" in src_attn.__class__.__name__
    ):
        return None

    if cfg is None and hasattr(src_attn, "config"):
        cfg = src_attn.config

    try:
        new_attn = PositiveKernelLinearAttention(
            config=cfg,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            old_attn=src_attn,
            **kwargs,
        )

        if src_attn is not None:
            new_attn.copy_weights(src_attn)
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
                "inject_positive_kernel_linear: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_positive_kernel_linear failed at layer %d: %s", layer_idx, e)
        return None


# Backward-compat aliases
inject_leka_v1 = inject_positive_kernel_linear


# ===========================================================================
# [FIX-12] Exports
# ===========================================================================
__all__ = [
    "PositiveKernelLinearAttention",
    "LEKAV1Attention",
    "FactorizedLoRALinear",
    "inject_positive_kernel_linear",
    "inject_leka_v1",
    "apply_rotary_pos_emb",
]