"""
Locality-Sensitive Hashing (LSH) Attention – Reformer style (Corrected).

Approximates dense attention by clustering keys and queries into discrete hash buckets
via angular random hyperplane projections.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- FactorizedLoRALinear (parameter-reducing) with frozen base weights
- GQA with repeat_kv
- RoPE with past_len offset
- Causal masking for decoders (falls back to exact attention)
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

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE (dtype-safe, past_len-aware)
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
    position_ids: torch.Tensor | None = None,
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
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
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
# LSH-08: FactorizedLoRALinear
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
        orig_dtype = x.dtype
        if x.dtype != self.down.weight.dtype:
            x = x.to(self.down.weight.dtype)
        out = self.up(self.down(x))
        if self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out.to(orig_dtype)


# ===========================================================================
# Main LSH attention
# ===========================================================================

@ATTENTION_REGISTRY.register("lsh", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("lsh_attention", supports_causal=True, supports_gqa=True)
class LSHAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: int | None = None,
        rounds: int = 4,
        num_buckets: int = 16,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # LSH-02: robust super().__init__
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        # LSH-10: set self.config explicitly
        self.config = config

        # LSH-03, LSH-13: robust dimension detection (independent of ModelAdapter)
        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = self._detect_head_dim(config, source_attn, kwargs, hidden_size, num_heads)

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # LSH-11: clamp num_buckets
        self.rounds = max(1, int(getattr(config, "lsh_rounds", rounds) if config else rounds))
        self.num_buckets = max(2, int(getattr(config, "lsh_buckets", num_buckets) if config else num_buckets))

        # Decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = int(
            (getattr(config, "lora_rank", None) if config else None)
            or (getattr(comp_config, "lora_rank", None) if comp_config else None)
            or lora_rank
        )
        self.lora_alpha = float(
            (getattr(config, "lora_alpha", None) if config else None)
            or (getattr(comp_config, "lora_alpha", None) if comp_config else None)
            or lora_alpha
        )

        # LSH-08: projection rank
        self.proj_rank = int(
            kwargs.get("proj_rank")
            or (getattr(comp_config, "proj_rank", None) if comp_config else None)
            or (rank if rank is not None else 64)
        )

        # FactorizedLoRALinear projections
        FF = FactorizedLoRALinear
        self.query = FF(self.hidden_size, self.all_head_size,
                        rank=self.proj_rank, lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha, bias=True)
        self.key = FF(self.hidden_size, self.kv_head_size,
                      rank=self.proj_rank, lora_rank=self.lora_rank,
                      lora_alpha=self.lora_alpha, bias=True)
        self.value = FF(self.hidden_size, self.kv_head_size,
                        rank=self.proj_rank, lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha, bias=True)
        self.out_proj = FF(self.all_head_size, self.hidden_size,
                           rank=self.proj_rank, lora_rank=self.lora_rank,
                           lora_alpha=self.lora_alpha, bias=True)

        dropout_p = float(
            getattr(config, "dropout",
                     getattr(config, "hidden_dropout_prob",
                             getattr(config, "attention_probs_dropout_prob", 0.1)))
             if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        # LSH-14: per-round random hyperplane projection, normalized per column
        # Shape: (rounds, head_dim, num_buckets // 2)
        half_buckets = max(1, self.num_buckets // 2)
        proj_data = torch.randn(self.rounds, self.head_dim, half_buckets)
        # Normalize each hyperplane (column) to unit L2 norm
        proj_data = F.normalize(proj_data, p=2, dim=1)
        self.register_buffer("proj_matrices", proj_data, persistent=False)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # LSH-13: dimension detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_hidden_size(config: Any, src: nn.Module | None, kwargs: dict[str, Any]) -> int:
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
    def _detect_num_heads(config: Any, src: nn.Module | None, kwargs: dict[str, Any],
                          hidden_size: int) -> int:
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
    def _detect_num_kv_heads(config: Any, src: nn.Module | None, kwargs: dict[str, Any],
                             num_heads: int) -> int:
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
    def _detect_head_dim(config: Any, src: nn.Module | None, kwargs: dict[str, Any],
                         hidden_size: int, num_heads: int) -> int:
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
    # SVD init
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(w: torch.Tensor, b: torch.Tensor | None,
                  proj: FactorizedLoRALinear) -> None:
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
                logger.warning("LSHAttention._svd_init SVD failed: %s", e)
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
    # LSH-07: copy_weights raises on missing
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (hasattr(src, "self") and hasattr(src.self, "query")
                and hasattr(src, "output") and hasattr(src.output, "dense")) or (cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                           "XLMRobertaSelfAttention", "ElectraSelfAttention",
                           "DebertaSelfAttention"]
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # Try UniversalAdapter first
        try:
            from framework.core.adapters import UniversalAdapter
            for proj in ["q", "k", "v", "o"]:
                try:
                    w, b = UniversalAdapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted[proj] = (w, b)
                except Exception:
                    pass
        except Exception:
            pass

        # 1. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None and \
                any(extracted[k][0] is None for k in ("q", "k", "v")):
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
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split, :], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2 * split, :], c_b[split:2 * split] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split:, :], c_b[2 * split:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"][0] is None:
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
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. BERT / RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query") and \
                any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))
            if (hasattr(src, "output") and hasattr(src.output, "dense")
                    and extracted["o"][0] is None):
                dense = src.output.dense
                extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 3. Flat modules (DistilBERT, Qwen, LLaMA)
        else:
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
                            mod = getattr(src, name)
                            if mod is not None and hasattr(mod, "weight"):
                                w_data = mod.weight.data
                                if (proj == "o"
                                        and w_data.ndim == 2
                                        and w_data.shape[0] != self.hidden_size
                                        and w_data.shape[1] == self.hidden_size):
                                    w_data = w_data.t().contiguous()
                                extracted[proj] = (w_data, getattr(mod, "bias", None))
                                break

        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            raise RuntimeError(
                f"LSHAttention.copy_weights: missing mandatory projections {missing} "
                f"from {cls_name}."
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
                eye = torch.eye(self.all_head_size, self.all_head_size,
                                device=self.out_proj.down.weight.device,
                                dtype=self.out_proj.down.weight.dtype)
                self._svd_init(eye, None, self.out_proj)

            # Norm verification
            for name, mod in (("query", self.query), ("key", self.key),
                              ("value", self.value), ("out_proj", self.out_proj)):
                try:
                    w_norm = float(torch.linalg.norm(mod.weight.float()).item())
                    if w_norm < 1e-4:
                        logger.warning(
                            "LSHAttention: weight '%s' has near-zero norm (%.2e).",
                            name, w_norm,
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
                dummy = torch.randn(1, 8, self.hidden_size, device=d, dtype=dt)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("LSHAttention forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    # LSH hashing
    # ------------------------------------------------------------------
    def _hash(self, vectors: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """
        Hash vectors via random hyperplane sign pattern.
        vectors: (B, H, N, d); proj: (d, num_buckets//2)
        Returns: (B, H, N) bucket indices in [0, num_buckets)
        """
        norm = F.normalize(vectors.float(), p=2, dim=-1)
        dots = torch.matmul(norm, proj.float())   # (B, H, N, num_buckets//2)
        signs = (dots >= 0).long()                # (B, H, N, num_buckets//2)
        # Combine sign bits into a single bucket index
        num_planes = signs.shape[-1]
        weights = (2 ** torch.arange(num_planes, device=signs.device)).long()
        bucket = (signs * weights).sum(dim=-1)    # (B, H, N)
        bucket = bucket % self.num_buckets
        return bucket

    # ------------------------------------------------------------------
    # LSH-09: chunked LSH attention
    # ------------------------------------------------------------------
    def _lsh_attention_chunked(
        self,
        Q: torch.Tensor,         # (B, H, N, d)
        K: torch.Tensor,         # (B, H, N, d)
        V: torch.Tensor,         # (B, H, N, d)
        valid_mask: torch.Tensor | None,  # (B, N) bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Chunked LSH attention. For each round, compute per-query chunk attention
        over keys in the same bucket. Memory O(B*H*N*C) per round (C=chunk).
        """
        B, H, N, d = Q.shape
        device = Q.device
        dtype = Q.dtype
        scale = d ** -0.5

        num_chunks = max(1, (N + 63) // 64)
        chunk_size = (N + num_chunks - 1) // num_chunks

        accum = torch.zeros_like(Q)
        rounds_eff = min(self.rounds, self.proj_matrices.size(0))

        for r in range(rounds_eff):
            proj_r = self.proj_matrices[r].to(device=device, dtype=dtype)
            q_hash = self._hash(Q, proj_r)   # (B, H, N)
            k_hash = self._hash(K, proj_r)

            for c in range(num_chunks):
                c_start = c * chunk_size
                c_end = min(c_start + chunk_size, N)
                Qc = Q[:, :, c_start:c_end]         # (B, H, C, d)
                hq_c = q_hash[:, :, c_start:c_end]   # (B, H, C)

                # bucket match: (B, H, C, N)
                bucket_match = (hq_c.unsqueeze(-1) == k_hash.unsqueeze(-2))

                # Mask padding keys
                if valid_mask is not None:
                    kv_mask = valid_mask[:, None, None, :]  # (B, 1, 1, N)
                    bucket_match = bucket_match & kv_mask

                # scores: (B, H, C, N)
                scores_c = torch.matmul(
                    Qc.float(), K.float().transpose(-2, -1).contiguous()
                ) * scale

                scores_c = scores_c.masked_fill(~bucket_match, -1e4)
                scores_c = torch.nan_to_num(scores_c, nan=-1e4, posinf=1e4, neginf=-1e4)

                attn_c = F.softmax(scores_c, dim=-1).to(dtype)
                attn_c = torch.nan_to_num(attn_c, nan=0.0)
                attn_c = self.dropout(attn_c)

                accum[:, :, c_start:c_end] = accum[:, :, c_start:c_end] + torch.matmul(attn_c, V)

        accum = accum / float(rounds_eff)
        return accum, None

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Handle key_value_states via kwargs
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross = key_value_states is not None
        kv_input = key_value_states if is_cross else hidden_states

        B, N_q, _ = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # RoPE with past_len offset
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
            if past_len > 0:
                cos = cos[..., past_len: past_len + N_q, :]
                sin = sin[..., past_len: past_len + N_q, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # Valid mask
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        valid_mask = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype == torch.bool:
                    valid_mask = eff_mask
                elif eff_mask.dtype.is_floating_point:
                    valid_mask = eff_mask > -1e4
                else:
                    valid_mask = eff_mask > 0
            elif eff_mask.dim() == 4:
                if eff_mask.dtype == torch.bool:
                    v = eff_mask
                elif eff_mask.dtype.is_floating_point:
                    v = eff_mask > -1e4
                else:
                    v = eff_mask > 0
                valid_mask = v.any(dim=-2).squeeze(1)

        # LSH-15: decoders fall back to exact attention (LSH doesn't naturally support causal)
        use_exact = (
            self.is_decoder
            or is_cross
            or N_q <= self.num_buckets
        )

        if use_exact:
            scale = d ** -0.5
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

            if valid_mask is not None:
                scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

            if self.is_decoder and not is_cross:
                causal_tril = torch.tril(
                    torch.ones((N_q, N_k), device=device, dtype=torch.bool)
                ).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~causal_tril, -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores, dim=-1).to(dtype)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    hm = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    hm = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    hm = head_mask[:, :, :1].unsqueeze(-1)
                else:
                    hm = head_mask
                attn = attn * hm.to(attn.dtype)

            context = torch.matmul(attn, V)
            attn_ret = attn if output_attentions else None
        else:
            # LSH-09: chunked LSH attention
            context, _ = self._lsh_attention_chunked(Q, K, V, valid_mask)
            attn_ret = None
            # Note: LSH path does not produce a single interpretable attention matrix.

        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out.to(dtype), attn_ret


# ===========================================================================
# Injection (LSH-04: recursion guard)
# ===========================================================================

@ATTENTION_REGISTRY.register("lsh_attention_inject")
@ATTENTION_REGISTRY.register("lsh_inject")
@ATTENTION_REGISTRY.register("inject_lsh_attention")
@ATTENTION_REGISTRY.register("inject_lsh")
def inject_lsh_attention(
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
    if _depth > 8:
        logger.warning("inject_lsh_attention: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_lsh_attention(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        src_attn = old_attn
        if src_attn is None and layer is not None:
            for attr in ("attn", "attention", "self_attn", "self"):
                if hasattr(layer, attr):
                    cand = getattr(layer, attr)
                    src_attn = getattr(cand, "self", cand)
                    break

        # Double-injection guard
        if isinstance(src_attn, LSHAttention) or "LSH" in src_attn.__class__.__name__:
            return None

        if cfg is None and src_attn is not None:
            cfg = getattr(src_attn, "config", None)

        new_attn = LSHAttention(
            config=cfg,
            pretrained_attn=src_attn,
            comp_config=comp_config,
            **kwargs,
        )

        p = next(src_attn.parameters(), None) if src_attn is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if layer is not None:
            if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            elif hasattr(layer, "attention"):
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
            elif hasattr(layer, "self_attn"):
                new_attn.is_nested_encoder = False
                layer.self_attn = new_attn
            elif hasattr(layer, "self"):
                new_attn.is_nested_encoder = True
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                new_attn.is_nested_encoder = False
                layer.attn = new_attn

        if src_attn is not None:
            new_attn.train(src_attn.training)
        elif layer is not None and hasattr(layer, 'training'):
            new_attn.train(layer.training)

        return new_attn
    except Exception as e:
        logger.exception("inject_lsh_attention failed at layer %d: %s", layer_idx, e)
        return None
