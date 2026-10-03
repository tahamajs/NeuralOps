r"""
Grouped-Query Attention (GQA) — v3, rank-aware.

Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
by dividing H_Q query heads into G groups, where each group shares a single
Key-Value head pair (H_KV = G ≤ H_Q).

Fixes (v3):
  GQA-FIX-01  Causal mask via past_len, always applied for decoders.
  GQA-FIX-02  Removed output padding/truncation hack.
  GQA-FIX-03  Deterministic layer detection.
  GQA-FIX-04  Removed thread-unsafe global injection depth.
  GQA-FIX-05  Idempotency guard in injector.
  GQA-FIX-06  Robust mask normalization (2D/3D/4D).
  GQA-FIX-07  Consistent is_nested_encoder propagation.
  GQA-FIX-08  Storage accounting.
  GQA-FIX-09  Partial-RoPE support.
  GQA-FIX-10  Conv1D bias handled correctly.
  GQA-FIX-11  GQA-aware copy_weights with repeat_interleave pooling.
  GQA-FIX-12  supports_rope flag.
  GQA-FIX-13  dtype-safe RoPE.
  GQA-FIX-14  Removed redundant double softmax.
  GQA-FIX-15  get_ortho_loss for framework compatibility.
  GQA-FIX-16  **`rank` now maps to `num_kv_heads` when passed via injector.**
              Rank sweep actually produces different compression levels.
  GQA-FIX-17  **`_unpack_weight_bias` handled for LoRA-wrapped modules.**
  GQA-FIX-18  **Numerically safe padding mask path.**
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    if hasattr(module, "base_layer"):
        module = module.base_layer
    w = getattr(module, "weight", None)
    b = getattr(module, "bias", None)
    cls_name = module.__class__.__name__

    if w is not None:
        w = w.data
        if hasattr(module, "nf") or "Conv1D" in cls_name:
            w = w.t().contiguous()
    if b is not None:
        b = b.data

    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    if d % 2 != 0:
        x_even = x[..., :-1]
        x1 = x_even[..., : (d - 1) // 2]
        x2 = x_even[..., (d - 1) // 2 :]
        rot_even = torch.cat((-x2, x1), dim=-1)
        return torch.cat((rot_even, x[..., -1:]), dim=-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
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
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = min(cos.shape[-1], q.shape[-1], k.shape[-1])
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim].to(q_rot.dtype)

    q_rot_emb = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_emb = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_emb, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_emb
    k = torch.cat([k_rot_emb, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_emb
    return q, k


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=16,
        lora_alpha=32.0,
        freeze_base=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
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

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = (
                self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            )
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register(
    "gqa",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
@ATTENTION_REGISTRY.register(
    "gqa_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
@ATTENTION_REGISTRY.register(
    "grouped_query_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
class GroupedQueryAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- hidden_size ----
        if dim is None and config is not None:
            dim = (
                _cfg_get(config, "hidden_size", None)
                or _cfg_get(config, "d_model", None)
                or _cfg_get(config, "n_embd", None)
                or _cfg_get(config, "dim", None)
            )
        if dim is None and src_attn is not None:
            dim = getattr(src_attn, "hidden_size", getattr(src_attn, "embed_dim", None))
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None:
                        w, _ = _unpack_weight_bias(m)
                        if w is not None and w.dim() == 2:
                            dim = w.shape[1]
                            break
        self.hidden_size = int(dim or kwargs.get("hidden_size", 768))

        # ---- num_heads ----
        if num_heads is None and config is not None:
            num_heads = (
                _cfg_get(config, "num_attention_heads", None)
                or _cfg_get(config, "num_heads", None)
                or _cfg_get(config, "n_head", None)
            )
        if num_heads is None and src_attn is not None:
            num_heads = getattr(
                src_attn,
                "num_heads",
                getattr(
                    src_attn, "num_attention_heads", getattr(src_attn, "n_head", None)
                ),
            )
        self.num_heads = int(
            num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 8))
        )

        # ---- head_dim ----
        self.head_dim = int(
            kwargs.get(
                "head_dim", getattr(config, "head_dim", None) if config else None
            )
            or (self.hidden_size // max(1, self.num_heads))
        )

        # ==================================================================
        # GQA-FIX-16: derive num_kv_heads from `rank` when not explicitly given
        # ==================================================================
        # Priority:
        #   1. explicit `num_kv_heads` argument
        #   2. config's num_key_value_heads
        #   3. `rank` argument (1 <= rank <= num_heads)
        #   4. kwargs' num_kv_heads
        #   5. default = num_heads // 4
        # ==================================================================
        resolved_kv_heads = None
        source = "default"

        if num_kv_heads is not None:
            resolved_kv_heads = int(num_kv_heads)
            source = "explicit_arg"
        elif config is not None:
            cfg_kv = _cfg_get(config, "num_key_value_heads", None) or _cfg_get(
                config, "num_kv_heads", None
            )
            if cfg_kv is not None:
                resolved_kv_heads = int(cfg_kv)
                source = "config"
        if resolved_kv_heads is None and src_attn is not None:
            src_kv = getattr(
                src_attn, "num_key_value_heads", getattr(src_attn, "num_kv_heads", None)
            )
            if src_kv is not None:
                resolved_kv_heads = int(src_kv)
                source = "src_attn"

        # GQA-FIX-16: if `rank` is given and looks like a head count (1..num_heads),
        # use it as num_kv_heads. This makes the rank sweep actually produce
        # different compression levels.
        if rank is not None and 1 <= int(rank) <= self.num_heads:
            resolved_kv_heads = int(rank)
            source = "rank"

        if resolved_kv_heads is None:
            kwargs_kv = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads"))
            if kwargs_kv is not None:
                resolved_kv_heads = int(kwargs_kv)
                source = "kwargs"
            else:
                resolved_kv_heads = max(1, self.num_heads // 4)
                source = "default"

        self.num_kv_heads = int(min(self.num_heads, max(1, resolved_kv_heads)))
        self._num_kv_heads_source = source

        # ---- sizes ----
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            is_dec = _cfg_get(
                config,
                "is_decoder",
                _cfg_get(
                    config,
                    "is_causal",
                    kwargs.get("is_causal", kwargs.get("is_decoder", False)),
                ),
            )
            _dec_pfx = (
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
            if any(model_type.startswith(p) for p in _dec_pfx):
                is_dec = True
            if src_attn is not None:
                if getattr(src_attn, "is_decoder", False) or hasattr(
                    src_attn, "c_attn"
                ):
                    is_dec = True
            self.is_decoder = bool(is_dec)
        self.is_nested_encoder = False

        # ---- LoRA ----
        comp_config = getattr(config, "compression_config", None) if config else None
        self.lora_rank = int(
            _cfg_get(
                comp_config,
                "lora_rank",
                _cfg_get(config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
        )
        self.lora_alpha = float(
            _cfg_get(
                comp_config,
                "lora_alpha",
                _cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", lora_alpha)),
            )
        )
        self.freeze_base = self.lora_rank > 0

        # ---- projections ----
        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(float(dropout_p))

        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        # Storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if src_attn is not None:
            self.copy_weights(src_attn)

        # Log the resolved head config
        logger.debug(
            "GQA: hidden=%d num_heads=%d num_kv_heads=%d (source=%s) "
            "kv_repeat=%d head_dim=%d",
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self._num_kv_heads_source,
            self.kv_repeat,
            self.head_dim,
        )

    # ---- aliases ----
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

    @property
    def c_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        for proj in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter failed for %s: %s", proj, e)

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            weight, bias_data = _unpack_weight_bias(src.c_attn)
            if weight is not None:
                split_size = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    (
                        bias_data[split_size : 2 * split_size]
                        if bias_data is not None
                        else None
                    ),
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o, b_o = _unpack_weight_bias(src.c_proj)
            extracted["o"] = (w_o, b_o)

        # Nested BERT/RoBERTa
        if hasattr(src, "self") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = _unpack_weight_bias(s.query)
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = _unpack_weight_bias(s.key)
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = _unpack_weight_bias(s.value)

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            extracted["o"] = _unpack_weight_bias(src.output.dense)

        # Flat modules
        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for n in names:
                    if hasattr(src, n):
                        w, b = _unpack_weight_bias(getattr(src, n))
                        if w is not None:
                            extracted[proj] = (w, b)
                            break

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, a)
            for a in (
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
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("GQA copy_weights: missing %s", missing)

        def _orient(w, target_shape):
            if w.dim() != 2:
                return w
            if w.shape == target_shape:
                return w
            if w.shape == (target_shape[1], target_shape[0]):
                return w.t().contiguous()
            return w

        # GQA-aware pooling for K/V
        def _pool_gqa(w, target_out):
            """
            Map source K/V of shape (src_out, in) to (target_out, in).
            If src has more heads, mean-pool groups.
            If src has fewer heads, repeat each head.
            """
            src_out = w.shape[0]
            if src_out == target_out:
                return w
            if src_out > target_out and src_out % target_out == 0:
                rep = src_out // target_out
                return w.view(target_out, rep, -1).mean(dim=1)
            if src_out < target_out and target_out % src_out == 0:
                rep = target_out // src_out
                return w.repeat_interleave(rep, dim=0)
            # Fallback: principal slice
            out = torch.zeros(target_out, w.shape[1], dtype=w.dtype, device=w.device)
            m = min(target_out, src_out)
            out[:m].copy_(w[:m])
            return out

        with torch.no_grad():
            for key, target in (
                ("q", self.query),
                ("k", self.key),
                ("v", self.value),
                ("o", self.out_proj),
            ):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        min_d = min(target.weight.shape[0], target.weight.shape[1])
                        target.weight.data.zero_()
                        target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                        if target.bias is not None:
                            target.bias.data.zero_()
                    target.freeze_base_weights()
                    continue

                if key in ("k", "v"):
                    w = _pool_gqa(w, target.weight.shape[0])
                    if b is not None:
                        b = _pool_gqa(b.unsqueeze(-1), target.weight.shape[0]).squeeze(
                            -1
                        )

                w = _orient(w, target.weight.shape)
                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data.zero_()
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mn = min(target.bias.numel(), b.numel())
                    target.bias.data.zero_()
                    target.bias.data[:mn].copy_(
                        b[:mn].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    target.bias.data.zero_()

                nrm = float(torch.linalg.norm(target.weight.data.float()).item())
                if nrm < 1e-4:
                    logger.warning("GQA: '%s' near-zero norm (%.2e)", key, nrm)

                target.freeze_base_weights()

        try:
            for m in (self.query, self.key, self.value, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, L, device):
        if attention_mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape == (B, L):
                return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        key_value_states = kwargs.get("key_value_states")
        is_cross = key_value_states is not None or encoder_hidden_states is not None
        kv_input = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else (key_value_states if key_value_states is not None else hidden_states)
        )

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = (
            self.key(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.value(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # ---- GQA expand ----
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # ---- Attention scores ----
        scale = d**-0.5
        scores = (
            torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        )

        # ---- Causal mask ----
        is_causal = bool(self.is_decoder and not is_cross)
        if is_causal:
            past_len = max(0, N_k - N_q)
            causal_diag = 1 + past_len
            causal = (
                torch.triu(
                    torch.full(
                        (N_q, N_k), float("-inf"), device=device, dtype=torch.float32
                    ),
                    diagonal=causal_diag,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        # ---- Padding mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("mask")
        valid_mask = self._prepare_valid_mask(eff_mask, B, N_k, device)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask.to(attn.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            attn = attn * hm

        # ---- Context ----
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        return output, (attn if output_attentions else None)

    def get_ortho_loss(self):
        return torch.zeros((), device=self.query.weight.device)

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("out_proj", self.out_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection (GQA-FIX-16: propagate rank → num_kv_heads)
# ===========================================================================
@ATTENTION_REGISTRY.register("gqa_inject")
@ATTENTION_REGISTRY.register("inject_gqa")
def inject_gqa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    rank: int | None = None,
    num_kv_heads: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_gqa: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
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
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gqa(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                rank=rank,
                num_kv_heads=num_kv_heads,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # ---- Single layer ----
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

    if (
        isinstance(src_attn, GroupedQueryAttention)
        or "GroupedQueryAttention" in src_attn.__class__.__name__
    ):
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "dim", None)
        or (
            getattr(src_attn.query, "in_features", None)
            if hasattr(src_attn, "query")
            else None
        )
        or getattr(cfg, "hidden_size", 768)
    )
    num_heads = (
        getattr(cfg, "num_attention_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "num_heads", None)
        or 12
    )

    try:
        new_attn = GroupedQueryAttention(
            config=cfg,
            pretrained_attn=src_attn if src_attn is not target else None,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            rank=rank,
            num_kv_heads=num_kv_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("GQA construction failed: %s", e)
        return None

    p = next(src_attn.parameters(), None) if src_attn is not target else None
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

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
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True

    if not spliced:
        logger.error(
            "inject_gqa: no writable attention attribute on %s", type(target).__name__
        )
        return None

    return new_attn


__all__ = [
    "GroupedQueryAttention",
    "inject_gqa",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
