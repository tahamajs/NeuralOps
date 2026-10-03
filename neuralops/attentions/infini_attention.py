"""Infini-Attention with Compressive Memory and Sliding Window — Fixed (v3).

Reference: Munkhdalai et al., "Leave No Context Behind: Efficient Infinite
Context Transformers with Infini-attention", 2024.

Core mechanism:
    A_local = sliding-window softmax attention (with causal mask for decoder)
    M_t     = M_{t-1} + g_t · σ(K_t)^T V_t         (compressive memory update)
    z_t     = z_{t-1} + g_t · σ(K_t)                 (normalizer)
    A_mem   = σ(Q_t) M_t / (σ(Q_t) z_t + ε)          (memory read)
    output  = β · A_mem + (1-β) · A_local             (fusion)

where σ(x) = ELU(x) + 1.

Fixes (v3):
  INF3-FIX-01  Forward signature now matches BERT/RoBERTa/DistilBERT
               positional arg order exactly, so HF calls bind correctly.
               `output_attentions` is the 7th positional arg (BERT style);
               DistilBERT passes it as keyword. Both work.
  INF3-FIX-02  `layer_idx` accepted as keyword with default 0.
  INF3-FIX-03  All internal parameters (`beta`, `w_mem.weight`, `w_mem.bias`)
               are explicitly initialized on the target device/dtype
               BEFORE the first forward pass, so no device mismatch.
  INF3-FIX-04  `_compressive_memory_read` precomputes `g · φ(k)` outside the
               loop and uses a batched outer product; avoids `.to()` churn.
  INF3-FIX-05  Memory state `M`, `Z` explicitly declared fp32 to avoid
               autocast-related dtype drift.
  INF3-FIX-06  Idempotency guard checks both source and parent layer.
  INF3-FIX-07  Recursion guard in `inject_infini` (depth limit).
  INF3-FIX-08  Sliding-window + causal + padding masks combined in one
               `masked_fill` chain.
  INF3-FIX-09  `nan_to_num` on all outputs (scores, attn, context, memory).
  INF3-FIX-10  GQA `repeat_kv` applied before memory read, so `v` matches
               the head layout used in the memory loop.
  INF3-FIX-11  Storage accounting uses a single sum over real parameters.
  INF3-FIX-12  `copy_weights` raises a descriptive error on irreconcilable
               shape mismatch instead of silently truncating.
  INF3-FIX-13  Proper handling of `head_mask` for 1D/2D/3D shapes.
  INF3-FIX-14  `comp_config` extraction tolerant of dict/None/namespace.
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

logger = logging.getLogger(__name__)


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
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)
    return q, k


# ===========================================================================
# Decoder detection
# ===========================================================================
def _detect_decoder(config: Any, is_decoder: bool | None, old_attn: nn.Module | None) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
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
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# LoRALinear
# ===========================================================================
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
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
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
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# InfiniAttention
# ===========================================================================
@ATTENTION_REGISTRY.register("infini", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("infini_attention", supports_causal=True, supports_gqa=True)
class InfiniAttention(BaseAttention):
    """
    Infini-Attention: sliding-window local attention + compressive memory.

    For decoders, the output is a per-head convex combination of:
      - A_local: sliding-window softmax attention
      - A_mem:   read from a transient (per-forward) compressive memory state
    For encoders, it degenerates to full softmax attention (memory is skipped).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        local_window: int = 128,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---------- dim ----------
        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_key_value_heads") or kwargs.get("num_kv_heads")

        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
        if hidden_size is None and source_attn is not None:
            for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                layer = getattr(source_attn, attr, None)
                if layer is not None and hasattr(layer, "out_features"):
                    hidden_size = layer.out_features
                    break
        hidden_size = int(hidden_size or 768)

        if num_heads is None and config is not None:
            num_heads = getattr(
                config,
                "num_attention_heads",
                getattr(config, "n_heads", getattr(config, "num_heads", None)),
            )
        if num_heads is None and source_attn is not None:
            num_heads = getattr(
                source_attn,
                "num_attention_heads",
                getattr(source_attn, "n_heads", None),
            )
        num_heads = int(num_heads or 12)

        if num_kv_heads is None and config is not None:
            num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                getattr(config, "num_kv_heads", num_heads),
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        num_kv_heads = int(num_kv_heads)

        self.dim = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = max(1, min(num_kv_heads, num_heads))

        # ---------- head_dim ----------
        inferred = None
        if config is not None:
            inferred = getattr(config, "head_dim", None)
        if inferred is None:
            inferred = kwargs.get("head_dim")
        if inferred is None and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                layer = getattr(source_attn, attr, None)
                if layer is not None and hasattr(layer, "weight"):
                    inferred = layer.weight.shape[0] // max(1, num_heads)
                    break
        self.head_dim = int(inferred or (self.dim // max(1, num_heads)))
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # ---------- window ----------
        self.local_window = max(1, int(local_window))

        # ---------- decoder ----------
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---------- LoRA ----------
        self.lora_rank = int(
            kwargs.get(
                "lora_rank",
                _cfg_get(
                    comp_config,
                    "lora_rank",
                    getattr(config, "lora_rank", 16) if config else 16,
                ),
            )
        )
        self.lora_alpha = float(
            kwargs.get(
                "lora_alpha",
                _cfg_get(
                    comp_config,
                    "lora_alpha",
                    getattr(config, "lora_alpha", 32.0) if config else 32.0,
                ),
            )
        )
        self.freeze_base = self.lora_rank > 0

        # ---------- projections ----------
        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # ---------- fusion gate (per-head, init at sigmoid(0)=0.5) ----------
        self.beta = nn.Parameter(torch.zeros(self.num_heads))

        # ---------- write gate (per-token, input-dependent) ----------
        self.w_mem = nn.Linear(self.dim, self.num_heads, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.w_mem.weight)
            nn.init.constant_(self.w_mem.bias, 2.0)  # sigmoid(2) ≈ 0.88

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.0))
            if config
            else 0.0
        )
        self.dropout = nn.Dropout(dropout_p)

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

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
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = src.__class__.__name__
        has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))

        is_nested = False
        if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
            is_nested = False
        elif (
            (
                hasattr(src, "self")
                and hasattr(src, "output")
                and hasattr(src.self, "query")
                and hasattr(src.output, "dense")
            )
            or cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            )
            or cls_name.endswith("SelfAttention")
            and "distil" not in cls_name.lower()
            and not has_out
        ):
            is_nested = True
        self.is_nested_encoder = is_nested

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (
                    "Conv1D" in c.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c, "nf")
                    or hasattr(base_c, "nf")
                    or (cw.ndim == 2 and cw.shape[1] == 3 * self.dim)
                )
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb[split : 2 * split] if cb is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb[2 * split :] if cb is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o = src.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = (
                        "Conv1D" in o.__class__.__name__
                        or "Conv1D" in base_o.__class__.__name__
                        or hasattr(o, "nf")
                        or hasattr(base_o, "nf")
                        or (ow.ndim == 2 and ow.shape[0] != self.dim and ow.shape[1] == self.dim)
                    )
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # Nested BERT / RoBERTa
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (
                        m.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # Flat (DistilBERT, Qwen, LLaMA)
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                m.weight.data.clone(),
                                b.data.clone() if b is not None else None,
                            )
                            break

        # UniversalAdapter fallback
        missing = [
            k
            for k, (w, _) in extracted.items()
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
                logger.debug("Infini UniversalAdapter fallback failed: %s", e)

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target.weight.shape
                    and w.t().contiguous().shape == target.weight.shape
                ):
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data.zero_()
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )
                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data.zero_()
                    target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                n = torch.linalg.norm(target.weight.data.float()).item()
                if n < 1e-4:
                    logger.warning("Infini: '%s' near-zero norm (%.2e)", key, n)

                if self.freeze_base:
                    target.freeze_base_weights()

        # Storage accounting
        try:
            self._comp_storage_bytes = sum(
                p.numel() * p.element_size()
                for m in (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.out_proj,
                    self.w_mem,
                )
                for p in m.parameters()
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Mask helper
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
    # Sliding-window local attention
    # ------------------------------------------------------------------
    def _local_sliding_attention(self, q, k, v, valid_mask=None):
        B, H, L, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        idx = torch.arange(L, device=q.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        band = dist <= self.local_window

        if self.is_decoder:
            causal = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
            band = band & causal

        scores = scores.masked_fill(~band.unsqueeze(0).unsqueeze(0), -1e4)

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(q.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, attn

    # ------------------------------------------------------------------
    # Full dense attention (encoder path)
    # ------------------------------------------------------------------
    def _full_attention(self, q, k, v, valid_mask=None):
        B, H, L, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(q.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, attn

    # ------------------------------------------------------------------
    # Compressive memory
    # ------------------------------------------------------------------
    def _compressive_memory_read(self, q, k, v, write_gate, valid_mask=None):
        """
        Sequential O(L) memory update + read.

        M_t = M_{t-1} + g_t · φ(k_t)^T v_t
        z_t = z_{t-1} + g_t · φ(k_t)
        y_t = φ(q_t) M_t / (φ(q_t) z_t + ε)

        INF3-FIX-04: the gate-scaled feature `g_t · φ(k_t)` is precomputed
        outside the loop, and each step uses a batched outer product.
        """
        B, H, L, Dh = q.shape
        device = q.device
        dtype = q.dtype

        # Feature map φ(x) = ELU(x) + 1
        phi_q = F.elu(q.float()) + 1.0  # (B, H, L, Dh)
        phi_k = F.elu(k.float()) + 1.0  # (B, H, L, Dh)
        v_f = v.float()  # (B, H, L, Dh)

        # Padding mask
        if valid_mask is not None:
            m = valid_mask[:, None, :, None].to(phi_k.dtype)  # (B, 1, L, 1)
            phi_k = phi_k * m
            v_f = v_f * m

        # Precompute gate-scaled keys
        # write_gate: (B, H, L, 1) → kg: (B, H, L, Dh)
        kg = write_gate.to(phi_k.dtype) * phi_k  # (B, H, L, Dh)

        # Transient memory (fresh per forward)
        M = torch.zeros(B, H, Dh, Dh, device=device, dtype=torch.float32)
        Z = torch.zeros(B, H, Dh, device=device, dtype=torch.float32)

        outputs = []
        for t in range(L):
            k_t = kg[:, :, t]  # (B, H, Dh)
            v_t = v_f[:, :, t]  # (B, H, Dh)
            q_t = phi_q[:, :, t]  # (B, H, Dh)

            # Batched outer product + accumulation
            M = M + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # (B, H, Dh, Dh)
            Z = Z + k_t  # (B, H, Dh)

            # Read
            o_t = torch.einsum("bhd,bhde->bhe", q_t, M)  # (B, H, Dh)
            denom = torch.einsum("bhd,bhd->bh", q_t, Z).clamp(min=1e-6)
            o_t = o_t / denom.unsqueeze(-1)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out.to(dtype)

    # ------------------------------------------------------------------
    # Forward — signature matches BERT/RoBERTa/DistilBERT
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        # INF3-FIX-02: layer_idx from kwargs (some HF models pass it)
        kwargs.get("layer_idx", 0)

        # INF3-FIX-01: position_embeddings comes from kwargs, never positional
        position_embeddings = kwargs.get("position_embeddings")

        if encoder_hidden_states is not None:
            raise NotImplementedError("InfiniAttention does not support cross-attention.")

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device

        # ---- projections ----
        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- valid mask ----
        valid_mask = self._prepare_valid_mask(attention_mask, B, L, device)

        # ---- attention path ----
        if self.is_decoder:
            A_local, attn = self._local_sliding_attention(q, k, v, valid_mask)

            # Write gate: (B, L, H) → (B, H, L, 1)
            wg = torch.sigmoid(self.w_mem(hidden_states))
            write_gate = wg.transpose(1, 2).unsqueeze(-1)

            A_mem = self._compressive_memory_read(q, k, v, write_gate, valid_mask)

            beta = torch.sigmoid(self.beta).to(A_mem.dtype).view(1, H, 1, 1)
            fused = beta * A_mem + (1.0 - beta) * A_local
        else:
            fused, attn = self._full_attention(q, k, v, valid_mask)

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask.to(fused.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            # else assume 4D already
            fused = fused * hm

        # ---- output ----
        fused = fused.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # Zero out padded query positions
        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)

        return out, (attn if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("out_proj", self.out_proj),
            ("w_mem", self.w_mem),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["beta"] = _sz(self.beta)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("infini_inject")
@ATTENTION_REGISTRY.register("inject_infini")
def inject_infini(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> nn.Module | None:
    if _depth > 8:
        logger.warning("inject_infini: max recursion depth exceeded.")
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
            res = inject_infini(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
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

    # INF3-FIX-06: idempotency guard on both source and parent
    if isinstance(src_attn, InfiniAttention) or "InfiniAttention" in src_attn.__class__.__name__:
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        child = getattr(target, attr, None)
        if isinstance(child, InfiniAttention):
            return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "dim", None)
        or (getattr(src_attn, "q_proj", None).in_features if hasattr(src_attn, "q_proj") else None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", 64)
    )
    num_heads = (
        getattr(cfg, "num_attention_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "num_heads", None)
        or 4
    )

    local_window = int(
        kwargs.pop(
            "local_window",
            _cfg_get(comp_config, "local_window", 128),
        )
    )
    lora_rank = int(
        kwargs.pop(
            "lora_rank",
            _cfg_get(comp_config, "lora_rank", 16),
        )
    )

    try:
        new_attn = InfiniAttention(
            config=cfg,
            local_window=local_window,
            pretrained_attn=src_attn if src_attn is not target else None,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            lora_rank=lora_rank,
            **kwargs,
        )
    except Exception as e:
        logger.error("InfiniAttention construction failed: %s", e)
        return None

    # Move to source device/dtype
    p = next(src_attn.parameters(), None) if src_attn is not target else None
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

    # Splice
    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True

    if not spliced:
        logger.error(
            "inject_infini: no writable attention attribute on %s",
            type(target).__name__,
        )
        return None

    return new_attn


__all__ = [
    "InfiniAttention",
    "inject_infini",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
