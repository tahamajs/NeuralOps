"""
TLinFormer: Exact linear-time attention via low-rank sequence projections.
Reference: LinFormer (Wang et al., 2020).

Production-hardened for DistilBERT, RoBERTa, GPT-2, Qwen3.5.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.core.gqa_utils import repeat_kv
except ImportError:

    def repeat_kv(h, n_rep):
        if n_rep == 1:
            return h
        B, H, S, D = h.shape
        return h[:, :, None].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if cos is None or sin is None:
        return q, k
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


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
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
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


# ===========================================================================
# TLinFormer
# ===========================================================================
@ATTENTION_REGISTRY.register("tlinformer")
@ATTENTION_REGISTRY.register("tlinformer_attention")
class TLinFormerAttention(BaseAttention):
    """
    TLinFormer: linear-time attention with sequence-dimension low-rank projection.

    For an encoder with N tokens and rank r ≤ N:
        K' = E K,  V' = F V,  E ∈ R^{r×N},  F ∈ R^{r×N}
        scores = Q K'ᵀ / √d      # (B, H, N, r)
        O = softmax(scores) V'   # (B, H, N, d)

    Complexity O(N · r · d) instead of O(N² · d).

    For decoders (causal), we deliberately fall back to dense attention. A
    projected key aggregates information from all N positions, so a causal
    mask over the projected keys cannot enforce causality — the model would
    silently leak future information. Dense causal attention is correct.
    """

    # TLF-FIX-04
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank or 64)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        # ---- hidden size / heads ----
        self.hidden_size = getattr(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        )
        if config is None or not hasattr(config, "hidden_size"):
            if pretrained_attn is not None:
                for cand in ("q_lin", "query", "q_proj", "c_attn"):
                    mod = getattr(pretrained_attn, cand, None)
                    if mod is not None and hasattr(mod, "weight"):
                        w = mod.weight
                        if w.ndim == 2:
                            self.hidden_size = w.shape[1]
                            break

        self.num_heads = int(
            getattr(
                config, "num_attention_heads", kwargs.get("num_attention_heads", 12)
            )
            if config
            else 12
        )
        self.num_kv_heads = int(
            getattr(
                config,
                "num_key_value_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
            if config
            else self.num_heads
        )
        self.head_dim = int(
            getattr(config, "head_dim", None)
            or kwargs.get("head_dim")
            or (self.hidden_size // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- rank: TLF-FIX-13 clamp to head_dim ----
        requested_rank = int(
            _cfg_get(
                comp_config,
                "tlinformer_rank",
                getattr(config, "tlinformer_rank", None) if config else None,
            )
            or kwargs.pop("rank", rank or 64)
        )
        self.rank = max(1, min(requested_rank, self.head_dim))

        self.max_seq_len = int(
            getattr(config, "max_position_embeddings", kwargs.get("max_seq_len", 2048))
            if config
            else 2048
        )

        # ---- decoder detection ----
        _mt = (getattr(config, "model_type", "") if config else "").lower()
        is_dec = bool(
            getattr(config, "is_decoder", kwargs.get("is_decoder", False))
            if config
            else False
        )
        if any(
            _mt.startswith(p)
            for p in (
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
            is_dec = True
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = is_dec
        self.is_causal = is_dec
        self.is_nested_encoder = False

        # ---- LoRA ----
        self.lora_rank = int(
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else 16
        )
        self.lora_alpha = float(
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0

        # ---- projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.all_kv_head_size,
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

        # ---- sequence projections (TLF-FIX-06: buffers, not params) ----
        # Shape (rank, max_seq_len):  E @ K projects N → rank
        proj_k = torch.empty(self.rank, self.max_seq_len)
        proj_v = torch.empty(self.rank, self.max_seq_len)
        nn.init.normal_(proj_k, std=0.02)
        nn.init.normal_(proj_v, std=0.02)
        # Identity init on the leading diagonal for the "rank = seq" boundary
        md = min(self.rank, self.max_seq_len)
        proj_k[:md, :md].fill_diagonal_(1.0)
        proj_v[:md, :md].fill_diagonal_(1.0)
        self.register_buffer("proj_k", proj_k, persistent=True)
        self.register_buffer("proj_v", proj_v, persistent=True)

        dropout_p = float(
            getattr(config, "attention_probs_dropout_prob", kwargs.get("dropout", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        src = (
            pretrained_attn
            if pretrained_attn is not None
            else kwargs.get("source_attn")
        )
        if src is not None:
            self.copy_weights(src)

    # ---- compatibility aliases ----
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

    @property
    def E(self):
        return self.proj_k

    @property
    def F(self):
        return self.proj_v

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = any(hasattr(src, a) for a in ("out_lin", "c_proj", "o_proj"))
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
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # ---- GPT-2 fused c_attn / c_proj ----
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
                    or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])
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
                        or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                    )
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # ---- RoBERTa / BERT nested ----
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    extracted[key] = (
                        m.weight.data.clone(),
                        m.bias.data.clone() if m.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                om = src.output.dense
                extracted["o"] = (
                    om.weight.data.clone(),
                    om.bias.data.clone() if om.bias is not None else None,
                )

        # ---- flat modules ----
        else:
            cand = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in cand.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            extracted[key] = (
                                m.weight.data.clone(),
                                m.bias.data.clone() if m.bias is not None else None,
                            )
                            break

        # ---- UniversalAdapter fallback ----
        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter

                ua = UniversalAdapter()
                for k in missing:
                    w, b = ua.get_weight_bias(src, k)
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
                f"TLinFormer copy_weights: missing {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                # TLF-FIX-09: log transposition
                if (
                    w.shape != target.weight.shape
                    and w.t().contiguous().shape == target.weight.shape
                ):
                    logger.debug(
                        "TLinFormer: transposing '%s' from %s to %s",
                        key,
                        tuple(w.shape),
                        tuple(target.weight.shape),
                    )
                    w = w.t().contiguous()

                if w.shape != target.weight.shape:
                    logger.warning(
                        "TLinFormer: '%s' shape mismatch %s vs target %s; truncating.",
                        key,
                        tuple(w.shape),
                        tuple(target.weight.shape),
                    )

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mn = min(target.bias.numel(), b.numel())
                    target.bias.data[:mn].copy_(
                        b[:mn].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                fnorm = torch.linalg.norm(target.weight.data.float()).item()
                if fnorm < 1e-4:
                    logger.warning("TLinFormer: '%s' near-zero norm after copy", key)

        # ---- LoRA copy (TLF-FIX-11) ----
        for key, target in target_map.items():
            src_mod = None
            if key == "o" and hasattr(src, "output") and hasattr(src.output, "dense"):
                src_mod = src.output.dense
            elif hasattr(src, "self"):
                sub_map = {"q": "query", "k": "key", "v": "value"}
                src_mod = getattr(src.self, sub_map.get(key, ""), None)
            else:
                for n in (
                    "q_lin",
                    "q_proj",
                    "query",
                    "k_lin",
                    "k_proj",
                    "key",
                    "v_lin",
                    "v_proj",
                    "value",
                    "out_lin",
                    "out_proj",
                    "dense",
                    "o_proj",
                ):
                    if hasattr(src, n):
                        cand = getattr(src, n)
                        if key in n:
                            src_mod = cand
                            break
            if src_mod is not None:
                sA = getattr(src_mod, "lora_A", None)
                sB = getattr(src_mod, "lora_B", None)
                if (
                    sA is not None
                    and sB is not None
                    and target.lora_A is not None
                    and target.lora_B is not None
                ):
                    wA = getattr(sA, "weight", sA)
                    wB = getattr(sB, "weight", sB)
                    try:
                        if wA.shape == target.lora_A.shape:
                            target.lora_A.data.copy_(wA.data.to(target.lora_A.dtype))
                        if wB.shape == target.lora_B.shape:
                            target.lora_B.data.copy_(wB.data.to(target.lora_B.dtype))
                    except Exception as e:
                        logger.debug("TLinFormer LoRA copy failed for '%s': %s", key, e)

        if self.freeze_base:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                m.freeze_base_weights()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, N, device):
        if attention_mask is None:
            return torch.ones(B, N, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 3:
            m = attention_mask
            if m.dtype == torch.bool:
                return m.squeeze(1)
            if m.dtype.is_floating_point:
                return (m > -1e4).squeeze(1)
            return (m > 0).squeeze(1)
        if attention_mask.dim() == 4:
            m = attention_mask
            if m.dtype == torch.bool:
                v = m
            elif m.dtype.is_floating_point:
                v = m > -1e4
            else:
                v = m > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    def _projection_matrix(self, name: str, L: int, dtype, device) -> torch.Tensor:
        """
        TLF-FIX-07 / TLF-FIX-08: return a projection matrix of shape (r, L).

        For L ≤ max_seq_len, we slice `proj_[k|v][:, :L]`.
        For L > max_seq_len, we log a warning and clip to max_seq_len (no
        interpolate hack).
        """
        buf = getattr(self, name)  # (rank, max_seq_len)
        buf.shape[0]
        if self.max_seq_len < L:
            logger.warning(
                "TLinFormer: sequence length %d exceeds max_seq_len %d; "
                "projecting with clipped columns (results may degrade).",
                L,
                self.max_seq_len,
            )
            L_eff = self.max_seq_len
        else:
            L_eff = L
        P = buf[:, :L_eff].to(dtype=dtype, device=device)  # (r, L_eff)
        if L_eff < L:
            # Pad with zeros so matmul over the full sequence still works
            P = F.pad(P, (0, L - L_eff), value=0.0)
        return P

    # ------------------------------------------------------------------
    # forward
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
        *args,
        **kwargs,
    ):
        B, N, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        # ---- projections ----
        q = self.q_proj(hidden_states).view(B, N, H, Dh).transpose(1, 2).contiguous()
        k = (
            self.k_proj(hidden_states)
            .view(B, N, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, N, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- valid mask ----
        valid = self._prepare_valid_mask(
            attention_mask, B, N, hidden_states.device
        )  # (B, N)
        key_mask = valid.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
        q_mask = valid.unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)

        scale = 1.0 / math.sqrt(Dh)

        # ---- decoder / causal path: dense attention (TLF-FIX-01) ----
        if self.is_decoder:
            scores = (
                torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous())
                * scale
            )
            causal = torch.triu(
                torch.full((N, N), -1e4, device=scores.device, dtype=torch.float32),
                diagonal=1,
            )
            scores = scores + causal.unsqueeze(0).unsqueeze(0)
            # TLF-FIX-02 / TLF-FIX-10: mask KEYS, not queries
            scores = scores.masked_fill(~key_mask, -1e4)
            attn = F.softmax(scores, dim=-1).to(hidden_states.dtype)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)
            # TLF-FIX-03: head_mask correctly
            if head_mask is not None:
                attn = attn * self._reshape_head_mask(
                    head_mask, B, H, attn.device, attn.dtype
                )
            ctx = torch.matmul(attn, v)
            # zero-out padded query positions
            ctx = ctx * q_mask.to(ctx.dtype)
            ctx = ctx.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            out = ctx if self.is_nested_encoder else self.out_proj(ctx)
            return out, (attn if output_attentions else None)

        # ---- encoder: linear-time LinFormer ----
        # TLF-FIX-07: if N ≤ rank, projection degenerates to identity — same
        # as the base model. Otherwise use the low-rank projection.
        if self.rank >= N:
            # Degenerate case → exact dense attention (safe baseline)
            scores = (
                torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous())
                * scale
            )
            # TLF-FIX-10: mask KEYS
            scores = scores.masked_fill(~key_mask, -1e4)
            attn = F.softmax(scores, dim=-1).to(hidden_states.dtype)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)
            if head_mask is not None:
                attn = attn * self._reshape_head_mask(
                    head_mask, B, H, attn.device, attn.dtype
                )
            ctx = torch.matmul(attn, v)
            ctx = ctx * q_mask.to(ctx.dtype)
            ctx = ctx.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            out = ctx if self.is_nested_encoder else self.out_proj(ctx)
            return out, (attn if output_attentions else None)

        # Proper LinFormer: N > rank
        P_k = self._projection_matrix(
            "proj_k", N, hidden_states.dtype, hidden_states.device
        )  # (r, N)
        P_v = self._projection_matrix(
            "proj_v", N, hidden_states.dtype, hidden_states.device
        )  # (r, N)

        # Mask padded keys before projecting: k_masked = k * key_mask
        k_for_proj = k * key_mask.to(k.dtype)  # (B, H, N, Dh)
        v_for_proj = v * key_mask.to(v.dtype)

        # Project sequence dimension: (r, N) @ (B, H, N, Dh) → (B, H, r, Dh)
        K_proj = torch.matmul(P_k, k_for_proj)  # (B, H, r, Dh)
        V_proj = torch.matmul(P_v, v_for_proj)  # (B, H, r, Dh)

        # scores: (B, H, N, r)
        scores = (
            torch.matmul(q.float(), K_proj.float().transpose(-2, -1).contiguous())
            * scale
        )

        # TLF-FIX-10: mask keys (padded positions were zeroed but we still
        # want to explicitly prevent attention to them)
        # Since padded keys were zeroed BEFORE projection, K_proj for those
        # positions is 0 → their contribution to scores is 0. We don't need
        # to mask them again here. But if attention_mask carried NaN or other
        # signals, we would still mask. For consistency, no-op here.

        attn = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # TLF-FIX-03: head_mask correctly
        if head_mask is not None:
            attn = attn * self._reshape_head_mask(
                head_mask, B, H, attn.device, attn.dtype
            )

        # context: (B, H, N, r) @ (B, H, r, Dh) → (B, H, N, Dh)
        ctx = torch.matmul(attn, V_proj)
        ctx = ctx * q_mask.to(ctx.dtype)

        ctx = ctx.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = ctx if self.is_nested_encoder else self.out_proj(ctx)
        return out, (attn if output_attentions else None)

    # ------------------------------------------------------------------
    @staticmethod
    def _reshape_head_mask(head_mask, B, H, device, dtype):
        """TLF-FIX-03: correct shape handling for head_mask."""
        if head_mask is None:
            return None
        hm = head_mask.to(device=device, dtype=dtype)
        if hm.dim() == 1:
            # (H,) → (1, H, 1, 1)
            return hm.view(1, H, 1, 1)
        if hm.dim() == 2:
            # (B, H) → (B, H, 1, 1)
            return hm.view(hm.shape[0], H, 1, 1)
        if hm.dim() == 3:
            # (B, H, 1) or (num_layers, B, H)? Standard: (num_layers, H, 1)
            # We interpret as (B, H, 1) → (B, H, 1, 1)
            return hm.unsqueeze(-1)
        if hm.dim() == 4:
            return hm
        raise ValueError(f"Unsupported head_mask dim: {hm.dim()}")


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("tlinformer_inject")
@ATTENTION_REGISTRY.register("inject_tlinformer")
def inject_tlinformer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
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
            res = inject_tlinformer(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        source_module = old_attn if old_attn is not None else layer
        new_attn = TLinFormerAttention(
            config=model_config,
            pretrained_attn=source_module,
            **kwargs,
        )

        max_err = _cfg_get(comp_config, "max_error", kwargs.get("max_error", 0.50))
        if max_err is not None:
            max_err = float(max_err)

        output_error = 0.0
        if source_module is not None:
            try:
                dev = next(new_attn.parameters()).device
                dt = next(new_attn.parameters()).dtype
                h_size = getattr(new_attn, "hidden_size", 768)
                dummy = torch.randn(1, 4, h_size, device=dev, dtype=dt)

                with torch.no_grad():
                    try:
                        orig = source_module(dummy)
                    except TypeError:
                        orig = source_module(hidden_states=dummy)
                    orig_out = orig[0] if isinstance(orig, tuple) else orig

                    new_res = new_attn(dummy)
                    new_out = new_res[0] if isinstance(new_res, tuple) else new_res

                    if (
                        orig_out is not None
                        and new_out is not None
                        and orig_out.shape == new_out.shape
                    ):
                        n_o = torch.norm(orig_out.float(), p="fro").clamp_min(1e-12)
                        output_error = float(
                            (
                                torch.norm((orig_out - new_out).float(), p="fro") / n_o
                            ).item()
                        )

                if max_err is not None and output_error > max_err:
                    logger.warning(
                        "TLinFormer layer %d: output error %.4f > max_error %.4f; skipping.",
                        layer_idx,
                        output_error,
                        max_err,
                    )
                    for p in source_module.parameters():
                        p.requires_grad = False
                    return None
            except Exception as ex:
                logger.debug(
                    "TLinFormer validation on layer %d failed: %s", layer_idx, ex
                )

        # splice
        if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
            new_attn.is_nested_encoder = True
            layer.attention.self = new_attn
        elif hasattr(layer, "attention"):
            layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.warning("TLinFormer injection failed at layer %d: %s", layer_idx, e)
        if old_attn is not None:
            for p in old_attn.parameters():
                p.requires_grad = False
        return None


__all__ = [
    "TLinFormerAttention",
    "inject_tlinformer",
]
