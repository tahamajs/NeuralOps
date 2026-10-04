"""
HGRN-2 Outer-Product Recurrent Linear Attention — Production-Hardened v2.

Multi-head bidirectional/causal outer-product associative matrix memory
updates with log-space bounded decay gating and denominator normalization.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
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

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


# ===========================================================================
# LoRALinear (HGRN2-FIX-04, HGRN2-FIX-07)
# ===========================================================================
class LoRALinear(nn.Module):
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, **fk))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, **fk))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            out = out + F.linear(F.linear(x, a), bb) * self.scaling
        return out


# ===========================================================================
# HGRN2OuterBlock
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "hgrn2_outer", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class HGRN2OuterBlock(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | Any | None = None,
        num_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- dim resolution ----
        if dim is None:
            if isinstance(config, int):
                dim = config
            elif config is not None:
                dim = getattr(
                    config,
                    "hidden_size",
                    getattr(
                        config,
                        "d_model",
                        getattr(config, "n_embd", getattr(config, "dim", None)),
                    ),
                )
        if dim is None:
            dim = kwargs.get("hidden_size", kwargs.get("d_model", 768))
        if hasattr(dim, "hidden_size"):
            dim = dim.hidden_size

        # ---- num_heads ----
        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = getattr(
                    config,
                    "num_attention_heads",
                    getattr(
                        config,
                        "num_heads",
                        getattr(config, "n_heads", getattr(config, "n_head", None)),
                    ),
                )
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        while self.num_heads > 1 and (self.dim % self.num_heads != 0):
            self.num_heads -= 1
        self.head_dim = self.dim // max(1, self.num_heads)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.chunk_size = int(chunk_size)

        cfg_kv = None
        if config is not None and not isinstance(config, int):
            cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
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
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False) if config else False)
                or bool(getattr(config, "causal", False) if config else False)
                or any(_mt.startswith(p) for p in _dec_pfx)
                or bool(kwargs.get("is_causal", False))
                or bool(kwargs.get("is_decoder", False))
            )

        self.is_nested_encoder: bool = False

        # ---- projections ----
        self.q_proj = LoRALinear(
            self.dim,
            self.q_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.W_f = nn.Linear(self.dim, self.num_heads, bias=True)
        self.state_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        with torch.no_grad():
            nn.init.normal_(self.W_f.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.W_f.bias, -2.5)

        # ---- storage ----
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
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
    def c_proj(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            cls_name = old_attn.__class__.__name__
            if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # GPT-2 fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                cls_c = src.c_attn.__class__.__name__
                if (
                    hasattr(src.c_attn, "nf")
                    or "Conv1D" in cls_c
                    or (c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim)
                ):
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size
                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w = src.c_proj.weight.detach()
                cls_p = src.c_proj.__class__.__name__
                if (
                    hasattr(src.c_proj, "nf")
                    or "Conv1D" in cls_p
                    or (out_w.ndim == 2 and out_w.shape[0] != self.dim)
                ):
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # BERT / RoBERTa nested
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Qwen / LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None:
                missing.append("Q")
            if w_k is None:
                missing.append("K")
            if w_v is None:
                missing.append("V")
            if w_o is None and not self.is_nested_encoder:
                missing.append("O")
            if missing:
                logger.warning(
                    "HGRN2OuterBlock: missing weights %s from %s",
                    missing,
                    type(old_attn).__name__,
                )
                return

            # Storage tracking
            try:
                for p in old_attn.parameters():
                    self._orig_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

            def _copy_to(target_mod: nn.Module, w_src: torch.Tensor, b_src):
                tw = target_mod.weight
                w_cast = w_src.to(tw.dtype)
                if w_cast.shape == tw.shape:
                    tw.data.copy_(w_cast)
                else:
                    min_r = min(tw.shape[0], w_cast.shape[0])
                    min_c = min(tw.shape[1], w_cast.shape[1])
                    tw.data.zero_()
                    tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])
                if b_src is not None and target_mod.bias is not None:
                    tb = target_mod.bias
                    b_cast = b_src.to(tb.dtype)
                    mb = min(tb.numel(), b_cast.numel())
                    tb.data.zero_()
                    tb.data[:mb].copy_(b_cast[:mb])

            _copy_to(self.q_proj, w_q, b_q)
            _copy_to(self.k_proj, w_k, b_k)
            _copy_to(self.v_proj, w_v, b_v)

            if w_o is not None:
                _copy_to(self.out_proj, w_o, b_o)

            # Norm checks
            for name, mod in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                n = float(torch.linalg.norm(mod.weight.float()))
                if n < 1e-4:
                    logger.warning("HGRN2OuterBlock: %s weight near-zero norm (%.2e)", name, n)

            try:
                for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                    for p in m.parameters():
                        self._comp_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HGRN2-FIX-01: chunked scan (vectorized)
    # ------------------------------------------------------------------
    def _chunked_scan(
        self,
        q: torch.Tensor,  # (B, H, L, d)
        k: torch.Tensor,  # (B, H, L, d)
        v: torch.Tensor,  # (B, H, L, d)
        f: torch.Tensor,  # (B, H, L)
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Chunked causal scan for outer-product recurrent linear attention.

        Recurrence:
            S_t = f_t * S_{t-1} + (1 - f_t) * v_t k_t^T
            z_t = f_t * z_{t-1} + (1 - f_t) * k_t
            y_t = (q_t @ S_t) / (q_t @ z_t)

        Within a chunk of size C, both numerator and denominator are computed
        in closed form via a causal decay matrix:
            decay[i, j] = exp(log_F[j] - log_F_prev[i])
        where log_F[j] = sum_{s<=j} log f_s within the chunk.
        """
        B, H, L, d = q.shape
        num_chunks = (L + chunk_size - 1) // chunk_size

        S_prev = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        z_prev = torch.zeros(B, H, d, device=q.device, dtype=q.dtype)

        outputs = []

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, L)
            C = end - start

            q_c = q[:, :, start:end]  # (B, H, C, d)
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            f_c = f[:, :, start:end]  # (B, H, C)

            # ---- within-chunk cumulative log-deccay ----
            log_f = torch.log(f_c.clamp_min(1e-6))  # (B, H, C)
            log_F_c = torch.cumsum(log_f, dim=-1)  # (B, H, C)

            # Extended prev: [0, log_F_c[0], ..., log_F_c[C-1]]
            log_F_prev = torch.cat(
                [
                    torch.zeros(B, H, 1, device=q.device, dtype=log_F_c.dtype),
                    log_F_c,
                ],
                dim=-1,
            )  # (B, H, C+1)

            # log_decay[i, j] = log_F_c[j] - log_F_prev[i]
            # i = 0 → previous-chunk state
            # i >= 1 → position i-1 in chunk
            log_decay = log_F_c.unsqueeze(-2) - log_F_prev.unsqueeze(-1)  # (B, H, C+1, C)

            # Mask: i=0 always valid; i>=1 valid iff i-1 <= j  (i <= j+1)
            i_idx = torch.arange(C + 1, device=q.device).view(C + 1, 1)
            j_idx = torch.arange(C, device=q.device).view(1, C)
            valid = (i_idx == 0) | (i_idx <= j_idx + 1)
            fill_val = -1e4 if log_decay.dtype == torch.float16 else -1e9
            log_decay = log_decay.masked_fill(~valid, fill_val)
            decay = torch.exp(log_decay).to(q.dtype)  # (B, H, C+1, C)

            # ---- within-chunk write contributions ----
            one_minus_f = (1.0 - f_c).to(q.dtype)  # (B, H, C)
            W_kv = one_minus_f[..., None, None] * (
                k_c.unsqueeze(-1) * v_c.unsqueeze(-2)
            )  # (B, H, C, d, d)
            W_k = one_minus_f[..., None] * k_c  # (B, H, C, d)

            # Combined with prev state as index 0
            combined_kv = torch.cat([S_prev.unsqueeze(2), W_kv], dim=2).to(
                q.dtype
            )  # (B, H, C+1, d, d)
            combined_k = torch.cat([z_prev.unsqueeze(2), W_k], dim=2).to(q.dtype)  # (B, H, C+1, d)

            # S_all[j] = sum_i decay[i, j] * combined[i]
            S_all = torch.einsum("bhij,bhiae->bhjae", decay, combined_kv)  # (B, H, C, d, d)
            z_all = torch.einsum("bhij,bhia->bhja", decay, combined_k)  # (B, H, C, d)

            # y_j = (q_j @ S_j) / (q_j @ z_j)
            num = torch.einsum("bhjd,bhjde->bhje", q_c, S_all)  # (B, H, C, d)
            denom = torch.einsum("bhjd,bhjd->bhj", q_c, z_all).clamp_min(1e-6)  # (B, H, C)
            y_c = num / denom.unsqueeze(-1)

            outputs.append(y_c)
            S_prev = S_all[:, :, -1]  # (B, H, d, d)
            z_prev = z_all[:, :, -1]  # (B, H, d)

        return torch.cat(outputs, dim=2).to(q.dtype)  # (B, H, L, d)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        return {
            "orig": self._orig_storage_bytes,
            "comp": self._comp_storage_bytes,
        }

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
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("HGRN2OuterBlock does not support cross-attention.")

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # ---- RoPE ----
        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- GQA ----
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # ---- Non-negative kernel ----
        q = F.elu(q) + 1.0
        k = (F.elu(k) + 1.0) / math.sqrt(d)

        # ---- Bounded forget gate ----
        f = torch.exp((-F.softplus(self.W_f(hidden_states))).clamp(min=-3.0, max=-1e-4)).transpose(
            1, 2
        )  # (B, H, L)

        # ---- Mask extraction (HGRN2-FIX-10) ----
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    mask_bool = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    mask_bool = attention_mask > -1e4
                else:
                    mask_bool = attention_mask > 0
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    m = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    m = attention_mask > -1e4
                else:
                    m = attention_mask > 0
                mask_bool = m.any(dim=-2).squeeze(1)

            if mask_bool is not None and mask_bool.shape[-1] != L:
                mask_bool = None

        # Pre-mask k, v
        if mask_bool is not None:
            m_exp = mask_bool.unsqueeze(1).unsqueeze(-1).to(k.dtype)  # (B, 1, L, 1)
            k = k * m_exp
            v = v * m_exp

        # ---- Chunked scan ----
        y_fwd = self._chunked_scan(q, k, v, f, chunk_size=self.chunk_size)

        if self.is_decoder:
            y = y_fwd
        else:
            q_r = torch.flip(q, dims=[2])
            k_r = torch.flip(k, dims=[2])
            v_r = torch.flip(v, dims=[2])
            f_r = torch.flip(f, dims=[2])
            y_bwd_r = self._chunked_scan(q_r, k_r, v_r, f_r, chunk_size=self.chunk_size)
            y_bwd = torch.flip(y_bwd_r, dims=[2])
            y = 0.5 * (y_fwd + y_bwd)
            y = y.to(hidden_states.dtype)

        sn_dtype = (
            self.state_norm.weight.dtype
            if hasattr(self.state_norm, "weight") and self.state_norm.weight is not None
            else hidden_states.dtype
        )
        y = self.state_norm(y.to(sn_dtype)).to(hidden_states.dtype)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            y = y * hm

        # ---- output ----
        y = y.transpose(1, 2).contiguous().view(B, L, H * d)

        if self.is_nested_encoder:
            out = y
        else:
            out = self.out_proj(y)

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


# ===========================================================================
# Injector (HGRN2-FIX-03, HGRN2-FIX-05, HGRN2-FIX-06)
# ===========================================================================
@ATTENTION_REGISTRY.register("hgrn2_outer_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_outer")
def inject_hgrn2_outer(
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
        logger.warning("inject_hgrn2_outer: max recursion depth exceeded.")
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
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_hgrn2_outer(
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

    # ---- Single-layer ----
    src = old_attn
    is_nested_target = False
    if src is None and target is not None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    if hasattr(cand, "self") and hasattr(cand.self, "query"):
                        src = cand.self
                        is_nested_target = True
                    else:
                        src = cand
                    break

    if src is None:
        return None

    # HGRN2-FIX-03: idempotency guard
    if isinstance(src, HGRN2OuterBlock) or "HGRN2" in src.__class__.__name__:
        return None

    # Resolve dims
    dim = 768
    num_heads = 12
    if cfg is not None:
        dim = getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 12))

    # HGRN2-FIX-06: detect decoder
    _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
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
    is_dec = (
        bool(getattr(cfg, "is_decoder", False) if cfg else False)
        or bool(getattr(cfg, "causal", False) if cfg else False)
        or any(_mt.startswith(p) for p in _dec_pfx)
        or hasattr(src, "c_attn")
    )

    try:
        kwargs.pop("dim", None)
        kwargs.pop("num_heads", None)
        new_block = HGRN2OuterBlock(
            config=cfg,
            dim=dim,
            num_heads=num_heads,
            is_decoder=is_dec,
            **kwargs,
        )
        new_block.copy_weights(src)

        p = next(src.parameters(), None)
        if p is not None:
            new_block.to(device=p.device, dtype=p.dtype)

        # Splice
        spliced = False
        if is_nested_target and hasattr(target, "attention") and hasattr(target.attention, "self"):
            new_block.is_nested_encoder = True
            target.attention.self = new_block
            spliced = True
        elif hasattr(target, "attention"):
            new_block.is_nested_encoder = False
            target.attention = new_block
            spliced = True
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
            spliced = True
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
            spliced = True
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
            spliced = True

        if not spliced:
            logger.error("inject_hgrn2_outer: no writable attribute on %s", type(target).__name__)
            return None

        # HGRN2-FIX-05: return the injected module
        return new_block

    except Exception as e:
        logger.error("inject_hgrn2_outer failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "HGRN2OuterBlock",
    "LoRALinear",
    "inject_hgrn2_outer",
    "apply_rotary_pos_emb",
]
