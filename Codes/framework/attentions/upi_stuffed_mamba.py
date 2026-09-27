"""
UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM Block.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
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
            nn.init.zeros_(self.lora_B)
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
        target_dtype = x.dtype
        w = self.weight.to(target_dtype) if self.weight.dtype != target_dtype else self.weight
        b = self.bias.to(target_dtype) if (self.bias is not None and self.bias.dtype != target_dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(target_dtype) if self.lora_A.dtype != target_dtype else self.lora_A
            bb = self.lora_B.to(target_dtype) if self.lora_B.dtype != target_dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.scaling
        return out.to(target_dtype)


@ATTENTION_REGISTRY.register("selective_mamba", supports_causal=True, supports_bidirectional=True)
@ATTENTION_REGISTRY.register("upi_mamba", supports_causal=True, supports_bidirectional=True)  # DEPRECATED alias
@ATTENTION_REGISTRY.register("upi_stuffed_mamba", supports_causal=True, supports_bidirectional=True)  # DEPRECATED alias
class SelectiveMamba(BaseAttention):
    r"""
    Selective Mamba SSM block.

    Implements a Mamba-style selective state space model with 
    input-dependent B, C, and dt parameters:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t + D · x_t
    where Ā_t = exp(dt_t · A).

    The "selectivity" comes from x_proj producing B_t and C_t 
    from the input.

    NOTE: The previous name "UPI-Stuffed" has been removed 
    because none of the claimed features (Uncertainty Penalty, 
    Interpolation, Stuffing) were actually implemented.

    Registered as "upi_mamba" (legacy alias) and 
    "selective_mamba" (canonical).
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        dim: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        if dim is None and d_model is None:
            if config is not None and not isinstance(config, int):
                dim = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", kwargs.get("hidden_size", 768)))
            else:
                dim = kwargs.get("hidden_size", 768)
        else:
            dim = dim or d_model

        if hasattr(dim, "hidden_size"):
            dim = dim.hidden_size

        self.dim: int = int(dim)
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand
        self.inner_dim: int = self.dim * expand
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None and not isinstance(config, int):
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.in_proj = LoRALinear(
            self.dim, self.inner_dim * 2, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )
        # A_log init: log([1, 2, ..., d_state]) repeated.
        # In our scan, we compute A = +exp(A_log) (positive),
        # then log_decay = -dt * A (negative), giving decay < 1.
        # This is mathematically equivalent to Mamba's 
        # A = -exp(A_log), decay = exp(dt * A).
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # Trainable skip connection D matching inner_dim
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Mamba paper: dt_rank = ceil(d_model / 16), not d_inner
        self.dt_rank = math.ceil(self.dim / 16)

        # x_proj maps inner_dim → dt_rank + 2*d_state
        self.x_proj = nn.Linear(
            self.inner_dim, 
            self.dt_rank + 2 * self.d_state, 
            bias=False,
        )
        self.x_proj.weight.requires_grad = True

        # dt_proj maps dt_rank → inner_dim
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # Initialize dt_proj bias (Mamba paper Eq. 12)
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.inner_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init_inv = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init_inv)
            # Weight init: uniform(-1/sqrt(dt_rank), 1/sqrt(dt_rank))
            bound = 1.0 / math.sqrt(self.dt_rank)
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
        self.dt_proj.weight.requires_grad = True
        self.dt_proj.bias.requires_grad = True

        self.out_proj = LoRALinear(
            self.inner_dim, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        if src_attn is not None:
            self.copy_weights(src_attn)

    @property
    def query(self) -> LoRALinear:
        return self.in_proj

    @property
    def key(self) -> LoRALinear:
        return self.in_proj

    @property
    def value(self) -> LoRALinear:
        return self.in_proj

    @property
    def q_lin(self) -> LoRALinear:
        return self.in_proj

    @property
    def k_lin(self) -> LoRALinear:
        return self.in_proj

    @property
    def v_lin(self) -> LoRALinear:
        return self.in_proj

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight.detach()
                    b_o = old_attn.output.dense.bias.detach() if old_attn.output.dense.bias is not None else None
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.dim() == 2 and (c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim):
                    c_attn_w = c_attn_w.t()

                split_q = min(self.dim, c_attn_w.shape[0] // 3)
                w_q = c_attn_w[: split_q, :]
                w_k = c_attn_w[split_q : 2 * split_q, :]
                w_v = c_attn_w[2 * split_q : 3 * split_q, :]

                out_w = src.c_proj.weight.detach()
                if out_w.dim() == 2 and (out_w.shape[0] != self.dim and out_w.shape[1] == self.dim):
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
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

            # Pattern 3: Qwen / LLaMA
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

            # Pattern 4: DistilBERT
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

            # Fallback: UniversalAdapter extraction
            if w_q is None or w_v is None or (w_o is None and not self.is_nested_encoder):
                for proj in ["q", "k", "v", "o"]:
                    try:
                        w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                        if w is not None:
                            if proj == "q" and w_q is None: w_q, b_q = w, b
                            elif proj == "k" and w_k is None: w_k, b_k = w, b
                            elif proj == "v" and w_v is None: w_v, b_v = w, b
                            elif proj == "o" and w_o is None: w_o, b_o = w, b
                    except Exception as e:
                        logger.debug("UniversalAdapter extraction for %s failed: %s", proj, e)

            missing = []
            if w_q is None: missing.append("Q")
            if w_v is None: missing.append("V")
            if w_o is None and not self.is_nested_encoder: missing.append("O")
            if missing:
                raise RuntimeError(
                    f"SelectiveMamba: cannot extract {missing} from "
                    f"{old_attn.__class__.__name__}. Aborting to avoid "
                    f"silent model corruption."
                )

            def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
                if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                    return w.t()
                return w

            def _adapt_to_inner(w: torch.Tensor, name: str) -> torch.Tensor:
                """
                Adapt weight from source head count to inner_dim.
                If source is smaller: tile and truncate.
                If source is larger: slice to fit.
                """
                src_out, src_in = w.shape
                if src_out == self.inner_dim:
                    return w

                if src_out > self.inner_dim:
                    # Source has more heads — take the first inner_dim
                    logger.debug(
                        "SelectiveMamba: slicing %s from %d to %d",
                        name, src_out, self.inner_dim,
                    )
                    return w[:self.inner_dim, :].contiguous()

                # Source is smaller — repeat to fill
                rep = math.ceil(self.inner_dim / src_out)
                logger.debug(
                    "SelectiveMamba: repeating %s from %d to %d (rep=%d)",
                    name, src_out, self.inner_dim, rep,
                )
                return w.repeat(rep, 1)[:self.inner_dim, :].contiguous()

            w_v_or = _orient(w_v, self.dim)
            w_q_or = _orient(w_q, self.dim)
            w_v_or = _adapt_to_inner(w_v_or, "W_V")
            w_q_or = _adapt_to_inner(w_q_or, "W_Q")

            v_chunk = w_v_or[:, :self.dim]
            q_chunk = w_q_or[:, :self.dim]

            self.in_proj.weight.data[:self.inner_dim, :self.dim].copy_(v_chunk)
            self.in_proj.weight.data[self.inner_dim:2*self.inner_dim, :self.dim].copy_(q_chunk)

            if w_o is not None:
                w_o_or = _orient(w_o, self.inner_dim)
                min_o_out = min(self.dim, w_o_or.shape[0])
                min_o_in = min(self.dim, w_o_or.shape[1])
                self.out_proj.weight.data[:min_o_out, :min_o_in].copy_(w_o_or[:min_o_out, :min_o_in])
                if b_o is not None and self.out_proj.bias is not None:
                    b_val = b_o.data if isinstance(b_o, nn.Parameter) else b_o
                    self.out_proj.bias.data[:min(len(self.out_proj.bias), len(b_val))].copy_(b_val[:min(len(self.out_proj.bias), len(b_val))])

            # Frobenius norm checks
            for name, w in [("in_proj", self.in_proj.weight), ("out_proj", self.out_proj.weight)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(f"SelectiveMamba: {name} has near-zero Frobenius norm (< 1e-4)")

            self.in_proj.freeze_base_weights()
            self.out_proj.freeze_base_weights()
            self.x_proj.weight.requires_grad = True
            self.dt_proj.weight.requires_grad = True
            self.dt_proj.bias.requires_grad = True

            # Preserve source LoRA adapters
            source_submodules = {
                "in": ["q_proj", "q_lin", "query", "W_q"],
                "out": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
            }
            for name in source_submodules["in"]:
                if hasattr(old_attn, name):
                    sub = getattr(old_attn, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and self.in_proj.lora_A is not None:
                            r_min = min(self.in_proj.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(self.in_proj.lora_A.shape[1], src_lA.shape[1])
                            self.in_proj.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            self.in_proj.lora_A.requires_grad = True
                        if hasattr(src_lB, "data") and self.in_proj.lora_B is not None:
                            out_min = min(self.in_proj.lora_B.shape[0], src_lB.shape[0])
                            r_min = min(self.in_proj.lora_B.shape[1], src_lB.shape[1])
                            self.in_proj.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                            self.in_proj.lora_B.requires_grad = True
                        if hasattr(sub, "lora_scaling"):
                            self.in_proj.scaling = sub.lora_scaling
                        break

            for name in source_submodules["out"]:
                if hasattr(old_attn, name):
                    sub = getattr(old_attn, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and self.out_proj.lora_A is not None:
                            r_min = min(self.out_proj.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(self.out_proj.lora_A.shape[1], src_lA.shape[1])
                            self.out_proj.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            self.out_proj.lora_A.requires_grad = True
                        if hasattr(src_lB, "data") and self.out_proj.lora_B is not None:
                            out_min = min(self.out_proj.lora_B.shape[0], src_lB.shape[0])
                            r_min = min(self.out_proj.lora_B.shape[1], src_lB.shape[1])
                            self.out_proj.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                            self.out_proj.lora_B.requires_grad = True
                        if hasattr(sub, "lora_scaling"):
                            self.out_proj.scaling = sub.lora_scaling
                        break

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B_t: torch.Tensor,
        C_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Bounded chunked associative SSM scan in FP32 with O(chunk=64) memory."""
        if reverse:
            x = torch.flip(x, dims=[1])
            dt = torch.flip(dt, dims=[1])
            B_t = torch.flip(B_t, dims=[1])
            C_t = torch.flip(C_t, dims=[1])
            if mask is not None:
                mask = torch.flip(mask, dims=[1])

        batch, seqlen, _ = x.shape
        A = torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0))  # [inner_dim, d_state]

        # In case of padding mask, zero out invalid token inputs
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            x = x * m
            dt = dt * m

        chunk_size = 64
        num_chunks = (seqlen + chunk_size - 1) // chunk_size
        h_running = x.new_zeros(batch, self.inner_dim, self.d_state, dtype=torch.float32)
        y_chunks = []
        A_view = A.view(1, 1, self.inner_dim, self.d_state)

        for c in range(num_chunks):
            s_idx = c * chunk_size
            e_idx = min(seqlen, s_idx + chunk_size)

            dt_c = dt[:, s_idx:e_idx]
            x_c = x[:, s_idx:e_idx]
            B_c = B_t[:, s_idx:e_idx]
            C_c = C_t[:, s_idx:e_idx]

            # Compute u and log_a strictly for current 64-token chunk to avoid buffer overflow
            log_a_c = -(dt_c.unsqueeze(-1) * A_view).float()
            u_c = (dt_c.unsqueeze(-1) * B_c.unsqueeze(2) * x_c.unsqueeze(-1)).float()

            lambda_c = torch.clamp(torch.cumsum(log_a_c, dim=1), min=-25.0, max=0.0)

            decay_c = torch.exp(lambda_c)
            decay_inv_c = torch.exp(-lambda_c).clamp(max=1e10)

            h_intra = decay_c * torch.cumsum(decay_inv_c * u_c, dim=1)
            h_inter = decay_c * h_running.unsqueeze(1)

            h_chunk = torch.clamp(h_intra + h_inter, min=-1e4, max=1e4)
            y_c = (C_c.unsqueeze(2).float() * h_chunk).sum(dim=-1).to(x.dtype)
            y_chunks.append(y_c)
            h_running = h_chunk[:, -1]

        y = torch.cat(y_chunks, dim=1)
        y = (y + self.D.to(x.dtype) * x).to(x.dtype)

        if reverse:
            y = torch.flip(y, dims=[1])

        return y

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "SelectiveMamba does not support cross-attention. "
                "This module replaces SELF-attention only. "
                "For cross-attention, use a different module."
            )

        B, L, D = hidden_states.shape
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)

        if head_mask is not None:
            logger.warning(
                "SelectiveMamba does not use head_mask (SSM has no heads). "
                "Ignoring head_mask."
            )

        mask_bool = None
        if mask is not None:
            if mask.dim() == 2:
                # Distinguish between int/bool masks and float additive masks
                if mask.dtype in (torch.bool, torch.int8, torch.int16, 
                                  torch.int32, torch.int64):
                    mask_bool = mask.bool().to(hidden_states.device)
                else:
                    # Float mask: -inf = invalid, 0 = valid
                    mask_bool = (mask > -1e4).to(hidden_states.device)
            elif mask.dim() == 4:
                mask_bool = (mask[:, 0, 0, :] > -1e4).to(hidden_states.device)
            else:
                try:
                    mask_bool = get_token_valid_mask(
                        mask, B, L, hidden_states.device,
                    )
                except Exception as e:
                    logger.debug("get_token_valid_mask failed: %s", e)
                    mask_bool = None

        target_dtype = hidden_states.dtype
        xz = self.in_proj(hidden_states)
        x_in, z = xz.chunk(2, dim=-1)

        conv_w = self.conv1d.weight.to(x_in.dtype)
        conv_b = self.conv1d.bias.to(x_in.dtype) if self.conv1d.bias is not None else None
        x_conv = F.conv1d(x_in.transpose(1, 2), conv_w, conv_b, stride=self.conv1d.stride, padding=self.conv1d.padding, dilation=self.conv1d.dilation, groups=self.conv1d.groups)[:, :, :L].transpose(1, 2).contiguous()
        x_ssm = F.silu(x_conv)

        # Selective projections: dt_raw [dt_rank], B_t [d_state], C_t [d_state]
        x_proj_out = self.x_proj(x_ssm)
        dt_raw, B_t, C_t = torch.split(
            x_proj_out, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw))
        dt = torch.clamp(dt, min=1e-3, max=1.0)

        mt = str(getattr(getattr(self, "config", None), "model_type", "") or "").lower()
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_decoder = (
            self.is_decoder
            or getattr(getattr(self, "config", None), "is_decoder", False)
            or any(mt.startswith(p) for p in decoder_prefixes)
        )

        y_fwd = self._selective_scan(x_ssm, dt, B_t, C_t, mask=mask_bool, reverse=False)

        if is_decoder:
            y = y_fwd
        else:
            y_bwd = self._selective_scan(x_ssm, dt, B_t, C_t, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)

        gated = (y.to(target_dtype) * F.silu(z.to(target_dtype))).to(target_dtype)

        # Inner SSM output is inner_dim; out_proj brings it back to dim.
        # Always apply out_proj.
        out = self.out_proj(gated).to(target_dtype)

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return (out, None)

    def step_recurrent(
        self,
        x_t: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single-step streaming inference. 
        
        x_t: (B, 1, D)
        state: dict with keys 'h', 'conv_buf' (optional)
        Returns: (y_t, new_state)
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, L, D = x_t.shape
        assert L == 1, "step_recurrent expects L=1"
        
        # Initialize state
        if state is None:
            h = torch.zeros(
                B, self.inner_dim, self.d_state,
                device=x_t.device, dtype=torch.float32,
            )
            conv_buf = torch.zeros(
                B, self.d_conv - 1, self.inner_dim,
                device=x_t.device, dtype=x_t.dtype,
            )
        else:
            h = state["h"]
            conv_buf = state.get("conv_buf")
        
        # in_proj
        xz = self.in_proj(x_t)  # (B, 1, 2*inner_dim)
        x_in, z = xz.chunk(2, dim=-1)
        
        # Conv1d with buffer
        x_conv_input = torch.cat([conv_buf, x_in], dim=1)  # (B, d_conv, inner_dim)
        x_conv = F.conv1d(
            x_conv_input.transpose(1, 2),
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.inner_dim,
        ).transpose(1, 2)  # (B, 1, inner_dim)
        new_conv_buf = x_conv_input[:, -(self.d_conv - 1):]
        
        x_ssm = F.silu(x_conv)
        
        # Selective params
        x_proj_out = self.x_proj(x_ssm)
        dt_raw, B_t, C_t = torch.split(
            x_proj_out, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw)).clamp(1e-3, 1.0)
        
        # Single step scan
        A = torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0))
        dA = torch.exp(-dt.unsqueeze(-1) * A.view(1, 1, self.inner_dim, self.d_state))
        dBx = dt.unsqueeze(-1) * B_t.unsqueeze(2) * x_ssm.unsqueeze(-1)
        
        h = dA.squeeze(1) * h + dBx.squeeze(1)  # (B, inner_dim, d_state)
        y = (C_t.squeeze(1).unsqueeze(1) * h).sum(dim=-1)  # (B, inner_dim)
        y = y + self.D * x_ssm.squeeze(1)
        
        # Gating
        gated = y * F.silu(z.squeeze(1))  # (B, inner_dim)
        
        # Output projection
        out = self.out_proj(gated.unsqueeze(1))  # (B, 1, D)
        
        new_state = {"h": h, "conv_buf": new_conv_buf}
        return out, new_state


UPIMambaBlock = SelectiveMamba  # Backward-compatibility alias


@ATTENTION_REGISTRY.register("selective_mamba_inject")
@ATTENTION_REGISTRY.register("inject_selective_mamba")
@ATTENTION_REGISTRY.register("upi_mamba_inject")  # DEPRECATED alias
@ATTENTION_REGISTRY.register("inject_upi_mamba")  # DEPRECATED alias
@ATTENTION_REGISTRY.register("upi_stuffed_mamba_inject")  # DEPRECATED alias
@ATTENTION_REGISTRY.register("inject_upi_stuffed_mamba")  # DEPRECATED alias
def inject_selective_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[SelectiveMamba]:
    """Safe injection hook for SelectiveMamba."""
    target = layer if layer is not None else model
    try:
        # Whole-model traversal support
        if target is not None and (
            hasattr(target, "transformer")
            or hasattr(target, "distilbert")
            or hasattr(target, "roberta")
            or hasattr(target, "bert")
            or (hasattr(target, "model") and hasattr(target.model, "layers"))
            or (hasattr(target, "encoder") and hasattr(target.encoder, "layer"))
        ):
            layer_list = None
            if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
                layer_list = target.distilbert.transformer.layer
            elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
                layer_list = target.roberta.encoder.layer
            elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
                layer_list = target.bert.encoder.layer
            elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
                layer_list = target.transformer.h
            elif hasattr(target, "model") and hasattr(target.model, "layers"):
                layer_list = target.model.layers
            elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
                layer_list = target.encoder.layer

            if layer_list is not None:
                tot = len(layer_list)
                for idx, lyr in enumerate(layer_list):
                    inject_selective_mamba(
                        layer=lyr,
                        model_config=model_config,
                        comp_config=comp_config,
                        layer_idx=idx,
                        total_layers=tot,
                        **kwargs,
                    )
                return target

        src = old_attn
        if src is None and target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    src = target.attention.self
                else:
                    src = target.attention
            elif hasattr(target, "self_attn"):
                src = target.self_attn
            elif hasattr(target, "attn"):
                src = target.attn
            elif hasattr(target, "self"):
                src = target.self

        if src is None or isinstance(src, SelectiveMamba):
            return None

        cfg = model_config or getattr(target, "config", getattr(src, "config", None))
        dim = (
            _cfg_get(cfg, "hidden_size")
            or _cfg_get(cfg, "d_model")
            or _cfg_get(cfg, "dim")
            or getattr(src, "hidden_size", None)
            or getattr(src, "embed_dim", None)
            or (src.q_proj.in_features if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features") else None)
            or (src.q_lin.in_features if hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features") else None)
            or (src.query.in_features if hasattr(src, "query") and hasattr(src.query, "in_features") else None)
            or 768
        )

        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_block = SelectiveMamba(
            config=cfg,
            dim=dim,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )
        if src is not None:
            p = next(src.parameters(), None)
            if p is not None:
                new_block.to(device=p.device, dtype=p.dtype)

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_block
                    new_block.is_nested_encoder = True
                    return new_block
                else:
                    target.attention = new_block
                    return new_block
            elif hasattr(target, "self_attn"):
                target.self_attn = new_block
            elif hasattr(target, "attn"):
                target.attn = new_block
            elif hasattr(target, "self"):
                target.self = new_block
                new_block.is_nested_encoder = True
            else:
                target.attention = new_block

        return new_block
    except Exception as e:
        logger.error("inject_selective_mamba failed at layer %d: %s", layer_idx, e)
        return None


inject_upi_mamba = inject_selective_mamba  # Backward-compatibility alias

