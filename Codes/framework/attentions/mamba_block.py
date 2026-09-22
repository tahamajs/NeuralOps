"""
Faithful Mamba Block Implementation.

Based on Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752).

Key properties:
- Full state dimension (B, d_inner, N) with N=16
- Input-dependent Δ, B, C (true selectivity via x_proj)
- Depthwise conv1d for local mixing (kernel size 4)
- Structured A parameterization: A = -exp(A_log)
- Clean SiLU gating branch
- Proper chunked selective scan kernel
"""

from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.selective_scan import selective_scan


@ATTENTION_REGISTRY.register("mamba")
@ATTENTION_REGISTRY.register("mamba_attention")
@ATTENTION_REGISTRY.register("mamba_block")
class MambaBlock(nn.Module):
    """
    Faithful Mamba block with full state dimension, input-dependent parameters,
    depthwise 1D conv, and chunked selective scan.
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
        bidirectional: bool = False,
        is_decoder_override: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, (int, float)):
            d_model = int(config)
            config = None

        super().__init__()
        self.config = config

        extracted_dim = (
            d_model
            or kwargs.get("hidden_size")
            or kwargs.get("dim")
            or (getattr(config, "hidden_size", None) if config else None)
            or (getattr(config, "d_model", None) if config else None)
            or 768
        )
        self.d_model: int = int(extracted_dim)
        self.d_state: int = int(d_state)
        self.d_conv: int = int(d_conv)
        self.expand: int = int(expand)
        self.d_inner: int = self.expand * self.d_model
        self.dt_rank: int = int(dt_rank or math.ceil(self.d_model / 16))
        self.bidirectional: bool = bool(bidirectional)

        # Decoder / Causal detection
        if is_decoder_override is not None:
            self.is_decoder = bool(is_decoder_override)
        else:
            model_type = str(getattr(config, "model_type", "")).lower() if config else ""
            _dec_prefixes = (
                "gpt", "qwen", "llama", "mistral", "gemma",
                "phi", "opt", "bloom", "falcon", "olmo"
            )
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False)) if config else False
            ) or any(model_type.startswith(p) for p in _dec_prefixes)

        # ---- Input projection ----
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

        # ---- Depthwise conv (local mixing) ----
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )

        # ---- Selective parameter projections ----
        # x_proj maps from d_inner to (dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        # dt_proj maps from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # ---- A parameter (Mamba paper Eq. 11) ----
        # S4D real initialization: A_n = -(n + 1)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))

        # ---- D (skip connection) ----
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- Output projection ----
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # ---- dt_proj bias initialization (Mamba paper) ----
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_init_inv = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init_inv)
            bound = 1.0 / math.sqrt(self.dt_rank)
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)

        # ---- Conv1d initialization ----
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))

        # ---- Bidirectional mode ----
        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A.clone() + 0.01 * torch.randn_like(A).abs()))
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(dt_init_inv)
                nn.init.uniform_(self.dt_proj_bwd.weight, -bound, bound)
            self.x_proj_bwd = nn.Linear(
                self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
            )
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if kwargs.get("encoder_hidden_states") is not None:
            raise NotImplementedError("Mamba does not support cross-attention")

        B, L, _ = hidden_states.shape

        # ---- Input projection ----
        xz = self.in_proj(hidden_states)               # (B, L, 2 * d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                 # both (B, L, d_inner)

        # ---- Depthwise conv (transpose for Conv1d) ----
        x_conv = self.conv1d(
            x_ssm.transpose(1, 2)                      # (B, d_inner, L + pad)
        )[..., :L].transpose(1, 2)                     # (B, L, d_inner)

        # ---- SiLU activation ----
        x_ssm = F.silu(x_conv)

        # ---- Selective parameters (input-dependent) ----
        x_dbl = self.x_proj(x_ssm)                     # (B, L, dt_rank + 2N)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)                          # (B, L, d_inner)

        # ---- Compute A ----
        A = -torch.exp(self.A_log)                     # (d_inner, N)

        # ---- Selective scan ----
        u_t = x_ssm.transpose(1, 2)                    # (B, d_inner, L)
        dt_t = dt.transpose(1, 2)                      # (B, d_inner, L)
        B_t = B_ssm.transpose(1, 2)                    # (B, N, L)
        C_t = C_ssm.transpose(1, 2)                    # (B, N, L)

        y = selective_scan(
            u=u_t,
            delta=dt_t,
            A=A,
            B=B_t,
            C=C_t,
            D=self.D,
            delta_softplus=True,
        )                                              # (B, d_inner, L)
        y = y.transpose(1, 2)                          # (B, L, d_inner)

        # ---- Bidirectional mode ----
        if self.bidirectional and not self.is_decoder:
            u_rev = torch.flip(x_ssm, dims=[1]).transpose(1, 2)
            x_rev = torch.flip(x_ssm, dims=[1])
            x_dbl_rev = self.x_proj_bwd(x_rev)
            dt_b, B_b, C_b = x_dbl_rev.split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_b = self.dt_proj_bwd(dt_b).transpose(1, 2)
            B_b = B_b.transpose(1, 2)
            C_b = C_b.transpose(1, 2)
            A_b = -torch.exp(self.A_log_bwd)

            y_b = selective_scan(
                u=u_rev,
                delta=dt_b,
                A=A_b,
                B=B_b,
                C=C_b,
                D=self.D_bwd,
                delta_softplus=True,
            )
            y_b = y_b.transpose(1, 2)
            y_b = torch.flip(y_b, dims=[1])
            y = 0.5 * (y + y_b)

        # ---- Gating ----
        y = y * F.silu(z)

        # ---- Output projection ----
        out = self.out_proj(y)

        # ---- Mask handling ----
        if attention_mask is not None and attention_mask.dim() == 2:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)

        return out, None
