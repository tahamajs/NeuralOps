"""
Selective State Space Model (SelectiveSSM).

Faithful implementation of Mamba-style selective state space recurrence block.
Features:
- State dimension (B, D, L, N)
- Selective (input-dependent) Delta, B, C parameters
- Structured A matrix initialization (Mamba paper Eq. 11)
- Depthwise 1D convolution with SiLU gating
- Bidirectional support with independent forward and backward parameters
- No internal LayerNorm duplication
- Memory-safe contiguous memory layouts and padding mask protections
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.selective_scan import selective_scan

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@ATTENTION_REGISTRY.register("selective_ssm")
@ATTENTION_REGISTRY.register("mamba_ssm")
class SelectiveSSM(BaseAttention):
    r"""
    Faithful Mamba-style selective SSM block with hardened execution paths.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "state_update"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_bias: bool = True,
        bias: bool = False,
        bidirectional: bool = False,
        lora_rank: int = 0,
        weights_init: str = "mamba",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if d_model is not None:
            self.d_model = int(d_model)
        else:
            self.d_model = int(_cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768)))

        self.d_state = int(_cfg_get(config, "d_state", kwargs.get("d_state", d_state)))
        self.d_conv = int(_cfg_get(config, "d_conv", kwargs.get("d_conv", d_conv)))
        self.expand = int(_cfg_get(config, "expand", kwargs.get("expand", expand)))
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else int(dt_rank)
        self.bidirectional = bool(
            _cfg_get(config, "bidirectional", kwargs.get("bidirectional", bidirectional))
        )
        self.weights_init = weights_init

        # Input projection: x -> (x_ssm, z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # Depthwise 1D conv
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )

        # Delta, B, C projections from x_ssm
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # Delta projection from dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameterization: A = -exp(A_log)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # Bidirectional components if requested
        if self.bidirectional:
            self.conv1d_bwd = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=self.d_conv,
                groups=self.d_inner,
                padding=self.d_conv - 1,
                bias=conv_bias,
            )
            self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            A_bwd = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log_bwd = nn.Parameter(torch.log(A_bwd))
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj_bwd = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
        self.triton_kwargs = {}
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.weights_init == "mamba":
            # Delta bias init: softplus^-1(Uniform(0.001, 0.1))
            dt_min, dt_max = 0.001, 0.1
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt_init_inv = dt_init + torch.log(-torch.expm1(-dt_init))
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt_init_inv)
                self.dt_proj.weight.copy_(
                    torch.ones(self.d_inner, self.dt_rank) * (0.5 / self.dt_rank) ** 0.5
                )
                if self.bidirectional:
                    self.dt_proj_bwd.bias.copy_(dt_init_inv)
                    self.dt_proj_bwd.weight.copy_(
                        torch.ones(self.d_inner, self.dt_rank) * (0.5 / self.dt_rank) ** 0.5
                    )
        elif self.weights_init == "xavier":
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.xavier_uniform_(self.x_proj.weight)

    def _forward_core(
        self,
        x_ssm: torch.Tensor,
        z: torch.Tensor,
        conv1d_layer: nn.Module,
        x_proj_layer: nn.Module,
        dt_proj_layer: nn.Module,
        A_log_param: nn.Parameter,
        D_param: nn.Parameter,
    ) -> torch.Tensor:
        B, L, _ = x_ssm.shape

        # FIX: Ensure transposed inputs to Conv1D are strictly contiguous
        x_ssm_t = x_ssm.transpose(1, 2).contiguous()
        x_conv = conv1d_layer(x_ssm_t)[..., :L].transpose(1, 2).contiguous()
        x_conv = F.silu(x_conv)

        # Selective projections
        x_dbl = x_proj_layer(x_conv)  # (B, L, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj_layer(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)

        A = -torch.exp(A_log_param.float())  # (d_inner, d_state)

        # FIX: Force strict contiguous memory layout for all inputs to Triton/CUDA kernel
        y = selective_scan(
            u=x_conv.transpose(1, 2).contiguous(),
            delta=dt.transpose(1, 2).contiguous(),
            A=A,
            B=B_ssm.transpose(1, 2).contiguous(),
            C=C_ssm.transpose(1, 2).contiguous(),
            D=D_param,
        )  # (B, d_inner, L)
        y = y.transpose(1, 2).contiguous()  # (B, L, d_inner)

        # Gating
        target_dtype = x_ssm.dtype
        y = (y.to(target_dtype) * F.silu(z.to(target_dtype)))
        return y

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
        if encoder_hidden_states is not None:
            raise NotImplementedError("SelectiveSSM does not support cross-attention.")

        target_dtype = hidden_states.dtype
        B, L, D = hidden_states.shape

        # Project directly into SSM and gating paths without duplicating LayerNorm
        x_proj = self.in_proj(hidden_states).to(target_dtype)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        # FIX: Prevent State Poisoning from Padded Tokens
        # Zeros out embeddings where padding exists so the state ignores them entirely
        if attention_mask is not None:
            valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
            if valid_mask is not None:
                mask_float = valid_mask.unsqueeze(-1).to(target_dtype)
                x_ssm = x_ssm * mask_float
                z = z * mask_float

        y_fwd = self._forward_core(
            x_ssm=x_ssm,
            z=z,
            conv1d_layer=self.conv1d,
            x_proj_layer=self.x_proj,
            dt_proj_layer=self.dt_proj,
            A_log_param=self.A_log,
            D_param=self.D,
        )
        out = self.out_proj(y_fwd.to(target_dtype))

        if self.bidirectional:
            # FIX: Memory layouts are immediately made contiguous after any flip operations
            x_ssm_bwd = torch.flip(x_ssm, [1]).contiguous()
            z_bwd = torch.flip(z, [1]).contiguous()
            
            y_bwd = self._forward_core(
                x_ssm=x_ssm_bwd,
                z=z_bwd,
                conv1d_layer=self.conv1d_bwd,
                x_proj_layer=self.x_proj_bwd,
                dt_proj_layer=self.dt_proj_bwd,
                A_log_param=self.A_log_bwd,
                D_param=self.D_bwd,
            )
            
            y_bwd = torch.flip(y_bwd, [1]).contiguous()
            out = out + self.out_proj_bwd(y_bwd.to(target_dtype))

        return (out, None)


SimpleSSM = SelectiveSSM
ATTENTION_REGISTRY.register("simple_ssm")(SimpleSSM)
ATTENTION_REGISTRY.register("simplessm")(SimpleSSM)


@ATTENTION_REGISTRY.register("selective_ssm_inject")
@ATTENTION_REGISTRY.register("inject_selective_ssm")
@ATTENTION_REGISTRY.register("inject_simple_ssm")
@ATTENTION_REGISTRY.register("inject_simple_ssm_attention")
def inject_selective_ssm(layer=None, model=None, model_config=None, old_attn=None, **kwargs):
    target = layer if layer is not None else model
    if target is None:
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        if hasattr(target, attr):
            new_block = SelectiveSSM(config=model_config, **kwargs)
            setattr(target, attr, new_block)
            return new_block
    return None


inject_simple_ssm = inject_selective_ssm
inject_simple_ssm_attention = inject_selective_ssm

__all__ = [
    "SelectiveSSM",
    "SimpleSSM",
    "inject_simple_ssm",
    "inject_simple_ssm_attention",
    "inject_selective_ssm",
]