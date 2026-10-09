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


@ATTENTION_REGISTRY.register("simplessm")
@ATTENTION_REGISTRY.register("selective_ssm")
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

        # Depthwise 1D conv (padding=0 to enforce causality via explicit left-padding)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=0,
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

        # Decoder detection
        is_dec = _cfg_get(config, "is_decoder", kwargs.get("is_decoder", False))
        model_type = str(_cfg_get(config, "model_type", "") or "").lower()
        dec_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo")
        if any(model_type.startswith(p) for p in dec_prefixes) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)

        # Bidirectional components if requested (disabled for causal decoders)
        if self.bidirectional and not self.is_decoder:
            self.conv1d_bwd = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=self.d_conv,
                groups=self.d_inner,
                padding=0,
                bias=conv_bias,
            )
            self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            A_bwd = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log_bwd = nn.Parameter(torch.log(A_bwd))
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj_bwd = nn.Linear(self.d_inner, self.d_model, bias=bias)
        else:
            self.bidirectional = False
        
        self.triton_kwargs = {}
        self._reset_parameters()

        source_attn = kwargs.get("source_attn") or kwargs.get("old_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

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

    def _read_state(
        self, past_state: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Returns (h, conv_state)."""
        if past_state is None:
            return None, None
        if isinstance(past_state, (tuple, list)):
            if len(past_state) >= 2:
                return past_state[0], past_state[1]
            elif len(past_state) == 1:
                return past_state[0], None
        return None, None

    def _step_ssm(
        self,
        u_t: torch.Tensor,       # (B, d_inner)
        dt_t: torch.Tensor,      # (B, d_inner)
        B_t: torch.Tensor,       # (B, d_state)
        C_t: torch.Tensor,       # (B, d_state)
        A_param: torch.Tensor,   # (d_inner, d_state)
        D_param: torch.Tensor,   # (d_inner,)
        h_prev: torch.Tensor | None,  # (B, d_inner, d_state)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step recurrent SSM update for autoregressive streaming inference."""
        B, d_inner = u_t.shape
        d_state = A_param.shape[1]
        dtype = u_t.dtype
        if h_prev is None:
            h_prev = torch.zeros(B, d_inner, d_state, device=u_t.device, dtype=dtype)
        else:
            h_prev = h_prev.to(dtype)

        # Discretize: delta_a = exp(dt * A)
        delta_a = torch.exp(dt_t.unsqueeze(-1) * A_param.unsqueeze(0))  # (B, d_inner, d_state)
        delta_b_u = dt_t.unsqueeze(-1) * (u_t.unsqueeze(-1) * B_t.unsqueeze(1))  # (B, d_inner, d_state)
        h_next = delta_a * h_prev + delta_b_u

        # Output: y = h_next @ C_t + D * u_t
        y = torch.matmul(h_next, C_t.unsqueeze(-1)).squeeze(-1) + D_param.unsqueeze(0) * u_t
        return y, h_next

    def _forward_core(
        self,
        x_ssm: torch.Tensor,
        z: torch.Tensor,
        conv1d_layer: nn.Module,
        x_proj_layer: nn.Module,
        dt_proj_layer: nn.Module,
        A_log_param: nn.Parameter,
        D_param: nn.Parameter,
        past_h: torch.Tensor | None = None,
        conv_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x_ssm.shape
        target_dtype = x_ssm.dtype
        device = x_ssm.device

        # 1. Depthwise 1D conv with explicit left-padding / streaming buffer
        x_ssm_t = x_ssm.transpose(1, 2).contiguous()  # (B, d_inner, L)
        if conv_prev is not None:
            conv_buf = conv_prev.to(target_dtype)  # (B, d_inner, d_conv - 1)
            x_cat = torch.cat([conv_buf, x_ssm_t], dim=2)
        else:
            # Explicit left-padding by (d_conv - 1) to enforce strict causality
            x_cat = F.pad(x_ssm_t, (self.d_conv - 1, 0))

        x_conv_full = conv1d_layer(x_cat)
        x_conv = x_conv_full[..., -L:].transpose(1, 2).contiguous()
        x_conv = F.silu(x_conv)

        # Update conv buffer: last (d_conv - 1) tokens
        if self.d_conv - 1 <= x_cat.shape[2]:
            conv_next = x_cat[..., -(self.d_conv - 1) :].contiguous()
        else:
            pad = self.d_conv - 1 - x_cat.shape[2]
            conv_next = torch.cat(
                [torch.zeros(B, self.d_inner, pad, device=device, dtype=target_dtype), x_cat],
                dim=2,
            ).contiguous()

        # 2. Selective projections
        x_dbl = x_proj_layer(x_conv)  # (B, L, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj_layer(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)

        A = -torch.exp(A_log_param.float())  # (d_inner, d_state)

        # 3. Scan or recurrent step
        if past_h is not None and L == 1:
            u_step = x_conv.squeeze(1)
            dt_step = dt.squeeze(1)
            B_step = B_ssm.squeeze(1)
            C_step = C_ssm.squeeze(1)
            y_step, h_next = self._step_ssm(
                u_t=u_step,
                dt_t=dt_step,
                B_t=B_step,
                C_t=C_step,
                A_param=A.to(target_dtype),
                D_param=D_param.to(target_dtype),
                h_prev=past_h,
            )
            y = y_step.unsqueeze(1)
        else:
            y = selective_scan(
                u=x_conv.transpose(1, 2).contiguous(),
                delta=dt.transpose(1, 2).contiguous(),
                A=A,
                B=B_ssm.transpose(1, 2).contiguous(),
                C=C_ssm.transpose(1, 2).contiguous(),
                D=D_param,
            )  # (B, d_inner, L)
            y = y.transpose(1, 2).contiguous()  # (B, L, d_inner)

            # Approximate final recurrent state for caching if requested
            delta_a_last = torch.exp(dt[:, -1:, :].transpose(1, 2) * A.unsqueeze(0).unsqueeze(-1))
            delta_b_u_last = dt[:, -1:, :].transpose(1, 2) * (
                x_conv[:, -1:, :].transpose(1, 2) * B_ssm[:, -1:, :].transpose(1, 2).permute(0, 2, 1)
            )
            if past_h is not None:
                h_next = delta_a_last.squeeze(-1) * past_h.to(target_dtype) + delta_b_u_last.squeeze(-1)
            else:
                h_next = delta_b_u_last.squeeze(-1)

        # 4. Gating
        y = (y.to(target_dtype) * F.silu(z.to(target_dtype)))
        present_state = (h_next, conv_next)
        return y, present_state

    def copy_weights(self, old_attn: Any) -> None:
        """
        Map attention weights into SSM parameters using UniversalAdapter:
        - W_v -> in_proj (first half: x_ssm path)
        - W_q -> in_proj (second half: z gating path)
        - W_o -> out_proj
        - Zero-initialize or identity-bias the depthwise conv weights.
        """
        if old_attn is None:
            return

        from neuralops.core.adapters import UniversalAdapter
        adapter = UniversalAdapter()

        w_v, b_v = adapter.get_weight_bias(old_attn, "v")
        w_q, b_q = adapter.get_weight_bias(old_attn, "q")
        w_o, b_o = adapter.get_weight_bias(old_attn, "o")

        def _to_shape(w: torch.Tensor | None, r: int, c: int) -> torch.Tensor | None:
            if w is None:
                return None
            w = w.float()
            if w.ndim != 2:
                return None
            min_r = min(r, w.shape[0])
            min_c = min(c, w.shape[1])
            out = torch.zeros(r, c, dtype=torch.float32, device=w.device)
            out[:min_r, :min_c].copy_(w[:min_r, :min_c])
            return out

        with torch.no_grad():
            dev = self.in_proj.weight.device
            dt = self.in_proj.weight.dtype

            # 1. in_proj: first half from W_v, second half from W_q
            if w_v is not None:
                mapped_v = _to_shape(w_v, self.d_inner, self.d_model)
                if mapped_v is not None:
                    self.in_proj.weight.data[: self.d_inner, :].copy_(mapped_v.to(dev, dt))
            if w_q is not None:
                mapped_q = _to_shape(w_q, self.d_inner, self.d_model)
                if mapped_q is not None:
                    self.in_proj.weight.data[self.d_inner :, :].copy_(mapped_q.to(dev, dt))

            if self.in_proj.bias is not None:
                self.in_proj.bias.data.zero_()
                if b_v is not None:
                    min_b = min(self.d_inner, b_v.numel())
                    self.in_proj.bias.data[:min_b].copy_(b_v[:min_b].to(dev, dt))
                if b_q is not None:
                    min_b = min(self.d_inner, b_q.numel())
                    self.in_proj.bias.data[self.d_inner : self.d_inner + min_b].copy_(b_q[:min_b].to(dev, dt))

            # 2. out_proj from W_o
            if w_o is not None:
                mapped_o = _to_shape(w_o, self.d_model, self.d_inner)
                if mapped_o is not None:
                    self.out_proj.weight.data.copy_(mapped_o.to(self.out_proj.weight.device, self.out_proj.weight.dtype))
            if self.out_proj.bias is not None:
                if b_o is not None:
                    min_b = min(self.d_model, b_o.numel())
                    self.out_proj.bias.data[:min_b].copy_(b_o[:min_b].to(self.out_proj.bias.device, self.out_proj.bias.dtype))
                else:
                    self.out_proj.bias.data.zero_()

            # 3. conv1d identity-at-init passthrough
            self.conv1d.weight.data.zero_()
            if self.conv1d.weight.dim() == 3:
                # Shape (d_inner, 1, d_conv); set the last tap to 1.0
                self.conv1d.weight.data[:, 0, -1] = 1.0
            if self.conv1d.bias is not None:
                self.conv1d.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_state: Any | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("SelectiveSSM does not support cross-attention.")

        # Accept past_key_value as alias for past_state
        if past_state is None and past_key_value is not None:
            past_state = past_key_value

        target_dtype = hidden_states.dtype
        B, L, D = hidden_states.shape

        h_prev, conv_prev = self._read_state(past_state)

        # Project directly into SSM and gating paths without duplicating LayerNorm
        x_proj = self.in_proj(hidden_states).to(target_dtype)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        # Prevent State Poisoning from Padded Tokens
        if attention_mask is not None:
            valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
            if valid_mask is not None:
                mask_float = valid_mask.unsqueeze(-1).to(target_dtype)
                x_ssm = x_ssm * mask_float
                z = z * mask_float

        y_fwd, present_state = self._forward_core(
            x_ssm=x_ssm,
            z=z,
            conv1d_layer=self.conv1d,
            x_proj_layer=self.x_proj,
            dt_proj_layer=self.dt_proj,
            A_log_param=self.A_log,
            D_param=self.D,
            past_h=h_prev,
            conv_prev=conv_prev,
        )
        out = self.out_proj(y_fwd.to(target_dtype))

        if self.bidirectional:
            x_ssm_bwd = torch.flip(x_ssm, [1]).contiguous()
            z_bwd = torch.flip(z, [1]).contiguous()
            
            y_bwd, _ = self._forward_core(
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

        if self.is_decoder:
            return (out, present_state)
        return (out, None)


SimpleSSM = SelectiveSSM
ATTENTION_REGISTRY.register("simple_ssm")(SimpleSSM)


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