"""
Mamba Block with SVD-Based Attention Weight Distillation

Selective state space sequence model enhanced with SVD-based weight distillation for compact deployment.

Author: Taha Majlesi
"""

from __future__ import annotations
from neuralops.core.common import _infer_d_model
from neuralops.core.common import _detect_return_format

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.core.selective_scan import HAS_MAMBA_CUDA, selective_scan
except Exception as e:
    logging.getLogger(__name__).warning("selective_scan import failed: %s", e)
    selective_scan = None
    HAS_MAMBA_CUDA = False

try:
    from neuralops.core.triton_kernels.selective_scan import (
        selective_scan_fn as triton_selective_scan,
    )

    HAS_TRITON_MAMBA = True
except Exception as _e:
    logging.getLogger(__name__).warning("Triton selective_scan unavailable: %s", _e)
    triton_selective_scan = None
    HAS_TRITON_MAMBA = False

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("mamba_block")
class MambaBlock(nn.Module):
    """Faithful S6 / Mamba block, production-hardened."""

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = False

    def __init__(
        self,
        config: Any | None = None,
        d_model: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_bias: bool = True,
        bias: bool = False,
        bidirectional: bool = False,
        is_decoder_override: bool | None = None,
        freeze_base: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, (int, float)) and d_model is None:
            d_model = int(config)
            config = None

        super().__init__()
        self.config = config

        if d_model is None:
            d_model = _infer_d_model(config, kwargs.get("source_attn"), None, **kwargs)
        self.d_model = int(d_model)

        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = int(dt_rank or math.ceil(self.d_model / 16))
        self.bidirectional = bool(bidirectional)
        self.freeze_base = bool(freeze_base)

        if is_decoder_override is not None:
            self.is_decoder = bool(is_decoder_override)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
            _dec_prefixes = (
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
            )
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False) or getattr(config, "causal", False))
                if config
                else False
            ) or any(_mt.startswith(p) for p in _dec_prefixes)

        if not self.is_decoder and "bidirectional" not in kwargs:
            self.bidirectional = True

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=0,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt_init_inv = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init_inv)
            bound = 1.0 / math.sqrt(self.dt_rank)
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
            nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
            if self.conv1d.bias is not None:
                nn.init.zeros_(self.conv1d.bias)

        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A.clone() + 0.01 * torch.randn_like(A).abs()))
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(dt_init_inv)
                nn.init.uniform_(self.dt_proj_bwd.weight, -bound, bound)
            self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        source_attn = kwargs.get("source_attn") or kwargs.get("old_attn")
        self.is_nested_encoder = False
        self._return_format = _detect_return_format(source_attn, False)

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        """MB-04: match the framework's LoRA-freeze convention."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_base_weights(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def copy_weights(self, src: nn.Module) -> None:
        """
        Distill attention weights (W_q, W_k, W_v, W_o) into Mamba's S6
        parameters via low-rank mapping:
            W_v  → in_proj (SSM input path)
            W_k  → x_proj (B branch, via SVD)
            W_q  → x_proj (C branch, via SVD)
            W_o  → out_proj
            dt   → kept at initialization (learnable, input-dependent via x_proj)
        """
        if src is None:
            return

        extracted: dict[str, torch.Tensor | None] = {
            "q": None,
            "k": None,
            "v": None,
            "o": None,
        }

        try:
            from neuralops.core.adapters import UniversalAdapter

            for proj in ("q", "k", "v", "o"):
                try:
                    w, _ = UniversalAdapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted[proj] = w
                except Exception:
                    pass
        except Exception:
            pass

        if hasattr(src, "c_attn") and src.c_attn is not None and extracted["q"] is None:
            c = src.c_attn
            base = getattr(c, "base_layer", c)
            w = getattr(base, "weight", getattr(c, "weight", None))
            if w is not None:
                w = w.data.clone()
                is_conv = (
                    c.__class__.__name__ == "Conv1D"
                    or base.__class__.__name__ == "Conv1D"
                    or (w.ndim == 2 and w.shape[1] == 3 * self.d_model)
                )
                if is_conv:
                    w = w.t().contiguous()
                split = w.size(0) // 3
                extracted["q"] = w[:split, :]
                extracted["k"] = w[split : 2 * split, :]
                extracted["v"] = w[2 * split :, :]

        if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"] is None:
            o = src.c_proj
            base = getattr(o, "base_layer", o)
            w = getattr(base, "weight", getattr(o, "weight", None))
            if w is not None:
                w = w.data.clone()
                if w.ndim == 2 and w.shape[0] < w.shape[1]:
                    w = w.t().contiguous()
                extracted["o"] = w

        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        nested = getattr(src, "self", None)
        for proj, names in named_map.items():
            if extracted[proj] is not None:
                continue
            if nested is not None:
                for n in names:
                    mod = getattr(nested, n, None)
                    if mod is not None and hasattr(mod, "weight"):
                        extracted[proj] = mod.weight.data
                        break
            if extracted[proj] is not None:
                continue
            for n in names:
                mod = getattr(src, n, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj] = mod.weight.data
                    break
            if extracted["o"] is None and proj == "o":
                out = getattr(src, "output", None)
                dense = getattr(out, "dense", None) if out is not None else None
                if dense is not None and hasattr(dense, "weight"):
                    extracted["o"] = dense.weight.data

        cls_name = src.__class__.__name__
        if cls_name in (
            "BertSelfAttention",
            "RobertaSelfAttention",
            "XLMRobertaSelfAttention",
        ) or (nested is not None and hasattr(nested, "query")):
            self.is_nested_encoder = True

        dtype = self.in_proj.weight.dtype
        device = self.in_proj.weight.device

        def _to_t(x: torch.Tensor | None, r: int, c: int) -> torch.Tensor | None:
            """Project W (any shape) to target shape (r, c) via SVD."""
            if x is None:
                return None
            x = x.float()
            if x.ndim != 2:
                return None
            M, N = x.shape
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            k = min(r, c, s.numel())

            u_r = u[: min(r, M), :k]
            vh_c = vh[:k, : min(c, N)]
            core = (u_r * s[:k].unsqueeze(0)) @ vh_c

            out = torch.zeros(r, c, dtype=torch.float32, device=x.device)
            out[: core.size(0), : core.size(1)] = core

            if r > core.size(0):
                rem_r = r - core.size(0)
                reps_r = (rem_r + core.size(0) - 1) // core.size(0)
                scale_r = 1.0 / math.sqrt(reps_r + 1)
                out[: core.size(0), :] *= scale_r
                tiled_r = (core * scale_r).repeat(reps_r, 1)[:rem_r, :]
                out[core.size(0) :, : core.size(1)] = tiled_r

            if c > core.size(1):
                rem_c = c - core.size(1)
                reps_c = (rem_c + core.size(1) - 1) // core.size(1)
                scale_c = 1.0 / math.sqrt(reps_c + 1)
                out[:, : core.size(1)] *= scale_c
                tiled_c = (out[:, : core.size(1)]).repeat(1, reps_c)[:, :rem_c]
                out[:, core.size(1) :] = tiled_c

            return out.to(device=device, dtype=dtype)

        with torch.no_grad():
            w_q = extracted["q"]
            w_k = extracted["k"]
            w_v = extracted["v"]
            w_o = extracted["o"]

            if w_v is not None:
                w_v_p = _to_t(w_v, self.d_inner, self.d_model)
                if w_v_p is not None:
                    self.in_proj.weight.data[: self.d_inner, : self.d_model].copy_(w_v_p)
                    self.in_proj.weight.data[self.d_inner :, : self.d_model].copy_(0.5 * w_v_p)

            with torch.no_grad():
                b_start = self.dt_rank
                b_end = self.dt_rank + self.d_state
                c_start = b_end
                c_end = c_start + self.d_state

                if w_k is not None:
                    b_mat = _to_t(w_k, self.d_state, self.d_inner)
                    if b_mat is not None:
                        self.x_proj.weight.data[b_start:b_end, :].copy_(b_mat)
                if w_q is not None:
                    c_mat = _to_t(w_q, self.d_state, self.d_inner)
                    if c_mat is not None:
                        self.x_proj.weight.data[c_start:c_end, :].copy_(c_mat)

            if w_o is not None:
                w_o_p = _to_t(w_o, self.d_model, self.d_inner)
                if w_o_p is not None:
                    self.out_proj.weight.data.copy_(w_o_p)
                    if self.out_proj.bias is not None:
                        self.out_proj.bias.data.zero_()

            self.conv1d.weight.data.zero_()
            if self.conv1d.weight.dim() == 3:
                self.conv1d.weight.data[:, 0, -1] = 1.0
            if self.conv1d.bias is not None:
                self.conv1d.bias.data.zero_()

        for name, mod in (
            ("in_proj", self.in_proj),
            ("x_proj", self.x_proj),
            ("out_proj", self.out_proj),
        ):
            n = float(torch.linalg.norm(mod.weight.data.float()).item())
            if n < 1e-6:
                logger.warning("MambaBlock: '%s' near-zero norm (%.2e)", name, n)

        if self.freeze_base:
            self.freeze_base_weights()

    def _read_state(
        self, past_state: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
        """Returns (h, conv_state, dt_state, past_len)."""
        if past_state is None:
            return None, None, None, 0
        if isinstance(past_state, (tuple, list)):
            if len(past_state) >= 3:
                h = past_state[0]
                conv_state = past_state[1]
                dt_state = past_state[2] if len(past_state) > 2 else None
                past_len = h.shape[-1] if h is not None else 0
                if conv_state is not None:
                    past_len = conv_state.shape[-1] - (self.d_conv - 1)
                    past_len = max(0, past_len)
                return h, conv_state, dt_state, past_len
            elif len(past_state) == 2:
                return past_state[0], past_state[1], None, 0
            elif len(past_state) == 1:
                return past_state[0], None, None, 0
        return None, None, None, 0

    def _recurrent_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_ssm: torch.Tensor,
        C_ssm: torch.Tensor,
        D: torch.Tensor,
        state: torch.Tensor | None,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent selective scan with chunked activation checkpointing and bmm.

        Fix 1: Split L into chunks of size chunk_size and checkpoint each chunk
               during training to bound autograd memory to O(chunk_size).
        Fix 2: Use torch.bmm instead of torch.einsum inside the loop for cuBLAS
               acceleration and shared autograd backward kernels.
        """
        B, d_inner, L = u.shape
        d_state = A.shape[1]

        target_dtype = u.dtype
        h = (
            state.to(target_dtype)
            if state is not None
            else torch.zeros(B, d_inner, d_state, device=u.device, dtype=target_dtype)
        )

        delta = F.softplus(delta).to(target_dtype)
        delta_a = delta.unsqueeze(-1) * A.view(1, d_inner, 1, d_state).to(target_dtype)
        delta_a = torch.exp(delta_a).to(target_dtype)
        delta_b_u = (
            delta.unsqueeze(-1)
            * B_ssm.permute(0, 2, 1).unsqueeze(1).to(target_dtype)
            * u.unsqueeze(-1)
        ).to(target_dtype)
        C_in = C_ssm.to(target_dtype)

        def _scan_chunk(
            h_init: torch.Tensor,
            da_c: torch.Tensor,
            dbu_c: torch.Tensor,
            c_c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            L_c = da_c.shape[2]
            ys_c = []
            curr_h = h_init
            for i in range(L_c):
                curr_h = da_c[:, :, i, :] * curr_h + dbu_c[:, :, i, :]
                c_t = (
                    c_c[:, :, i].contiguous().unsqueeze(-1)
                )
                y_t = torch.bmm(curr_h, c_t).squeeze(-1)
                ys_c.append(y_t)
            return torch.stack(ys_c, dim=-1), curr_h

        ys_all = []
        import torch.utils.checkpoint as cp

        use_cp = self.training and torch.is_grad_enabled()

        for s in range(0, L, chunk_size):
            e = min(s + chunk_size, L)
            da_slice = delta_a[:, :, s:e, :]
            dbu_slice = delta_b_u[:, :, s:e, :]
            c_slice = C_in[:, :, s:e]

            if use_cp:
                y_chunk, h = cp.checkpoint(
                    _scan_chunk,
                    h,
                    da_slice,
                    dbu_slice,
                    c_slice,
                    use_reentrant=False,
                )
            else:
                y_chunk, h = _scan_chunk(
                    h,
                    da_slice,
                    dbu_slice,
                    c_slice,
                )
            ys_all.append(y_chunk)

        y = torch.cat(ys_all, dim=-1)
        y = y + u * D.view(1, d_inner, 1).to(target_dtype)
        return y, h

    def _parallel_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_ssm: torch.Tensor,
        C_ssm: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel scan dispatch with 3-tier fallback:
          Path 1: Official CUDA kernel via selective_scan (Linux only, when compiled)
          Path 2: Triton selective_scan (if mamba_ssm Triton ops available)
          Path 3: Fast chunked parallel scan (framework.core.selective_scan, O(1) ops per chunk)
          Path 4: Chunked recurrent scan with activation checkpointing (failsafe fallback)
        """
        if HAS_MAMBA_CUDA and selective_scan is not None:
            try:
                return selective_scan(
                    u=u,
                    delta=delta,
                    A=A,
                    B=B_ssm,
                    C=C_ssm,
                    D=D,
                    delta_softplus=True,
                )
            except Exception as e:
                logger.debug("CUDA selective_scan failed: %s", e)

        if HAS_TRITON_MAMBA and triton_selective_scan is not None:
            try:
                y = triton_selective_scan(
                    u.transpose(1, 2).contiguous(),
                    delta.transpose(1, 2).contiguous(),
                    A,
                    B_ssm.transpose(1, 2).contiguous(),
                    C_ssm.transpose(1, 2).contiguous(),
                    D,
                    delta_softplus=True,
                )
                return y.transpose(1, 2).contiguous()
            except Exception as e:
                logger.warning("Triton selective_scan failed: %s", e)

        if selective_scan is not None:
            try:
                return selective_scan(
                    u=u,
                    delta=delta,
                    A=A,
                    B=B_ssm,
                    C=C_ssm,
                    D=D,
                    delta_softplus=True,
                )
            except Exception as e:
                logger.debug("Parallel selective_scan fallback failed: %s", e)

        y, _ = self._recurrent_scan(u, delta, A, B_ssm, C_ssm, D, None)
        return y

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        past_state: Any | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ):
        if kwargs.get("encoder_hidden_states") is not None:
            raise NotImplementedError("MambaBlock does not support cross-attention")

        if past_state is None and past_key_value is not None:
            past_state = past_key_value

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        target_dtype = hidden_states.dtype

        h_prev, conv_prev, _, past_len = self._read_state(past_state)

        xz = self.in_proj(hidden_states)
        x_ssm, z = xz.chunk(2, dim=-1)

        if conv_prev is not None:
            conv_buf = conv_prev.transpose(1, 2).contiguous()
            x_cat = torch.cat([conv_buf, x_ssm], dim=1)
        else:
            x_cat = F.pad(x_ssm.transpose(1, 2), (self.d_conv - 1, 0)).transpose(1, 2)

        x_conv_full = self.conv1d(x_cat.transpose(1, 2))
        x_conv = x_conv_full[..., -L:].transpose(1, 2)

        if self.d_conv - 1 <= L:
            conv_next = x_cat[:, -(self.d_conv - 1) :, :].transpose(1, 2).contiguous()
        else:
            pad = self.d_conv - 1 - L
            conv_next = (
                torch.cat(
                    [
                        torch.zeros(B, pad, self.d_inner, device=device, dtype=x_cat.dtype),
                        x_cat,
                    ],
                    dim=1,
                )
                .transpose(1, 2)
                .contiguous()
            )

        x_ssm = F.silu(x_conv)

        x_dbl = self.x_proj(x_ssm)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        A = -torch.exp(self.A_log)

        u_t = x_ssm.transpose(1, 2)
        dt_t = dt.transpose(1, 2)
        B_t = B_ssm.transpose(1, 2)
        C_t = C_ssm.transpose(1, 2)

        if h_prev is None:
            y = self._parallel_scan(u_t, dt_t, A, B_t, C_t, self.D)
            h_next = None
        else:
            y, h_next = self._recurrent_scan(u_t, dt_t, A, B_t, C_t, self.D, h_prev)

        y = y.transpose(1, 2)

        if self.bidirectional and not self.is_decoder:
            x_rev = torch.flip(x_ssm, dims=[1])
            x_dbl_rev = self.x_proj_bwd(x_rev)
            dt_b, B_b, C_b = x_dbl_rev.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_bwd(dt_b)
            A_b = -torch.exp(self.A_log_bwd)
            y_b = self._parallel_scan(
                torch.flip(x_ssm, dims=[1]).transpose(1, 2),
                dt_b.transpose(1, 2),
                A_b,
                B_b.transpose(1, 2),
                C_b.transpose(1, 2),
                self.D_bwd,
            )
            y_b = torch.flip(y_b.transpose(1, 2), dims=[1])
            y = 0.5 * (y + y_b)

        y_gated = y.float() * F.silu(z.float())
        y = torch.nan_to_num(y_gated, nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(-65000.0, 65000.0).to(target_dtype)

        out = self.out_proj(y).to(target_dtype)
        out = torch.nan_to_num(out.float(), nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(-65000.0, 65000.0).to(target_dtype)

        if attention_mask is not None:
            m = attention_mask
            if m.dtype.is_floating_point:
                m = (m > -1e4).to(out.dtype)
            else:
                m = m.to(out.dtype)
            if m.dim() == 2:
                out = out * m.unsqueeze(-1)
            elif m.dim() == 3:
                m_red = (
                    m.diagonal(dim1=-2, dim2=-1)
                    if m.size(-1) == L
                    else m[..., -1]
                )
                out = out * m_red.unsqueeze(-1)
            elif m.dim() == 4:
                m_sq = m.squeeze(1)
                m_red = m_sq.diagonal(dim1=-2, dim2=-1) if m_sq.size(-1) == L else m_sq[..., -1]
                out = out * m_red.unsqueeze(-1)

        present_state = (h_next, conv_next)
        attn_weights = None

        fmt = self._return_format
        if fmt == "bert":
            return (out, attn_weights)
        if fmt == "gpt2":
            if output_attentions:
                return (out, present_state, attn_weights)
            return (out, present_state)
        if self.is_decoder:
            if output_attentions:
                return (out, attn_weights, present_state)
            return (out, present_state)
        return (out, attn_weights)

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Single-step autoregressive recurrent update for MambaBlock."""
        from neuralops.core.triton_kernels.state_update import causal_conv1d_step, fused_recurrent_ssm_step

        if state is None:
            state = {}

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm_1d = x_ssm.squeeze(1) if x_ssm.dim() == 3 else x_ssm

        conv_state = state.get("conv_state", None)
        if conv_state is None:
            conv_state = torch.zeros(B, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        x_conv, next_conv_state = causal_conv1d_step(x_ssm_1d, conv_state, self.conv1d.weight, self.conv1d.bias)
        x_act = F.silu(x_conv)

        x_dbl = self.x_proj(x_act)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        ssm_state = state.get("ssm_state", None)
        if ssm_state is None:
            ssm_state = torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=dtype)
        A = -torch.exp(self.A_log)
        y_ssm, next_ssm_state = fused_recurrent_ssm_step(x_act, dt, A, B_ssm, C_ssm, self.D, ssm_state)

        z_1d = z.squeeze(1) if z.dim() == 3 else z
        y_gated = y_ssm * F.silu(z_1d)
        out = self.out_proj(y_gated).unsqueeze(1)

        new_state = {
            "conv_state": next_conv_state,
            "ssm_state": next_ssm_state,
        }
        return out, new_state


Mamba = MambaBlock


def inject_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> nn.Module | Any | None:
    if _depth > 8:
        logger.warning("inject_mamba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_mamba(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if isinstance(src_attn, MambaBlock):
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        child = getattr(target, attr, None)
        if isinstance(child, MambaBlock):
            return None

    try:
        d_model = _infer_d_model(cfg, src_attn, None, **kwargs)
    except ValueError as e:
        logger.error("inject_mamba: %s", e)
        return None

    try:
        kwargs.pop("d_model", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("source_attn", None)
        kwargs.pop("old_attn", None)
        new_module = MambaBlock(
            config=cfg,
            d_model=d_model,
            source_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.error("MambaBlock instantiation failed: %s", e)
        return None

    if src_attn is not None:
        try:
            new_module.copy_weights(src_attn)
        except Exception as e:
            logger.error("MambaBlock.copy_weights failed: %s", e)
            return None

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_module.to(device=p.device, dtype=p.dtype)

    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_module.is_nested_encoder = True
                target.attention.self = new_module
            else:
                new_module.is_nested_encoder = False
                target.attention = new_module
            spliced = True
        elif hasattr(target, "self_attn"):
            new_module.is_nested_encoder = False
            target.self_attn = new_module
            spliced = True
        elif hasattr(target, "self"):
            new_module.is_nested_encoder = True
            target.self = new_module
            spliced = True
        elif hasattr(target, "attn"):
            new_module.is_nested_encoder = False
            target.attn = new_module
            spliced = True

        if not spliced:
            logger.error(
                "inject_mamba: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        new_module._return_format = _detect_return_format(src_attn, new_module.is_nested_encoder)

    return new_module


__all__ = ["MambaBlock", "Mamba", "inject_mamba"]
