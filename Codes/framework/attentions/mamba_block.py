"""
Faithful Mamba Block with SVD-based Attention Weight Distillation.

Based on Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (arXiv:2312.00752).

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.core.selective_scan import selective_scan
except Exception as e:  # pragma: no cover
    logging.getLogger(__name__).warning("selective_scan import failed: %s", e)
    selective_scan = None  # type: ignore

logger = logging.getLogger(__name__)


# =========================================================================
# Helpers
# =========================================================================
def _detect_return_format(source_attn: Optional[nn.Module], is_nested: bool) -> str:
    """MB-12: choose output tuple layout for the host model."""
    if is_nested:
        return "bert"   # (out, attn_weights)  [2-tuple]
    if source_attn is None:
        return "llama"
    cls = source_attn.__class__.__name__.lower()
    if any(k in cls for k in ("gpt2", "gptne", "gptj", "gpt_neo", "gpt_j")):
        return "gpt2"
    if hasattr(source_attn, "c_attn"):
        return "gpt2"
    return "llama"


def _infer_d_model(
    config: Optional[Any],
    source_attn: Optional[nn.Module],
    target: Optional[nn.Module],
    kwargs: Dict[str, Any],
) -> int:
    """MB-11: robust dimension inference with explicit error on failure."""
    # 1. explicit kwargs
    for key in ("d_model", "hidden_size", "dim", "n_embd"):
        v = kwargs.get(key)
        if isinstance(v, int) and v > 0:
            return v

    # 2. config
    if config is not None:
        for key in ("hidden_size", "d_model", "dim", "n_embd"):
            v = getattr(config, key, None)
            if isinstance(v, int) and v > 0:
                return v

    # 3. source attention
    if source_attn is not None:
        for attr in ("out_proj", "out_lin", "o_proj", "dense", "c_proj"):
            mod = getattr(source_attn, attr, None)
            if mod is not None and hasattr(mod, "out_features"):
                return int(mod.out_features)
        for attr in ("q_proj", "q_lin", "query", "c_attn"):
            mod = getattr(source_attn, attr, None)
            if mod is not None and hasattr(mod, "in_features"):
                return int(mod.in_features)
        for attr in ("hidden_size", "embed_dim", "dim"):
            v = getattr(source_attn, attr, None)
            if isinstance(v, int) and v > 0:
                return v

    # 4. target layer
    if target is not None:
        for attr in ("hidden_size", "d_model", "dim"):
            v = getattr(target, attr, None)
            if isinstance(v, int) and v > 0:
                return v

    raise ValueError(
        "MambaBlock: could not infer d_model. Pass d_model= or hidden_size= explicitly."
    )


# =========================================================================
# Mamba Block
# =========================================================================
@ATTENTION_REGISTRY.register("mamba")
@ATTENTION_REGISTRY.register("mamba_attention")
@ATTENTION_REGISTRY.register("mamba_block")
class MambaBlock(nn.Module):
    """Faithful S6 / Mamba block, production-hardened."""

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = False

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
        freeze_base: bool = False,
        **kwargs: Any,
    ) -> None:
        # Convenience: MambaBlock(768) support
        if isinstance(config, (int, float)) and d_model is None:
            d_model = int(config)
            config = None

        super().__init__()
        self.config = config

        # ---- d_model inference (MB-11) ----
        if d_model is None:
            d_model = _infer_d_model(config, kwargs.get("source_attn"), None, kwargs)
        self.d_model = int(d_model)

        # ---- Dimensions ----
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = int(dt_rank or math.ceil(self.d_model / 16))
        self.bidirectional = bool(bidirectional)
        self.freeze_base = bool(freeze_base)

        # ---- Decoder detection (MB-10) ----
        if is_decoder_override is not None:
            self.is_decoder = bool(is_decoder_override)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
            _dec_prefixes = (
                "gpt", "qwen", "llama", "mistral", "gemma",
                "phi", "opt", "bloom", "falcon", "olmo",
            )
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False) or getattr(config, "causal", False))
                if config else False
            ) or any(_mt.startswith(p) for p in _dec_prefixes)

        # ---- Projections ----
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # ---- S4D-real initialization of A ----
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- dt initialization ----
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
            nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
            if self.conv1d.bias is not None:
                nn.init.zeros_(self.conv1d.bias)

        # ---- Bidirectional (backward SSM) ----
        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(
                torch.log(A.clone() + 0.01 * torch.randn_like(A).abs())
            )
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(dt_init_inv)
                nn.init.uniform_(self.dt_proj_bwd.weight, -bound, bound)
            self.x_proj_bwd = nn.Linear(
                self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
            )
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        # ---- Output-format detection (MB-12) ----
        source_attn = kwargs.get("source_attn") or kwargs.get("old_attn")
        self.is_nested_encoder = False
        self._return_format = _detect_return_format(source_attn, False)

        if freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    def freeze_base_weights(self) -> None:
        """MB-04: match the framework's LoRA-freeze convention."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_base_weights(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Weight distillation (MB-07, MB-08)
    # ------------------------------------------------------------------
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

        extracted: Dict[str, Optional[torch.Tensor]] = {
            "q": None, "k": None, "v": None, "o": None
        }

        # --- 1. UniversalAdapter ---
        try:
            from framework.core.adapters import UniversalAdapter
            for proj in ("q", "k", "v", "o"):
                try:
                    w, _ = UniversalAdapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted[proj] = w
                except Exception:
                    pass
        except Exception:
            pass

        # --- 2. GPT-2 fused c_attn / c_proj ---
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
                extracted["k"] = w[split:2 * split, :]
                extracted["v"] = w[2 * split:, :]

        if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"] is None:
            o = src.c_proj
            base = getattr(o, "base_layer", o)
            w = getattr(base, "weight", getattr(o, "weight", None))
            if w is not None:
                w = w.data.clone()
                if w.ndim == 2 and w.shape[0] < w.shape[1]:
                    w = w.t().contiguous()
                extracted["o"] = w

        # --- 3. Flat / nested (BERT / RoBERTa) ---
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
            # Try nested first
            if nested is not None:
                for n in names:
                    mod = getattr(nested, n, None)
                    if mod is not None and hasattr(mod, "weight"):
                        extracted[proj] = mod.weight.data
                        break
            if extracted[proj] is not None:
                continue
            # Then flat on src
            for n in names:
                mod = getattr(src, n, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj] = mod.weight.data
                    break
            # Nested output.dense for "o"
            if extracted["o"] is None and proj == "o":
                out = getattr(src, "output", None)
                dense = getattr(out, "dense", None) if out is not None else None
                if dense is not None and hasattr(dense, "weight"):
                    extracted["o"] = dense.weight.data

        # Mark nested encoder for later (MB-05)
        cls_name = src.__class__.__name__
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                        "XLMRobertaSelfAttention") \
                or (nested is not None and hasattr(nested, "query")):
            self.is_nested_encoder = True

        # ---- Apply mapping (MB-07, MB-08) ----
        dtype = self.in_proj.weight.dtype
        device = self.in_proj.weight.device

        def _to_t(x: Optional[torch.Tensor], r: int, c: int) -> Optional[torch.Tensor]:
            """Project W (any shape) to target shape (r, c) via SVD."""
            if x is None:
                return None
            x = x.float()
            if x.ndim != 2:
                return None
            M, N = x.shape
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            k = min(r, c, s.numel())
            
            u_r = u[:min(r, M), :k]
            vh_c = vh[:k, :min(c, N)]
            core = (u_r * s[:k].unsqueeze(0)) @ vh_c
            
            out = torch.zeros(r, c, dtype=torch.float32, device=x.device)
            out[:core.size(0), :core.size(1)] = core
            
            if r > core.size(0):
                rem_r = r - core.size(0)
                reps_r = (rem_r + core.size(0) - 1) // core.size(0)
                tiled_r = core.repeat(reps_r, 1)[:rem_r, :]
                out[core.size(0):, :core.size(1)] = tiled_r
                
            if c > core.size(1):
                rem_c = c - core.size(1)
                reps_c = (rem_c + core.size(1) - 1) // core.size(1)
                tiled_c = out[:, :core.size(1)].repeat(1, reps_c)[:, :rem_c]
                out[:, core.size(1):] = tiled_c
                
            return out.to(device=device, dtype=dtype)

        with torch.no_grad():
            w_q = extracted["q"]
            w_k = extracted["k"]
            w_v = extracted["v"]
            w_o = extracted["o"]

            # ---------- in_proj: (2*d_inner, d_model) ----------
            # x_ssm path = rows [0 : d_inner]
            # z     path = rows [d_inner : 2*d_inner]
            if w_v is not None:
                w_v_p = _to_t(w_v, self.d_inner, self.d_model)
                if w_v_p is not None:
                    # x_ssm: full projection from W_v
                    self.in_proj.weight.data[:self.d_inner, :self.d_model].copy_(w_v_p)
                    # z: scaled copy so the gate starts balanced
                    self.in_proj.weight.data[self.d_inner:, :self.d_model].copy_(0.5 * w_v_p)

            # ---------- x_proj: (dt_rank + 2*d_state, d_inner) ----------
            # rows [0 : dt_rank]                     → dt  (keep init)
            # rows [dt_rank : dt_rank + d_state]     → B   (from W_k)
            # rows [dt_rank + d_state : ...]         → C   (from W_q)
            # MB-01: DO NOT zero the whole x_proj; only reset B/C rows.
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

            # ---------- out_proj: (d_model, d_inner) ----------
            if w_o is not None:
                w_o_p = _to_t(w_o, self.d_model, self.d_inner)
                if w_o_p is not None:
                    self.out_proj.weight.data.copy_(w_o_p)
                    if self.out_proj.bias is not None:
                        self.out_proj.bias.data.zero_()

            # ---------- conv1d: identity-at-init passthrough ----------
            self.conv1d.weight.data.zero_()
            if self.conv1d.weight.dim() == 3:
                # shape (d_inner, 1, d_conv)
                self.conv1d.weight.data[:, 0, -1] = 1.0
            if self.conv1d.bias is not None:
                self.conv1d.bias.data.zero_()

        # ---- Sanity check ----
        for name, mod in (("in_proj", self.in_proj),
                          ("x_proj", self.x_proj),
                          ("out_proj", self.out_proj)):
            n = float(torch.linalg.norm(mod.weight.data.float()).item())
            if n < 1e-6:
                logger.warning("MambaBlock: '%s' near-zero norm (%.2e)", name, n)

        if self.freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _read_state(self, past_state: Any) -> Tuple[Optional[torch.Tensor],
                                                     Optional[torch.Tensor],
                                                     Optional[torch.Tensor],
                                                     int]:
        """Returns (h, conv_state, dt_state, past_len)."""
        if past_state is None:
            return None, None, None, 0
        if isinstance(past_state, (tuple, list)):
            if len(past_state) >= 3:
                h = past_state[0]
                conv_state = past_state[1]
                dt_state = past_state[2] if len(past_state) > 2 else None
                past_len = h.shape[-1] if h is not None else 0
                # past_len from conv_state if available
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
        u: torch.Tensor,       # (B, d_inner, L)
        delta: torch.Tensor,   # (B, d_inner, L)
        A: torch.Tensor,       # (d_inner, d_state)
        B_ssm: torch.Tensor,   # (B, d_state, L)
        C_ssm: torch.Tensor,   # (B, d_state, L)
        D: torch.Tensor,       # (d_inner,)
        state: Optional[torch.Tensor],  # (B, d_inner, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent selective scan with state. For the cached path only."""
        B, d_inner, L = u.shape
        d_state = A.shape[1]

        h = state if state is not None else torch.zeros(
            B, d_inner, d_state, device=u.device, dtype=torch.float32
        )

        delta = F.softplus(delta)
        delta_a = delta.unsqueeze(-1) * A.view(1, d_inner, 1, d_state)  # (B, d_inner, L, d_state)
        delta_a = torch.exp(delta_a)
        delta_b_u = delta.unsqueeze(-1) * B_ssm.permute(0, 2, 1).unsqueeze(1) \
                    * u.unsqueeze(-1)  # (B, d_inner, L, d_state)

        ys = []
        for t in range(L):
            h = delta_a[:, :, t, :] * h + delta_b_u[:, :, t, :]
            y_t = torch.einsum("bhd,bd->bh", h, C_ssm[:, :, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=-1)  # (B, d_inner, L)
        y = y + u * D.view(1, d_inner, 1)
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
        """Prefill path: use the fast selective_scan if available."""
        if selective_scan is not None:
            try:
                y = selective_scan(
                    u=u, delta=delta, A=A, B=B_ssm, C=C_ssm, D=D,
                    delta_softplus=True,
                )
                return y
            except Exception as e:
                logger.debug("selective_scan failed, falling back to recurrent: %s", e)
        y, _ = self._recurrent_scan(u, delta, A, B_ssm, C_ssm, D, None)
        return y

    # ------------------------------------------------------------------
    # Forward (MB-02, MB-03, MB-09, MB-12)
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        past_state: Optional[Any] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ):
        if kwargs.get("encoder_hidden_states") is not None:
            raise NotImplementedError("MambaBlock does not support cross-attention")

        # Backward-compat: accept past_key_value as an alias for past_state
        if past_state is None and past_key_value is not None:
            past_state = past_key_value

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        target_dtype = hidden_states.dtype

        # ---- Read state ----
        h_prev, conv_prev, _, past_len = self._read_state(past_state)

        # ---- 1. Input projection ----
        xz = self.in_proj(hidden_states)                    # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                      # each (B, L, d_inner)

        # ---- 2. Depthwise conv1d with streaming state ----
        if conv_prev is not None:
            # conv_prev: (B, d_inner, d_conv - 1)
            # Concatenate along time axis
            x_cat = torch.cat(
                [conv_prev.transpose(1, 2), x_ssm], dim=1
            )  # (B, L + d_conv - 1, d_inner)
        else:
            x_cat = x_ssm

        x_conv_full = self.conv1d(x_cat.transpose(1, 2))    # (B, d_inner, L' + d_conv - 1)
        # Take only the L last positions
        x_conv = x_conv_full[..., -L:].transpose(1, 2)      # (B, L, d_inner)

        # New conv state: last (d_conv - 1) tokens of input
        if L >= self.d_conv - 1:
            conv_next = x_cat[:, -(self.d_conv - 1):, :].transpose(1, 2).contiguous()
        else:
            # pad with zero if sequence shorter than kernel
            pad = self.d_conv - 1 - L
            conv_next = torch.cat(
                [torch.zeros(B, pad, self.d_inner, device=device, dtype=x_cat.dtype),
                 x_cat],
                dim=1,
            ).transpose(1, 2).contiguous()

        x_ssm = F.silu(x_conv)

        # ---- 3. Selective parameters ----
        x_dbl = self.x_proj(x_ssm)                          # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)                                # (B, L, d_inner)

        A = -torch.exp(self.A_log)                           # (d_inner, d_state)

        # ---- 4. Selective scan ----
        u_t = x_ssm.transpose(1, 2)                          # (B, d_inner, L)
        dt_t = dt.transpose(1, 2)
        B_t = B_ssm.transpose(1, 2)
        C_t = C_ssm.transpose(1, 2)

        if h_prev is None:
            y = self._parallel_scan(u_t, dt_t, A, B_t, C_t, self.D)
            h_next = None
            if kwargs.get("use_cache", False):
                _, h_next = self._recurrent_scan(u_t, dt_t, A, B_t, C_t, self.D, None)
        else:
            y, h_next = self._recurrent_scan(u_t, dt_t, A, B_t, C_t, self.D, h_prev)

        y = y.transpose(1, 2)                                # (B, L, d_inner)

        # ---- 5. Bidirectional (encoder only) ----
        if self.bidirectional and not self.is_decoder:
            x_rev = torch.flip(x_ssm, dims=[1])
            x_dbl_rev = self.x_proj_bwd(x_rev)
            dt_b, B_b, C_b = x_dbl_rev.split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
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

        # ---- 6. Gating ----
        y = y.to(target_dtype) * F.silu(z.to(target_dtype))

        # ---- 7. Output projection ----
        if self.is_nested_encoder:
            # Nested encoder: outer layer will project. Skip out_proj.
            out = y.to(target_dtype)
        else:
            out = self.out_proj(y).to(target_dtype)

        # ---- 8. attention_mask (MB-03, MB-09) ----
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                out = out * attention_mask.unsqueeze(-1).to(out.dtype)
            elif attention_mask.dim() == 3:
                # (B, Lq, Lk) — take the last column (query's validity) or diagonal
                m = attention_mask.diagonal(dim1=-2, dim2=-1) if attention_mask.size(-1) == L \
                    else attention_mask[..., -1]
                out = out * m.unsqueeze(-1).to(out.dtype)
            elif attention_mask.dim() == 4:
                # (B, 1, Lq, Lk) — same reduction
                m = attention_mask.squeeze(1)
                m = m.diagonal(dim1=-2, dim2=-1) if m.size(-1) == L else m[..., -1]
                out = out * m.unsqueeze(-1).to(out.dtype)

        # ---- 9. Output tuple (MB-12) ----
        present_state = (h_next, conv_next)
        attn_weights = None

        fmt = self._return_format
        if fmt in ("bert", "gpt2"):
            return (out, attn_weights)
        # llama
        return (out, attn_weights, present_state)


# Convenience alias
Mamba = MambaBlock


# =========================================================================
# Injector (MB-06: no dead `rank`)
# =========================================================================
@ATTENTION_REGISTRY.register("mamba_module_inject")
@ATTENTION_REGISTRY.register("mamba_inject")
@ATTENTION_REGISTRY.register("inject_mamba")
def inject_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[nn.Module, Any]]:
    if _depth > 8:
        logger.warning("inject_mamba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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

    # ---- Single layer ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    # Idempotency
    if isinstance(src_attn, MambaBlock):
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        child = getattr(target, attr, None)
        if isinstance(child, MambaBlock):
            return None

    # ---- d_model inference (MB-11) ----
    try:
        d_model = _infer_d_model(cfg, src_attn, target, kwargs)
    except ValueError as e:
        logger.error("inject_mamba: %s", e)
        return None

    # ---- Build the block ----
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

    # ---- Weight transfer ----
    if src_attn is not None:
        try:
            new_module.copy_weights(src_attn)
        except Exception as e:
            logger.error("MambaBlock.copy_weights failed: %s", e)
            return None

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_module.to(device=p.device, dtype=p.dtype)

    # ---- Splice into the layer ----
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
            logger.error("inject_mamba: no writable attention attribute on %s",
                         type(target).__name__)
            return None

        # Recompute return format (MB-12)
        new_module._return_format = _detect_return_format(
            src_attn, new_module.is_nested_encoder
        )

    return new_module


__all__ = ["MambaBlock", "Mamba", "inject_mamba"]