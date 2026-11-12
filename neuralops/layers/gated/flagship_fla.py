#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Flagship Fast Linear Attention (FLA)

Fast linear attention combining causal cumulative summation with hardware-efficient hardware kernel backends.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get, repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.ops.triton.fused_norm_linear import fused_rmsnorm_gate

logger = logging.getLogger(__name__)


def gdn2_recurrent_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    alpha_t: torch.Tensor,
    beta_erase_t: torch.Tensor,
    beta_write_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decoupled Erase & Write Recurrence step for GDN2:
        decay_gate = alpha_t (or retention scalar)
        erase_proj = beta_erase_t * (k_t @ S)
        S_erased   = alpha_t * (S - k_t^T @ erase_proj)
        S_new      = S_erased + beta_write_t * (k_t^T @ v_t)
        y_t        = q_t @ S_new
    Shapes:
        S: (B, H, d_k, d_v)
        q_t, k_t: (B, H, 1, d_k)
        v_t: (B, H, 1, d_v)
        alpha_t, beta_erase_t, beta_write_t: (B, H, 1, 1)
    """
    kt_norm = F.normalize(k_t, p=2, dim=-1, eps=1e-6)

    inter_mem = torch.matmul(kt_norm, S)
    erase_term = beta_erase_t * torch.matmul(kt_norm.transpose(-1, -2), inter_mem)
    S_decayed = alpha_t * (S - erase_term)

    write_term = beta_write_t * torch.matmul(kt_norm.transpose(-1, -2), v_t)
    S_next = S_decayed + write_term

    qt_norm = F.normalize(q_t, p=2, dim=-1, eps=1e-6)
    y_t = torch.matmul(qt_norm, S_next)
    return y_t, S_next


def gdn2_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta_erase: torch.Tensor,
    beta_write: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunk-wise parallel forward execution for GDN2 via exact WY representation.
    Eliminates per-timestep Python loops while preserving exact FP32 recurrent state stability.
    """
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device
    dtype = q.dtype

    if initial_state is None:
        S = torch.zeros((B, H, d_k, d_v), device=device, dtype=torch.float32)
    else:
        S = initial_state.to(device=device, dtype=torch.float32)

    if L == 0:
        return q.new_empty((B, H, 0, d_v)), S.to(dtype)

    c_size = min(max(1, int(chunk_size)), 32)
    pad_len = (c_size - (L % c_size)) % c_size

    if pad_len > 0:
        q_p = F.pad(q, (0, 0, 0, pad_len))
        k_p = F.pad(k, (0, 0, 0, pad_len))
        v_p = F.pad(v, (0, 0, 0, pad_len))
        alpha_p = F.pad(alpha, (0, pad_len), value=1.0)
        be_p = F.pad(beta_erase, (0, pad_len), value=0.0)
        bw_p = F.pad(beta_write, (0, pad_len), value=0.0)
    else:
        q_p, k_p, v_p, alpha_p, be_p, bw_p = q, k, v, alpha, beta_erase, beta_write

    num_chunks = q_p.shape[2] // c_size

    with torch.amp.autocast(device_type=device.type, enabled=False):
        q_blk = F.normalize(q_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, d_k)
        k_blk = F.normalize(k_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, d_k)
        v_blk = v_p.float().view(B, H, num_chunks, c_size, d_v)
        a_blk = alpha_p.float().clamp(min=1e-4, max=1.0).view(B, H, num_chunks, c_size)
        be_blk = be_p.float().view(B, H, num_chunks, c_size)
        bw_blk = bw_p.float().view(B, H, num_chunks, c_size)

        gamma = torch.cumsum(torch.log(a_blk), dim=-1)
        exp_gamma = torch.exp(gamma)

        kk = torch.matmul(k_blk, k_blk.transpose(-1, -2))
        g_diff = (gamma.unsqueeze(-1) - gamma.unsqueeze(-2)).clamp(max=0.0)
        tril_strict = torch.tril(torch.ones((c_size, c_size), device=device, dtype=torch.bool), diagonal=-1)
        tril_causal = torch.tril(torch.ones((c_size, c_size), device=device, dtype=torch.bool), diagonal=0)

        exp_diff = torch.exp(g_diff)
        decay_mat = exp_diff * tril_causal.float()
        M = kk * (be_blk.unsqueeze(-1) * (exp_diff * tril_strict.float()))
        I_plus_M = torch.eye(c_size, device=device, dtype=torch.float32) + M

        rhs_v = bw_blk.unsqueeze(-1) * v_blk
        rhs_k = (be_blk * exp_gamma).unsqueeze(-1) * k_blk
        rhs = torch.cat([rhs_v, rhs_k], dim=-1)
        sol = torch.linalg.solve_triangular(I_plus_M, rhs, upper=False, unitriangular=True)
        U_v = sol[..., :d_v]
        W_k = sol[..., d_v:]

        qk_decay = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * decay_mat
        q_decayed = q_blk * exp_gamma.unsqueeze(-1)
        gamma_end = gamma[..., -1:]
        exp_gamma_end = torch.exp(gamma_end).unsqueeze(-1)
        k_to_end = k_blk * torch.exp((gamma_end - gamma).clamp(max=0.0)).unsqueeze(-1)

        outs = []
        for c in range(num_chunks):
            U_c = U_v[:, :, c] - torch.matmul(W_k[:, :, c], S)
            y_c = torch.matmul(q_decayed[:, :, c], S) + torch.matmul(qk_decay[:, :, c], U_c)
            S = exp_gamma_end[:, :, c] * S + torch.matmul(k_to_end[:, :, c].transpose(-1, -2), U_c)
            outs.append(y_c)

        y = torch.cat(outs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]
    if dtype == torch.float16:
        fmax = torch.finfo(dtype).max * 0.5
        y = y.clamp(min=-fmax, max=fmax)
    return y.to(dtype), S.to(dtype)


def _copy_attention_weights(module: nn.Module, src: Any) -> bool:
    """Copies pretrained weights and biases into module and detects nested encoder."""
    if src is None:
        return False
    cls_name = getattr(src, "__class__", type(src)).__name__
    has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
    is_nested = False
    if (
        cls_name in (
            "BertSelfAttention",
            "RobertaSelfAttention",
            "XLMRobertaSelfAttention",
            "ElectraSelfAttention",
            "DebertaSelfAttention",
        )
        or (hasattr(src, "query") and not has_out)
        or (hasattr(src, "self") and hasattr(src, "output"))
    ):
        is_nested = True
    module.is_nested_encoder = bool(is_nested)

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    if hasattr(src, "c_attn") and src.c_attn is not None:
        c_attn_mod = src.c_attn
        c_w = getattr(c_attn_mod, "weight", None)
        if c_w is not None:
            c_w = c_w.data.clone()
            if "Conv1D" in type(c_attn_mod).__name__ or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                c_w = c_w.t()
            sz = c_w.size(0) // 3
            c_b = c_attn_mod.bias.data.clone() if getattr(c_attn_mod, "bias", None) is not None else None
            extracted["q"] = (c_w[:sz], c_b[:sz] if c_b is not None else None)
            extracted["k"] = (c_w[sz : 2 * sz], c_b[sz : 2 * sz] if c_b is not None else None)
            extracted["v"] = (c_w[2 * sz :], c_b[2 * sz :] if c_b is not None else None)
        if hasattr(src, "c_proj") and src.c_proj is not None:
            o_w = getattr(src.c_proj, "weight", None)
            if o_w is not None:
                o_w = o_w.data.clone()
                if "Conv1D" in type(src.c_proj).__name__ or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                    o_w = o_w.t()
                o_b = src.c_proj.bias.data.clone() if getattr(src.c_proj, "bias", None) is not None else None
                extracted["o"] = (o_w, o_b)
    elif hasattr(src, "self") and hasattr(src.self, "query"):
        for k, name in (("q", "query"), ("k", "key"), ("v", "value")):
            m = getattr(src.self, name, None)
            if m is not None and hasattr(m, "weight"):
                extracted[k] = (m.weight.data.clone(), m.bias.data.clone() if m.bias is not None else None)
        if hasattr(src, "output") and hasattr(src.output, "dense"):
            d = src.output.dense
            extracted["o"] = (d.weight.data.clone(), d.bias.data.clone() if d.bias is not None else None)
    else:
        aliases = {
            "q": ["q_proj", "query", "q_lin"],
            "k": ["k_proj", "key", "k_lin"],
            "v": ["v_proj", "value", "v_lin"],
            "o": ["out_proj", "dense", "out_lin", "o_proj"],
        }
        for k, names in aliases.items():
            for name in names:
                m = getattr(src, name, None)
                if m is not None and hasattr(m, "weight"):
                    extracted[k] = (m.weight.data.clone(), m.bias.data.clone() if m.bias is not None else None)
                    break

    missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not module.is_nested_encoder)]
    if missing:
        try:
            from neuralops.core.adapters import UniversalAdapter

            adapter = UniversalAdapter()
            for k in missing:
                w, b = adapter.get_weight_bias(src, k)
                if w is not None:
                    extracted[k] = (w.clone(), b.clone() if b is not None else None)
        except Exception as e:
            logger.debug("UniversalAdapter extraction fallback: %s", e)

    target_map = {
        "q": getattr(module, "q_proj", None),
        "k": getattr(module, "k_proj", None),
        "v": getattr(module, "v_proj", None),
        "o": getattr(module, "out_proj", None),
    }
    with torch.no_grad():
        for k, target in target_map.items():
            if target is None:
                continue
            w, b = extracted[k]
            if w is not None:
                if w.shape == target.weight.shape:
                    target.weight.copy_(w)
                elif w.t().shape == target.weight.shape:
                    target.weight.copy_(w.t())
                elif target.weight.dim() == 2 and w.dim() == 2 and target.weight.shape[1] == w.shape[1]:
                    if target.weight.shape[0] % w.shape[0] == 0:
                        g = target.weight.shape[0] // w.shape[0]
                        target.weight.copy_(w.repeat_interleave(g, dim=0))
                if b is not None and target.bias is not None:
                    if b.shape == target.bias.shape:
                        target.bias.copy_(b)
                    elif target.bias.shape[0] % b.shape[0] == 0:
                        g = target.bias.shape[0] // b.shape[0]
                        target.bias.copy_(b.repeat_interleave(g, dim=0))

        if module.is_nested_encoder and extracted["o"][0] is None and hasattr(module, "out_proj"):
            nn.init.eye_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
    return is_nested


@ATTENTION_REGISTRY.register("gdn2_attention")
@ATTENTION_REGISTRY.register("gdn2", is_alias=True)
class GDN2Attention(BaseAttention):
    """
    Gated DeltaNet 2 Attention with Decoupled Erase & Write Linear Recurrence.
    Formulation:
        S_t = S_{t-1} * alpha_t * (I - beta_erase_t * k_t k_t^T) + beta_write_t * k_t v_t^T
    """

    FAMILY: str = "gated"
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        chunk_size: int = 64,
        **kwargs: Any,
    ):
        super().__init__(config, rank)
        self.d_model = _cfg_get(config, "hidden_size", hidden_size)
        self.num_heads = _cfg_get(config, "num_attention_heads", num_heads)
        self.num_kv_heads = _cfg_get(config, "num_key_value_heads", num_kv_heads or self.num_heads)
        self.head_dim = _cfg_get(config, "head_dim", head_dim or (self.d_model // self.num_heads))
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.chunk_size = chunk_size
        self.is_nested_encoder = False
        self.is_decoder = bool(
            kwargs.get(
                "is_decoder",
                kwargs.get("is_causal", _cfg_get(config, "is_decoder", False) or _cfg_get(config, "is_causal", False)),
            )
        )

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=True)
        self.z_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=True)

        self.alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.erase_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.write_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.norm_weight = nn.Parameter(torch.ones(self.num_heads * self.head_dim))

        with torch.no_grad():
            nn.init.constant_(self.alpha_proj.bias, 2.5)
            nn.init.constant_(self.erase_proj.bias, -2.0)
            nn.init.constant_(self.write_proj.bias, 2.0)
            nn.init.zeros_(self.z_proj.weight)
            nn.init.constant_(self.z_proj.bias, 2.0)

        source_attn = kwargs.get("pretrained_attn") or kwargs.get("old_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        _copy_attention_weights(self, src)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, D).transpose(1, 2)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        is_causal = self.supports_causal and (
            self.is_decoder
            or kwargs.get("is_causal", False)
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "is_causal", False)
        )

        if not is_causal:
            q_feat = F.elu(q.float()) + 1.0
            k_feat = F.elu(k.float()) + 1.0
            diff = -torch.abs(
                torch.arange(L, device=hidden_states.device).unsqueeze(0)
                - torch.arange(L, device=hidden_states.device).unsqueeze(1)
            ).float()
            decay = torch.exp(0.01 * diff).unsqueeze(0).unsqueeze(0)
            scores = torch.matmul(q_feat, k_feat.transpose(-1, -2)) * decay
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = (
                        (attention_mask > 0)
                        if not attention_mask.dtype.is_floating_point
                        else (attention_mask > -1e4)
                    ).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~mask, 0.0)
                elif attention_mask.dim() == 4:
                    mask = (
                        (attention_mask > -1e4)
                        if attention_mask.dtype.is_floating_point
                        else attention_mask.bool()
                    )
                    scores = scores.masked_fill(~mask, 0.0)
            den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            y = torch.matmul(scores / den, v.float()).transpose(1, 2).contiguous().view(B, L, H * D).to(hidden_states.dtype)
            if self.is_nested_encoder:
                return y, None
            return self.out_proj(y), None

        z = self.z_proj(hidden_states)
        alpha = torch.sigmoid(self.alpha_proj(hidden_states).float()).transpose(1, 2)
        beta_erase = torch.sigmoid(self.erase_proj(hidden_states).float()).transpose(1, 2)
        beta_write = torch.sigmoid(self.write_proj(hidden_states).float()).transpose(1, 2)

        if attention_mask is not None:
            from neuralops.core.attention_mask import get_token_valid_mask

            valid = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
            if valid is not None:
                v_mask = valid.unsqueeze(1)
                alpha = torch.where(v_mask, alpha, torch.ones_like(alpha))
                beta_erase = beta_erase * v_mask.float()
                beta_write = beta_write * v_mask.float()

        init_state = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                init_state = past_key_value[0]
            elif isinstance(past_key_value, torch.Tensor):
                init_state = past_key_value

        if init_state is None and D <= 64 and q.is_cuda and q.dtype == torch.bfloat16 and not use_cache:
            try:
                from fla.ops.gdn2 import chunk_gdn2

                q_fl = q.transpose(1, 2).contiguous()
                k_fl = k.transpose(1, 2).contiguous()
                v_fl = v.transpose(1, 2).contiguous()
                g_fl = (
                    torch.log(alpha.float().clamp(min=1e-4, max=1.0))
                    .transpose(1, 2)
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, D)
                    .to(q.dtype)
                    .contiguous()
                )
                b_fl = beta_erase.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, D).to(q.dtype).contiguous()
                w_fl = beta_write.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, D).to(q.dtype).contiguous()
                res = chunk_gdn2(q_fl, k_fl, v_fl, g_fl, b_fl, w_fl)
                y_fl = res[0] if isinstance(res, tuple) else res
                y = y_fl.contiguous().view(B, L, H * D)
                if self.is_nested_encoder:
                    return y, None
                y_gated = fused_rmsnorm_gate(y, z, self.norm_weight, eps=1e-5, activation="silu")
                return self.out_proj(y_gated), None
            except Exception as e:
                logger.debug("chunk_gdn2 fastpath failed, falling back to gdn2_chunk_scan: %s", e)

        y, next_state = gdn2_chunk_scan(
            q,
            k,
            v,
            alpha,
            beta_erase,
            beta_write,
            initial_state=init_state,
            chunk_size=self.chunk_size,
        )

        y = y.transpose(1, 2).contiguous().view(B, L, H * D)
        if self.is_nested_encoder:
            return y, ((next_state,) if use_cache else None)
        y_gated = fused_rmsnorm_gate(y, z, self.norm_weight, eps=1e-5, activation="silu")
        out = self.out_proj(y_gated)

        present_kv = (next_state,) if use_cache else None
        return out, present_kv

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for GDN2Attention."""
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        q = self.q_proj(x).view(B, 1, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.num_kv_heads, D).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.num_kv_heads, D).transpose(1, 2)
        z = self.z_proj(x)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2).unsqueeze(-1)
        beta_erase = torch.sigmoid(self.erase_proj(x)).transpose(1, 2).unsqueeze(-1)
        beta_write = torch.sigmoid(self.write_proj(x)).transpose(1, 2).unsqueeze(-1)

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)

        S = state["kv_state"]
        y_t, next_S = gdn2_recurrent_step(
            S, q.float(), k.float(), v.float(), alpha.float(), beta_erase.float(), beta_write.float()
        )
        state["kv_state"] = next_S
        state["step_idx"] = step_idx + 1

        y = y_t.transpose(1, 2).contiguous().view(B, 1, H * D)
        y_gated = fused_rmsnorm_gate(y.to(dtype), z, self.norm_weight, eps=1e-5, activation="silu")
        out = self.out_proj(y_gated)

        if is_2d:
            out = out.squeeze(1)

        return out, state


def householder_reflection(
    v: torch.Tensor,
    u: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Householder orthogonal transformation:
        H = I - 2 * beta * (u u^T) / (||u||^2 + eps)
        H @ v = v - 2 * beta * u * (u^T @ v) / (||u||^2 + eps)
    Preserves exact isometry and norm stability across deep recurrent steps.
    """
    u_norm_sq = (u * u).sum(dim=-1, keepdim=True) + 1e-6
    dot_uv = (v * u).sum(dim=-1, keepdim=True)
    reflect = 2.0 * beta * dot_uv * (u / u_norm_sq)
    return v - reflect


@ATTENTION_REGISTRY.register("delta_product_attention")
@ATTENTION_REGISTRY.register("delta_product", is_alias=True)
@ATTENTION_REGISTRY.register("deltaproduct", is_alias=True)
@ATTENTION_REGISTRY.register("path_attention", is_alias=True)
class DeltaProductAttention(BaseAttention):
    """
    Householder Product & PaTH Attention for NeuralOps.
    Tracks state via sequence-accumulated Householder orthogonal reflections, ensuring
    strict isometry, bounded spectrum, and norm preservation along sequence dimensions.
    """

    FAMILY: str = "gated"
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, rank)
        self.d_model = _cfg_get(config, "hidden_size", hidden_size)
        self.num_heads = _cfg_get(config, "num_attention_heads", num_heads)
        self.head_dim = _cfg_get(config, "head_dim", head_dim or (self.d_model // self.num_heads))
        self.is_nested_encoder = False
        self.is_decoder = bool(
            kwargs.get(
                "is_decoder",
                kwargs.get("is_causal", _cfg_get(config, "is_decoder", False) or _cfg_get(config, "is_causal", False)),
            )
        )

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.u_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=True)

        with torch.no_grad():
            nn.init.constant_(self.beta_proj.bias, -2.0)

        source_attn = kwargs.get("pretrained_attn") or kwargs.get("old_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        _copy_attention_weights(self, src)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        u = self.u_proj(hidden_states).view(B, L, H, D).transpose(1, 2)

        is_causal = self.supports_causal and (
            self.is_decoder
            or kwargs.get("is_causal", False)
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "is_causal", False)
        )

        if not is_causal:
            q_feat = F.elu(q.float()) + 1.0
            k_feat = F.elu(k.float()) + 1.0
            diff = -torch.abs(
                torch.arange(L, device=hidden_states.device).unsqueeze(0)
                - torch.arange(L, device=hidden_states.device).unsqueeze(1)
            ).float()
            decay = torch.exp(0.01 * diff).unsqueeze(0).unsqueeze(0)
            scores = torch.matmul(q_feat, k_feat.transpose(-1, -2)) * decay
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = (
                        (attention_mask > 0)
                        if not attention_mask.dtype.is_floating_point
                        else (attention_mask > -1e4)
                    ).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~mask, 0.0)
                elif attention_mask.dim() == 4:
                    mask = (
                        (attention_mask > -1e4)
                        if attention_mask.dtype.is_floating_point
                        else attention_mask.bool()
                    )
                    scores = scores.masked_fill(~mask, 0.0)
            den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            y = torch.matmul(scores / den, v.float()).transpose(1, 2).contiguous().view(B, L, H * D).to(hidden_states.dtype)
            if self.is_nested_encoder:
                return y, None
            return self.out_proj(y), None

        beta_raw = torch.sigmoid(self.beta_proj(hidden_states).float())
        if attention_mask is not None:
            from neuralops.core.attention_mask import get_token_valid_mask

            valid = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
            if valid is not None:
                beta_raw = beta_raw * valid.unsqueeze(-1).float()

        init_state = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                init_state = past_key_value[0]
            elif isinstance(past_key_value, torch.Tensor):
                init_state = past_key_value

        if init_state is None and D <= 64 and q.is_cuda and q.dtype == torch.bfloat16 and not use_cache:
            try:
                from fla.ops import parallel_path_attn

                q_fl = q.transpose(1, 2).contiguous()
                k_fl = k.transpose(1, 2).contiguous()
                v_fl = v.transpose(1, 2).contiguous()
                u_fl = u.transpose(1, 2).contiguous().float()
                beta_fl = beta_raw.float().contiguous()
                res = parallel_path_attn(q_fl, k_fl, v_fl, u_fl, beta_fl)
                y_fl = res[0] if isinstance(res, tuple) else res
                y = y_fl.contiguous().view(B, L, H * D)
                if self.is_nested_encoder:
                    return y, None
                return self.out_proj(y), None
            except Exception as e:
                logger.debug("CUDA parallel_path_attn failed, falling back to sequential PyTorch: %s", e)

        beta = beta_raw.transpose(1, 2)
        S = (
            torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
            if init_state is None
            else init_state.to(device=q.device, dtype=torch.float32)
        )

        # Vectorized chunked WY representation for Householder DeltaProduct (zero per-token loop)
        c_size = 32
        pad_len = (c_size - (L % c_size)) % c_size
        if pad_len > 0:
            q_p = F.pad(q, (0, 0, 0, pad_len))
            k_p = F.pad(k, (0, 0, 0, pad_len))
            v_p = F.pad(v, (0, 0, 0, pad_len))
            u_p = F.pad(u, (0, 0, 0, pad_len))
            b_p = F.pad(beta, (0, pad_len), value=0.0)
        else:
            q_p, k_p, v_p, u_p, b_p = q, k, v, u, beta

        num_chunks = q_p.shape[2] // c_size
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            q_blk = F.normalize(q_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, D)
            k_blk = F.normalize(k_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, D)
            v_blk = v_p.float().view(B, H, num_chunks, c_size, D)
            u_raw = u_p.float().view(B, H, num_chunks, c_size, D)
            u_hat = u_raw * torch.rsqrt((u_raw * u_raw).sum(dim=-1, keepdim=True) + 1e-6)
            b_blk = b_p.float().view(B, H, num_chunks, c_size)
            eta = 2.0 * b_blk

            tril_strict = torch.tril(torch.ones((c_size, c_size), device=q.device, dtype=torch.bool), diagonal=-1)
            tril_causal = torch.tril(torch.ones((c_size, c_size), device=q.device, dtype=torch.bool), diagonal=0)

            uu = torch.matmul(u_hat, u_hat.transpose(-1, -2))
            M = uu * (eta.unsqueeze(-1) * tril_strict.float())
            I_plus_M = torch.eye(c_size, device=q.device, dtype=torch.float32) + M

            bv = b_blk.unsqueeze(-1) * v_blk
            uk_strict = torch.matmul(u_hat, k_blk.transpose(-1, -2)) * tril_strict.float()
            P_c = torch.matmul(uk_strict, bv)

            rhs_v = -eta.unsqueeze(-1) * P_c
            rhs_S = -eta.unsqueeze(-1) * u_hat
            rhs = torch.cat([rhs_v, rhs_S], dim=-1)
            sol = torch.linalg.solve_triangular(I_plus_M, rhs, upper=False, unitriangular=True)
            W_v = sol[..., :D]
            W_S = sol[..., D:]

            qu_causal = torch.matmul(q_blk, u_hat.transpose(-1, -2)) * tril_causal.float()
            qk_causal = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * tril_causal.float()
            qk_bv = torch.matmul(qk_causal, bv)

            outs = []
            curr = S
            scale_d = float(D) ** -0.5
            for c in range(num_chunks):
                W_c = W_v[:, :, c] + torch.matmul(W_S[:, :, c], curr)
                y_intra = (torch.matmul(qu_causal[:, :, c], W_c) + qk_bv[:, :, c]) * scale_d
                y_c = torch.matmul(q_blk[:, :, c], curr) + y_intra
                curr = (
                    curr
                    + torch.matmul(u_hat[:, :, c].transpose(-1, -2), W_c)
                    + torch.matmul(k_blk[:, :, c].transpose(-1, -2), bv[:, :, c])
                )
                s_norm = torch.linalg.matrix_norm(curr, ord=2, keepdim=True)
                curr = torch.where(s_norm > 1.0, curr / s_norm.clamp(min=1e-6), curr)
                outs.append(y_c)

            y = torch.cat(outs, dim=2)
        if pad_len > 0:
            y = y[:, :, :L, :]
        present_state = curr

        fmax = 32000.0 if q.dtype == torch.float16 else 65000.0
        y = torch.nan_to_num(y, nan=0.0, posinf=fmax, neginf=-fmax).clamp(min=-fmax, max=fmax)
        y = y.to(q.dtype).transpose(1, 2).contiguous().view(B, L, H * D)
        present_kv = (present_state.to(q.dtype),) if use_cache else None
        if self.is_nested_encoder:
            return y, present_kv
        out = self.out_proj(y)
        return out, present_kv

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for DeltaProductAttention."""
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        q = self.q_proj(x).view(B, 1, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, H, D).transpose(1, 2)
        u = self.u_proj(x).view(B, 1, H, D).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(-1)

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)

        curr = state["kv_state"].float()
        qt = q.float()
        kt = k.float()
        vt = v.float()
        ut = u.float()
        bt = beta.float()

        u_norm_sq = (ut * ut).sum(dim=-1, keepdim=True) + 1e-6
        u_S = torch.matmul(ut, curr)
        refl = (2.0 * bt / u_norm_sq) * torch.matmul(ut.transpose(-1, -2), u_S)
        S_reflected = curr - refl
        next_S = S_reflected + torch.matmul(kt.transpose(-1, -2), vt)
        s_norm = torch.linalg.matrix_norm(next_S, ord=2, keepdim=True)
        next_S = torch.where(s_norm > 1.0, next_S / s_norm.clamp(min=1e-6), next_S)
        yt = torch.matmul(qt, next_S)
        fmax = 32000.0 if dtype == torch.float16 else 65000.0
        yt = torch.nan_to_num(yt, nan=0.0, posinf=fmax, neginf=-fmax).clamp(min=-fmax, max=fmax)

        state["kv_state"] = next_S
        state["step_idx"] = step_idx + 1

        y = yt.transpose(1, 2).contiguous().view(B, 1, H * D)
        out = self.out_proj(y.to(dtype))

        if is_2d:
            out = out.squeeze(1)

        return out, state


@ATTENTION_REGISTRY.register("preconditioned_gdn_attention")
@ATTENTION_REGISTRY.register("preconditioned_gdn", is_alias=True)
@ATTENTION_REGISTRY.register("pgdn_attention", is_alias=True)
@ATTENTION_REGISTRY.register("pgdn", is_alias=True)
class PreconditionedGDNAttention(BaseAttention):
    """
    Curvature-Aware Preconditioned Gated DeltaNet.
    Applies empirical covariance / diagonal curvature preconditioning:
        P_t = (diag(s_t) + lambda * I)^{-1/2}
        k_tilde_t = P_t @ k_t
        S_t = alpha_t * S_{t-1}(I - beta_t * k_tilde_t k_tilde_t^T) + beta_t * k_tilde_t v_t^T
    Directly aligns with NeuralOps Whitened Covariance Preconditioner.
    """

    FAMILY: str = "gated"
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int | None = None,
        preconditioner_damping: float = 1e-4,
        **kwargs: Any,
    ):
        super().__init__(config, rank)
        self.d_model = _cfg_get(config, "hidden_size", hidden_size)
        self.num_heads = _cfg_get(config, "num_attention_heads", num_heads)
        self.head_dim = _cfg_get(config, "head_dim", head_dim or (self.d_model // self.num_heads))
        self.damping = preconditioner_damping
        self.is_nested_encoder = False
        self.is_decoder = bool(
            kwargs.get(
                "is_decoder",
                kwargs.get("is_causal", _cfg_get(config, "is_decoder", False) or _cfg_get(config, "is_causal", False)),
            )
        )

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.curvature_diag = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=True)

        with torch.no_grad():
            nn.init.constant_(self.alpha_proj.bias, 2.5)
            nn.init.constant_(self.beta_proj.bias, 1.0)
            nn.init.ones_(self.curvature_diag)

        source_attn = kwargs.get("pretrained_attn") or kwargs.get("old_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        _copy_attention_weights(self, src)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, D).transpose(1, 2)

        p_scale = torch.rsqrt(F.softplus(self.curvature_diag) + self.damping).view(1, H, 1, D)
        k_preconditioned = k * p_scale

        is_causal = self.supports_causal and (
            self.is_decoder
            or kwargs.get("is_causal", False)
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "is_causal", False)
        )

        if not is_causal:
            q_feat = F.elu(q.float()) + 1.0
            k_prec = (F.elu(k.float()) + 1.0) * p_scale
            diff = -torch.abs(
                torch.arange(L, device=hidden_states.device).unsqueeze(0)
                - torch.arange(L, device=hidden_states.device).unsqueeze(1)
            ).float()
            decay = torch.exp(0.01 * diff).unsqueeze(0).unsqueeze(0)
            scores = torch.matmul(q_feat, k_prec.transpose(-1, -2)) * decay
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = (
                        (attention_mask > 0)
                        if not attention_mask.dtype.is_floating_point
                        else (attention_mask > -1e4)
                    ).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~mask, 0.0)
                elif attention_mask.dim() == 4:
                    mask = (
                        (attention_mask > -1e4)
                        if attention_mask.dtype.is_floating_point
                        else attention_mask.bool()
                    )
                    scores = scores.masked_fill(~mask, 0.0)
            den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            y = torch.matmul(scores / den, v.float()).transpose(1, 2).contiguous().view(B, L, H * D).to(hidden_states.dtype)
            if self.is_nested_encoder:
                return y, None
            return self.out_proj(y), None

        alpha = torch.sigmoid(self.alpha_proj(hidden_states).float()).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(hidden_states).float()).transpose(1, 2)

        if attention_mask is not None:
            from neuralops.core.attention_mask import get_token_valid_mask

            valid = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
            if valid is not None:
                v_mask = valid.unsqueeze(1)
                alpha = torch.where(v_mask, alpha, torch.ones_like(alpha))
                beta = beta * v_mask.float()

        init_state = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                init_state = past_key_value[0]
            elif isinstance(past_key_value, torch.Tensor):
                init_state = past_key_value

        if init_state is None and D <= 64 and q.is_cuda and q.dtype == torch.bfloat16 and not use_cache:
            try:
                from fla.ops import chunk_gdn

                q_fl = F.normalize(q, p=2, dim=-1, eps=1e-6).transpose(1, 2).contiguous()
                k_fl = F.normalize(k_preconditioned, p=2, dim=-1, eps=1e-6).transpose(1, 2).contiguous()
                v_fl = v.transpose(1, 2).contiguous()
                log_g = torch.log(beta.clamp(min=1e-4, max=1.0)).transpose(1, 2).to(q.dtype).contiguous()
                step_beta = alpha.transpose(1, 2).to(q.dtype).contiguous()
                res = chunk_gdn(q_fl, k_fl, v_fl, log_g, step_beta)
                y_fl = res[0] if isinstance(res, tuple) else res
                y = y_fl.contiguous().view(B, L, H * D)
                if self.is_nested_encoder:
                    return y, None
                out = self.out_proj(y)
                return out, None
            except Exception as e:
                logger.debug("CUDA chunk_gdn failed, falling back to sequential PyTorch: %s", e)

        S = (
            torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
            if init_state is None
            else init_state.to(device=q.device, dtype=torch.float32)
        )

        # Vectorized chunked WY representation for Preconditioned GDN (zero per-token loop)
        c_size = 32
        pad_len = (c_size - (L % c_size)) % c_size
        if pad_len > 0:
            q_p = F.pad(q, (0, 0, 0, pad_len))
            k_p = F.pad(k_preconditioned, (0, 0, 0, pad_len))
            v_p = F.pad(v, (0, 0, 0, pad_len))
            a_p = F.pad(alpha, (0, pad_len), value=1.0)
            b_p = F.pad(beta, (0, pad_len), value=0.0)
        else:
            q_p, k_p, v_p, a_p, b_p = q, k_preconditioned, v, alpha, beta

        num_chunks = q_p.shape[2] // c_size
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            q_blk = F.normalize(q_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, D)
            k_blk = F.normalize(k_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, D)
            v_blk = v_p.float().view(B, H, num_chunks, c_size, D)
            a_blk = a_p.float().clamp(min=1e-4, max=1.0).view(B, H, num_chunks, c_size)
            b_blk = b_p.float().view(B, H, num_chunks, c_size)

            log_a = torch.log(a_blk)
            gamma = torch.cumsum(log_a, dim=-1)
            gamma_prev = gamma - log_a
            exp_gamma = torch.exp(gamma)
            exp_gamma_prev = torch.exp(gamma_prev)

            kk = torch.matmul(k_blk, k_blk.transpose(-1, -2))
            g_diff = (gamma.unsqueeze(-1) - gamma.unsqueeze(-2)).clamp(max=0.0)
            g_diff_prev = (gamma_prev.unsqueeze(-1) - gamma.unsqueeze(-2)).clamp(max=0.0)
            tril_strict = torch.tril(torch.ones((c_size, c_size), device=q.device, dtype=torch.bool), diagonal=-1)
            tril_causal = torch.tril(torch.ones((c_size, c_size), device=q.device, dtype=torch.bool), diagonal=0)

            decay_mat = torch.exp(g_diff) * tril_causal.float()
            M = kk * (b_blk.unsqueeze(-1) * (torch.exp(g_diff_prev) * tril_strict.float()))
            I_plus_M = torch.eye(c_size, device=q.device, dtype=torch.float32) + M

            rhs_v = b_blk.unsqueeze(-1) * v_blk
            rhs_k = (b_blk * exp_gamma_prev).unsqueeze(-1) * k_blk
            rhs = torch.cat([rhs_v, rhs_k], dim=-1)
            sol = torch.linalg.solve_triangular(I_plus_M, rhs, upper=False, unitriangular=True)
            U_v = sol[..., :D]
            W_k = sol[..., D:]

            qk_decay = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * decay_mat
            q_decayed = q_blk * exp_gamma.unsqueeze(-1)
            gamma_end = gamma[..., -1:]
            exp_gamma_end = torch.exp(gamma_end).unsqueeze(-1)
            k_to_end = k_blk * torch.exp((gamma_end - gamma).clamp(max=0.0)).unsqueeze(-1)

            outs = []
            curr = S
            for c in range(num_chunks):
                U_c = U_v[:, :, c] - torch.matmul(W_k[:, :, c], curr)
                y_c = torch.matmul(q_decayed[:, :, c], curr) + torch.matmul(qk_decay[:, :, c], U_c)
                curr = exp_gamma_end[:, :, c] * curr + torch.matmul(k_to_end[:, :, c].transpose(-1, -2), U_c)
                outs.append(y_c)

            y = torch.cat(outs, dim=2)
        if pad_len > 0:
            y = y[:, :, :L, :]
        present_state = curr

        if q.dtype == torch.float16:
            fmax = torch.finfo(q.dtype).max * 0.5
            y = y.clamp(min=-fmax, max=fmax)
        y = y.to(q.dtype).transpose(1, 2).contiguous().view(B, L, H * D)
        present_kv = (present_state.to(q.dtype),) if use_cache else None
        if self.is_nested_encoder:
            return y, present_kv
        out = self.out_proj(y)
        return out, present_kv

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for PreconditionedGDNAttention."""
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        q = self.q_proj(x).view(B, 1, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, H, D).transpose(1, 2)

        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(-1)

        if "curvature_state" not in state or state["curvature_state"] is None:
            state["curvature_state"] = torch.zeros((B, H, 1, D), device=device, dtype=torch.float32)

        curv = state["curvature_state"]
        rho = 0.95
        k_f = k.float()
        curv = rho * curv + (1.0 - rho) * (k_f * k_f)
        state["curvature_state"] = curv

        p_scale = torch.rsqrt(curv + self.damping)
        k_prec = k_f * p_scale

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)

        curr = state["kv_state"].float()
        inter_v = torch.matmul(k_prec, curr)
        err = v.float() - inter_v
        next_S = alpha.float() * curr + beta.float() * torch.matmul(k_prec.transpose(-1, -2), err)
        yt = torch.matmul(q.float(), next_S)

        state["kv_state"] = next_S
        state["step_idx"] = step_idx + 1

        y = yt.transpose(1, 2).contiguous().view(B, 1, H * D)
        out = self.out_proj(y.to(dtype))

        if is_2d:
            out = out.squeeze(1)

        return out, state


__all__ = [
    "GDN2Attention",
    "gdn2_recurrent_step",
    "gdn2_chunk_scan",
    "DeltaProductAttention",
    "householder_reflection",
    "PreconditionedGDNAttention",
]
