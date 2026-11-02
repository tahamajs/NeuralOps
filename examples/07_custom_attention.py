#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 07: Custom Attention and Custom Compressor Registration.

Subsystem Demonstrated:
    `@neuralops.register_custom_attention` and `COMPRESSOR_REGISTRY.register`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Implementing and registering an authentic causal linear attention operator:
       - Mathematical formulation: O_t = phi(Q_t) @ sum_{i=0}^t (phi(K_i)^T @ V_i)
       - Feature activation map: phi(x) = elu(x) + 1.0
       - Autoregressive recurrent state update: S_t = S_{t-1} + phi(K_t)^T @ V_t
    2. Verifying strict temporal causality (future perturbations do not alter past tokens).
    3. Registering a custom low-rank SVD compression engine via the compressor registry.
    4. Transplanting the custom attention module into a transformer block in-place.
    5. Validating dynamic registry discovery via `neuralops.list_attentions()`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralops
from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    tiny_inputs,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


# =====================================================================
# 1. Custom Causal Linear Attention Module
# =====================================================================

@neuralops.register_custom_attention("my_causal_linear")
class MyCausalLinearAttention(nn.Module):
    """
    User-defined causal linear attention operator.

    Uses kernel feature mapping phi(x) = elu(x) + 1 and associative cumulative summation.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = max(embed_dim // num_heads, 1)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        past_state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, d = hidden_states.shape
        h = self.num_heads
        dh = self.head_dim

        q = self.q_proj(hidden_states).view(b, t, h, dh).transpose(1, 2)  # (B, H, T, dh)
        k = self.k_proj(hidden_states).view(b, t, h, dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, h, dh).transpose(1, 2)

        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        if is_causal:
            # Sequential recurrence: S_t = S_{t-1} + k_t^T @ v_t
            S = past_state.clone() if past_state is not None else torch.zeros(b, h, dh, dh, device=hidden_states.device, dtype=hidden_states.dtype)
            outputs: list[torch.Tensor] = []
            for step in range(t):
                k_t = phi_k[:, :, step : step + 1, :]  # (B, H, 1, dh)
                v_t = v[:, :, step : step + 1, :]      # (B, H, 1, dh)
                q_t = phi_q[:, :, step : step + 1, :]  # (B, H, 1, dh)

                S = S + (k_t.transpose(-1, -2) @ v_t)
                o_t = q_t @ S                          # (B, H, 1, dh)
                outputs.append(o_t)

            out = torch.cat(outputs, dim=2)
            final_state = S
        else:
            # Full sequence parallel contraction
            S = phi_k.transpose(-1, -2) @ v
            out = phi_q @ S
            final_state = S

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        return out, final_state


# =====================================================================
# 2. Custom Compressor Registration
# =====================================================================

@COMPRESSOR_REGISTRY.register("my_svd_variant")
class MySVDVariantCompressor(nn.Module):
    """Custom low-rank factorization operator."""

    def __init__(self, rank: int = 8, **kwargs: Any) -> None:
        super().__init__()
        self.rank = rank

    def factorize_layer(self, layer: nn.Linear, target_rank: int | None = None) -> tuple[nn.Linear, nn.Linear]:
        r = target_rank or self.rank
        w = layer.weight.data.float()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        r = min(r, len(s))

        u_r = u[:, :r] * s[:r].unsqueeze(0)
        vh_r = vh[:r, :]

        l1 = nn.Linear(vh_r.shape[1], r, bias=False)
        l2 = nn.Linear(r, u_r.shape[0], bias=layer.bias is not None)

        l1.weight.data.copy_(vh_r)
        l2.weight.data.copy_(u_r)
        if layer.bias is not None:
            l2.bias.data.copy_(layer.bias.data)

        return l1, l2


# =====================================================================
# Verification Steps
# =====================================================================

def verify_causality() -> bool:
    """Verify that perturbing future token input[t+5] does not change output[t]."""
    section("Part 1: Strict Temporal Causality Verification")

    module = MyCausalLinearAttention(embed_dim=32, num_heads=2).eval()
    x1 = torch.randn(2, 12, 32)
    x2 = x1.clone()

    target_t = 3
    future_t = target_t + 5
    x2[:, future_t, :] += 10.0  # Significant perturbation at t=8

    with torch.no_grad():
        out1, _ = module(x1, is_causal=True)
        out2, _ = module(x2, is_causal=True)

    diff_past = (out1[:, target_t, :] - out2[:, target_t, :]).abs().max().item()
    diff_future = (out1[:, future_t, :] - out2[:, future_t, :]).abs().max().item()

    is_strictly_causal = diff_past == 0.0 and diff_future > 0.0
    pass_fail(
        "Causal Invariant: Past Token Independence",
        is_strictly_causal,
        f"diff@t={target_t}: {diff_past:.6e}, diff@t={future_t}: {diff_future:.4f}",
    )
    return is_strictly_causal


def verify_surgery_and_registry(device: str) -> bool:
    """Verify transplantation of custom attention and compressor registration."""
    section("Part 2: In-Place Surgery with Custom Attention")

    # 1. Verify custom attention is in registry
    registered_attns = neuralops.list_attentions()
    attn_in_reg = "my_causal_linear" in registered_attns
    pass_fail("Custom Attention Registration Discovery", attn_in_reg, "'my_causal_linear' found in registry")

    # 2. Transplant into tiny model
    model = TinyClassifier().to(device)
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}

    neuralops.replace_attention(model, attention="my_causal_linear", copy_weights=False)

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out

    is_finite = bool(torch.isfinite(logits).all().item())
    pass_fail("Custom Attention In-Place Model Forward Pass", is_finite, f"logits shape={tuple(logits.shape)}")

    # 3. Verify custom compressor
    section("Part 3: Custom Compressor Registration & Parameter Reduction")
    registered_comps = list(COMPRESSOR_REGISTRY.keys())
    comp_in_reg = "my_svd_variant" in registered_comps
    pass_fail("Custom Compressor Registration Discovery", comp_in_reg, "'my_svd_variant' found in registry")

    # Test parameter reduction on a linear projection
    original_layer = nn.Linear(64, 32, bias=True)
    orig_params = sum(p.numel() for p in original_layer.parameters())

    compressor = MySVDVariantCompressor(rank=8)
    l1, l2 = compressor.factorize_layer(original_layer, target_rank=8)
    factored_params = sum(p.numel() for p in l1.parameters()) + sum(p.numel() for p in l2.parameters())

    reduction_pct = ((orig_params - factored_params) / orig_params) * 100.0
    reduced = factored_params < orig_params
    pass_fail(
        "Custom Compressor Parameter Reduction",
        reduced,
        f"{orig_params} -> {factored_params} params ({reduction_pct:.1f}% reduction)",
    )

    return bool(attn_in_reg and is_finite and comp_in_reg and reduced)


def main() -> int:
    section("NeuralOps Example 07: Custom Attention & Compressor Registration")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    p1 = verify_causality()
    p2 = verify_surgery_and_registry(backend)

    success = p1 and p2
    print()
    if success:
        pass_fail("Overall Example 07 Status", True, "Custom attention and compressor fully operational")
        return 0
    else:
        pass_fail("Overall Example 07 Status", False, "One or more custom registration checks failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
