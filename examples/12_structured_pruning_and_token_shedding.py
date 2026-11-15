#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 12: Structured Pruning, 2:4 Hardware Sparsity, and Token Shedding.

Subsystem Demonstrated:
    Comprehensive sparsity and structural token reduction techniques:
    1. L1-Norm / Group Magnitude Structured Channel Pruning on intermediate MLP layers
    2. NVIDIA Ampere/Ada 2:4 Structured Hardware Sparsity (Exact 50% block-sparse mask)
    3. Dynamic Token Shedding (Attention score-guided sequence truncation)

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (accelerates sparse tensor GEMMs).
    - Network: Not required; uses offline synthetic tokens and weights.

What You Will Learn:
    1. How structured channel pruning removes intermediate neurons from MLP layers.
    2. How 2:4 sparse masks guarantee exactly 2 zeros per 4 contiguous elements for Tensor Cores.
    3. How token shedding inspects cross-attention distributions to drop redundant sequence tokens.
    4. How to measure parameter reduction, sparsity percentage, and sequence speedup.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    synthetic_tokens,
    table,
)


def apply_24_sparsity(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Enforce exact NVIDIA 2:4 structured sparsity along rows:

    For every 4 contiguous elements in each row, exactly the 2 elements with the
    smallest magnitude are pruned to zero.

    Returns:
        (sparse_weight, boolean_mask, sparsity_ratio)
    """
    rows, cols = weight.shape
    assert cols % 4 == 0, f"Columns {cols} must be divisible by 4 for 2:4 sparsity."

    w_blocks = weight.view(rows, cols // 4, 4)
    # Find top-2 magnitudes in each 4-element group
    top2_idx = torch.topk(w_blocks.abs(), k=2, dim=-1).indices

    mask_blocks = torch.zeros_like(w_blocks, dtype=torch.bool)
    mask_blocks.scatter_(-1, top2_idx, True)

    w_sparse = (w_blocks * mask_blocks).view(rows, cols)
    mask = mask_blocks.view(rows, cols)

    actual_sparsity = (w_sparse == 0).float().mean().item()
    return w_sparse, mask, actual_sparsity


def prune_mlp_intermediate(
    lin1: nn.Linear,
    lin2: nn.Linear,
    prune_ratio: float = 0.5,
) -> tuple[nn.Linear, nn.Linear, int, int]:
    """L1 / L2 Structured Channel Pruning on an MLP block:

    Given Lin1: d_in -> d_mid and Lin2: d_mid -> d_out,
    channel importance is calculated using group norm:
    s_j = ||W1[j, :]||_2 * ||W2[:, j]||_2.
    Prune lowest s_j channels and resize weights in-place.
    """
    d_mid = lin1.out_features
    keep_count = max(4, int(d_mid * (1.0 - prune_ratio)))

    # Compute group norm importance per intermediate channel
    w1 = lin1.weight.data
    w2 = lin2.weight.data
    norm1 = torch.norm(w1, p=2, dim=1)  # (d_mid,)
    norm2 = torch.norm(w2, p=2, dim=0)  # (d_mid,)
    scores = norm1 * norm2

    # Retain top-k channels
    retained_idx = torch.topk(scores, k=keep_count).indices.sort().values

    device = lin1.weight.device
    # Create resized linear modules on same device
    new_lin1 = nn.Linear(lin1.in_features, keep_count, bias=(lin1.bias is not None), device=device)
    new_lin2 = nn.Linear(keep_count, lin2.out_features, bias=(lin2.bias is not None), device=device)

    new_lin1.weight.data.copy_(w1[retained_idx, :])
    if lin1.bias is not None:
        new_lin1.bias.data.copy_(lin1.bias.data[retained_idx])

    new_lin2.weight.data.copy_(w2[:, retained_idx])
    if lin2.bias is not None:
        new_lin2.bias.data.copy_(lin2.bias.data)

    old_params = w1.numel() + w2.numel()
    new_params = new_lin1.weight.numel() + new_lin2.weight.numel()
    return new_lin1, new_lin2, old_params, new_params


class TokenSheddingWrapper(nn.Module):
    """Dynamic Token Shedding wrapper:

    Scores tokens by attention saliency (average attention received by each key token).
    Drops the lowest scoring tokens while preserving special sink tokens (position 0).
    """

    def __init__(self, keep_ratio: float = 0.5) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Args:

            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)

        Returns:
            (shortened_hidden_states, retained_indices, length_reduction_pct)
        """
        bsz, seq_len, dim = hidden_states.shape
        keep_tokens = max(2, int(seq_len * self.keep_ratio))

        # Compute token importance: sum attention received across queries and heads
        # (bsz, seq_len)
        saliency = attention_weights.sum(dim=(1, 2))

        # Always protect position 0 (CLS / prompt sink token)
        saliency[:, 0] = float("inf")

        # Select top tokens per sequence
        topk_idx = torch.topk(saliency, k=keep_tokens, dim=-1).indices.sort(dim=-1).values

        # Gather retained tokens
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
        shortened_states = torch.gather(hidden_states, dim=1, index=idx_expanded)

        reduction_pct = (1.0 - keep_tokens / seq_len) * 100.0
        return shortened_states, topk_idx, reduction_pct


def main() -> int:
    section("NeuralOps Example 12: Structured Pruning & Token Shedding")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    summary_rows: list[dict[str, str]] = []

    # Initialize model to obtain real architecture weights and forward activations
    base_model = TinyClassifier().to(device)
    base_model.eval()

    # =========================================================================
    # Part 1: NVIDIA 2:4 Structured Hardware Sparsity
    # =========================================================================
    section("Part 1: NVIDIA 2:4 Structured Hardware Sparsity")
    # Extract real weight matrix from TinyClassifier layer 0 attention projection
    weight_dense = base_model.transformer.distilbert.transformer.layer[0].attention.q_lin.weight.detach().cpu().clone()
    m, n = weight_dense.shape  # 32 x 32 (8 groups of 4 per row)
    w_sparse, mask_24, sparsity = apply_24_sparsity(weight_dense)

    # Invariant: exactly 2 zeros in every 4-element block along each row
    zeros_per_block = (w_sparse.view(m, n // 4, 4) == 0).sum(dim=-1)
    is_exact_24 = bool((zeros_per_block == 2).all().item())
    is_50pct = bool(abs(sparsity - 0.50) < 1e-6)
    p_sparsity_ok = bool(is_exact_24 and is_50pct)

    pass_fail("Exact 2:4 Block Pattern Check", is_exact_24)
    pass_fail("Overall 50.0% Sparsity Verification", is_50pct)
    kv("Weight Matrix Source", "TinyClassifier layer[0].attention.q_lin.weight")
    kv("Matrix Dimensions", f"{m} x {n} ({weight_dense.numel():,} elements)")
    kv("Sparsity Ratio", f"{sparsity * 100:.1f}%")
    kv("Zero Elements Count", f"{(w_sparse == 0).sum().item():,} / {w_sparse.numel():,}")

    summary_rows.append({
        "Technique": "NVIDIA 2:4 Structured Sparsity",
        "Target": f"Weight Matrix ({m}x{n})",
        "Metric": "50.0% Sparsity (2 of 4 zero)",
        "Status": "PASS" if p_sparsity_ok else "FAIL",
    })

    # =========================================================================
    # Part 2: Structured MLP Channel Pruning
    # =========================================================================
    section("Part 2: Structured MLP Channel Pruning")
    # Use real FFN block layers from TinyClassifier
    lin1 = base_model.transformer.distilbert.transformer.layer[0].ffn.lin1
    lin2 = base_model.transformer.distilbert.transformer.layer[0].ffn.lin2
    d_in = lin1.in_features
    d_mid = lin1.out_features

    # Forward check with real synthetic embeddings
    tokens = synthetic_tokens(batch_size=4, seq_len=16, device=device)
    word_emb = base_model.transformer.distilbert.embeddings.word_embeddings(tokens["input_ids"])
    # Take token vectors at position 0: shape (4, d_in)
    x_input = word_emb[:, 0, :].detach()

    with torch.no_grad():
        out_before = lin2(torch.relu(lin1(x_input)))

    prune_ratio = 0.50
    new_lin1, new_lin2, p_old, p_new = prune_mlp_intermediate(lin1, lin2, prune_ratio=prune_ratio)
    with torch.no_grad():
        out_after = new_lin2(torch.relu(new_lin1(x_input)))

    param_reduction = (1.0 - p_new / p_old) * 100.0
    dim_ok = bool(new_lin1.out_features == int(d_mid * (1 - prune_ratio)))
    finite_ok = bool(torch.isfinite(out_after).all().item())
    p_prune_ok = bool(dim_ok and finite_ok)

    pass_fail("MLP Intermediate Dimension Pruning", dim_ok)
    pass_fail("Post-Pruning Forward Finite Check", finite_ok)
    kv("Original MLP Parameters", f"{p_old:,}")
    kv("Pruned MLP Parameters", f"{p_new:,} ({param_reduction:.1f}% reduction)")
    kv("Intermediate Width", f"{d_mid} -> {new_lin1.out_features}")

    summary_rows.append({
        "Technique": "Structured Channel Pruning",
        "Target": f"MLP Intermediate ({d_mid} -> {new_lin1.out_features})",
        "Metric": f"{param_reduction:.1f}% Param Reduction",
        "Status": "PASS" if p_prune_ok else "FAIL",
    })

    # =========================================================================
    # Part 3: Dynamic Token Shedding
    # =========================================================================
    section("Part 3: Dynamic Token Shedding")
    batch_size, seq_len = 2, 16
    tokens = synthetic_tokens(batch_size=batch_size, seq_len=seq_len, device=device)
    base_model.transformer.config._attn_implementation = "eager"

    with torch.no_grad():
        out = base_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states[-1]
        attn_weights = out.attentions[-1]

    shedder = TokenSheddingWrapper(keep_ratio=0.50)
    shortened, retained_idx, red_pct = shedder(hidden_states, attn_weights)

    target_tokens = int(seq_len * 0.50)
    len_ok = bool(shortened.size(1) == target_tokens)
    sink_ok = bool((retained_idx[:, 0] == 0).all().item())
    p_shed_ok = bool(len_ok and sink_ok)

    pass_fail(f"Sequence Length Shortening ({seq_len} -> {target_tokens} tokens)", len_ok)
    pass_fail("Sink Token Protection (Token 0 retained)", sink_ok)
    kv("Initial Sequence Length", seq_len)
    kv("Shedded Sequence Length", shortened.size(1))
    kv("Sequence Reduction", f"{red_pct:.1f}%")

    summary_rows.append({
        "Technique": "Attention Token Shedding",
        "Target": f"Sequence (L={seq_len} -> {shortened.size(1)})",
        "Metric": f"{red_pct:.1f}% Token Reduction",
        "Status": "PASS" if p_shed_ok else "FAIL",
    })

    # =========================================================================
    # Summary Table
    # =========================================================================
    section("Sparsity & Token Shedding Techniques Summary")
    headers = ["Technique", "Target", "Metric", "Status"]
    rows = [[str(r[h]) for h in headers] for r in summary_rows]
    table(rows, headers=headers)

    all_passed = bool(p_sparsity_ok and p_prune_ok and p_shed_ok)
    pass_fail(
        "Overall Example 12 Status",
        all_passed,
        "Structured pruning, 2:4 sparsity, and token shedding verified"
        if all_passed
        else "Sub-check failure detected in pruning or shedding",
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
