#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 16: Distributed Parallelism Simulation (Context, Tensor, and Layer Parallelism).

Subsystem Demonstrated:
    Deterministic offline mathematical simulation of 3 core distributed paradigms:
    1. Context Parallelism (Ring Attention / Sequence Chunking with Online Softmax Correction)
       - Splits sequence L into 2 chunks across virtual ranks
       - Computes local attention blocks with running max/sum normalization
       - Reconstructs exact full sequence attention output
    2. Tensor Parallelism (Megatron-LM Column-Row Split MLP)
       - Splits W1 column-wise and W2 row-wise across 2 virtual ranks
       - Computes partial activations and performs All-Reduce (Sum)
       - Verifies parity against single-device dense MLP
    3. Layer / Pipeline Parallelism
       - Partitions multi-layer model across sequential virtual stages
       - Propagates boundary activations between stages
       - Verifies 100% exact parity against monolithic end-to-end forward pass

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (distributed tests execute on CPU virtual devices).
    - Network: Not required; uses synthetic offline tokens and tensors.

What You Will Learn:
    1. How online softmax scaling enables mathematically lossless context parallelism.
    2. How column and row parallel linear layers compose without intermediate communication.
    3. How layer pipeline boundaries partition compute without altering logits.
    4. How to verify numerical parity (atol=1e-4) across simulated parallel ranks.
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
import torch.nn.functional as F

from examples._shared import (
    TinyClassifier,
    assert_parity,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
    tiny_inputs,
)


def simulate_context_parallel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Simulate 2-way Context Parallelism (Ring Attention style with online softmax).

    Args:
        q, k, v: [B, L, D] query, key, value tensors.

    Returns:
        (y_single, y_cp, max_abs_diff)
    """
    _bsz, seq_len, dim = q.shape
    scale = 1.0 / (dim**0.5)

    # 1. Monolithic Single-Device Full Attention
    scores_dense = (q @ k.transpose(-1, -2)) * scale
    probs_dense = F.softmax(scores_dense, dim=-1)
    y_single = probs_dense @ v

    # 2. Simulated 2-Way Sequence Chunking
    half = seq_len // 2
    q1, q2 = q[:, :half, :], q[:, half:, :]
    k1, k2 = k[:, :half, :], k[:, half:, :]
    v1, v2 = v[:, :half, :], v[:, half:, :]

    # Chunk 1: queries q1 against (k1, v1) and (k2, v2)
    s11 = (q1 @ k1.transpose(-1, -2)) * scale
    s12 = (q1 @ k2.transpose(-1, -2)) * scale
    m1 = torch.maximum(
        s11.max(dim=-1, keepdim=True).values,
        s12.max(dim=-1, keepdim=True).values,
    )
    p11 = torch.exp(s11 - m1)
    p12 = torch.exp(s12 - m1)
    l1 = p11.sum(dim=-1, keepdim=True) + p12.sum(dim=-1, keepdim=True)
    y1 = (p11 @ v1 + p12 @ v2) / l1

    # Chunk 2: queries q2 against (k1, v1) and (k2, v2)
    s21 = (q2 @ k1.transpose(-1, -2)) * scale
    s22 = (q2 @ k2.transpose(-1, -2)) * scale
    m2 = torch.maximum(
        s21.max(dim=-1, keepdim=True).values,
        s22.max(dim=-1, keepdim=True).values,
    )
    p21 = torch.exp(s21 - m2)
    p22 = torch.exp(s22 - m2)
    l2 = p21.sum(dim=-1, keepdim=True) + p22.sum(dim=-1, keepdim=True)
    y2 = (p21 @ v1 + p22 @ v2) / l2

    y_cp = torch.cat([y1, y2], dim=1)
    diff = (y_single - y_cp).abs().max().item()
    return y_single, y_cp, diff


def simulate_tensor_parallel_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Simulate Megatron-LM style Tensor Parallelism on an MLP block.

    Args:
        x: [B, D_in] input activations.
        w1: [D_mid, D_in] column-parallel weights.
        w2: [D_out, D_mid] row-parallel weights.

    Returns:
        (y_dense, y_tp, max_abs_diff)
    """
    # 1. Dense Monolithic Execution
    h_dense = F.gelu(x @ w1.T)
    y_dense = h_dense @ w2.T

    # 2. 2-Way Tensor Parallel Split
    # W1 column split along out_features (dim 0)
    w1_1, w1_2 = w1.chunk(2, dim=0)
    # W2 row split along in_features (dim 1)
    w2_1, w2_2 = w2.chunk(2, dim=1)

    # Rank 0 local compute
    h1 = F.gelu(x @ w1_1.T)
    y1 = h1 @ w2_1.T

    # Rank 1 local compute
    h2 = F.gelu(x @ w1_2.T)
    y2 = h2 @ w2_2.T

    # All-Reduce (Sum)
    y_tp = y1 + y2
    diff = (y_dense - y_tp).abs().max().item()
    return y_dense, y_tp, diff


def simulate_layer_parallel_transformer(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Simulate 2-Stage Pipeline / Layer Parallelism across virtual ranks."""
    model.eval()
    with torch.no_grad():
        out_single = model(input_ids=input_ids)
        y_single = out_single.logits if hasattr(out_single, "logits") else out_single

        # Virtual Rank 0: Embeddings + Layer 0
        distil = model.transformer.distilbert
        emb = distil.embeddings(input_ids)
        h0 = distil.transformer.layer[0](emb)

        # Activations transferred across virtual network boundary to Rank 1
        # Virtual Rank 1: Layer 1 + Classification Head
        h1 = distil.transformer.layer[1](h0)
        pooled = h1[:, 0]
        pooled = model.transformer.pre_classifier(pooled)
        pooled = nn.ReLU()(pooled)
        y_pipeline = model.transformer.classifier(pooled)

        diff = (y_single - y_pipeline).abs().max().item()
        return y_single, y_pipeline, diff


def main() -> int:
    section("NeuralOps Example 16: Distributed Parallelism Simulation")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    torch.manual_seed(42)
    summary_records: list[dict[str, str]] = []

    # =========================================================================
    # Paradigm 1: Context Parallelism (Ring Attention / Online Softmax)
    # =========================================================================
    section("Paradigm 1: Context Parallelism (Ring Attention)")
    b, seq_len, d_head = 2, 16, 32
    q = torch.randn(b, seq_len, d_head)
    k = torch.randn(b, seq_len, d_head)
    v = torch.randn(b, seq_len, d_head)

    y_single_cp, y_cp, diff_cp = simulate_context_parallel_attention(q, k, v)
    assert_parity(y_single_cp, y_cp, atol=1e-4, label="Context Parallelism Numerical Parity")
    kv("Sequence Length", f"{seq_len} tokens (split into 2 chunks of {seq_len // 2})")
    kv("Max Absolute Difference", f"{diff_cp:.2e}")
    summary_records.append({
        "Paradigm": "Context Parallelism (Ring)",
        "Partition Dimension": "Sequence (dim 1)",
        "Max Absolute Diff": f"{diff_cp:.2e}",
        "Parity Status": "PASS (atol < 1e-4)",
    })

    # =========================================================================
    # Paradigm 2: Tensor Parallelism (Megatron Column-Row MLP)
    # =========================================================================
    section("Paradigm 2: Tensor Parallelism (Megatron Column-Row)")
    batch_size, d_in, d_mid, d_out = 4, 32, 64, 32
    x_in = torch.randn(batch_size, d_in)
    w1_dense = torch.randn(d_mid, d_in)
    w2_dense = torch.randn(d_out, d_mid)

    y_dense_tp, y_tp, diff_tp = simulate_tensor_parallel_mlp(x_in, w1_dense, w2_dense)
    assert_parity(y_dense_tp, y_tp, atol=1e-4, label="Tensor Parallelism Numerical Parity")
    kv("Linear Weight Shapes", f"W1: {d_mid}x{d_in}, W2: {d_out}x{d_mid}")
    kv("Max Absolute Difference", f"{diff_tp:.2e}")
    summary_records.append({
        "Paradigm": "Tensor Parallelism (Megatron)",
        "Partition Dimension": "Hidden / Channels (dim 0/1)",
        "Max Absolute Diff": f"{diff_tp:.2e}",
        "Parity Status": "PASS (atol < 1e-4)",
    })

    # =========================================================================
    # Paradigm 3: Layer / Pipeline Parallelism
    # =========================================================================
    section("Paradigm 3: Layer / Pipeline Parallelism")
    model = TinyClassifier()
    model.eval()
    inp = tiny_inputs()
    input_ids = inp["input_ids"]

    y_single_lp, y_lp, diff_lp = simulate_layer_parallel_transformer(model, input_ids)
    assert_parity(y_single_lp, y_lp, atol=1e-4, label="Layer Parallelism Numerical Parity")
    kv("Model Architecture", "TinyClassifier (2 Transformer Blocks)")
    kv("Pipeline Partition", "Rank 0: Layer 0, Rank 1: Layer 1 + Head")
    kv("Max Absolute Difference", f"{diff_lp:.2e}")
    summary_records.append({
        "Paradigm": "Layer / Pipeline Parallelism",
        "Partition Dimension": "Layers / Depth (Stages 0/1)",
        "Max Absolute Diff": f"{diff_lp:.2e}",
        "Parity Status": "PASS (atol < 1e-4)",
    })

    # =========================================================================
    # Summary Table
    # =========================================================================
    section("Distributed Parallelism Simulation Summary")
    headers = ["Paradigm", "Partition Dimension", "Max Absolute Diff", "Parity Status"]
    rows = [[str(r[h]) for h in headers] for r in summary_records]
    table(rows, headers=headers)

    pass_fail(
        "Overall Example 16 Status",
        True,
        "Context, Tensor, and Layer parallelism simulations mathematically verified with parity atol=1e-4",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
