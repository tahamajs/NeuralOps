# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity tests for MoE Routing dispatch and gather (Module 5).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.moe_routing import (
    moe_dispatch_gather,
    moe_route_pytorch_ref,
)


def test_moe_routing_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MoE Routing test")

    torch.manual_seed(42)
    num_tokens = 64
    hidden_dim = 128
    num_experts = 8
    top_k = 2

    tokens = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)
    gate_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)

    def dummy_expert(x):
        return x * 2.0

    out_cuda = moe_dispatch_gather(tokens, gate_logits, dummy_expert, top_k=top_k)

    routing_weights, topk_indices, expert_offsets = moe_route_pytorch_ref(tokens, gate_logits, top_k=top_k)
    dispatched = tokens.repeat_interleave(top_k, dim=0)
    ref_expert_out = dummy_expert(dispatched)
    out_ref = (ref_expert_out.view(num_tokens, top_k, hidden_dim) * routing_weights.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
