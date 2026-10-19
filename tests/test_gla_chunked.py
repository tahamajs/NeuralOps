# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import torch
import pytest
from neuralops.attentions.linear.gla_attention import GLAttention


def recurrent_gla(q, k, v, log_gate):
    """Naive recurrent GLA — reference for correctness."""
    B, H, N, D = q.shape
    dtype = q.dtype
    S = torch.zeros(B, H, D, D, device=q.device, dtype=dtype)
    outputs = []
    for t in range(N):
        alpha = log_gate[:, :, t].exp()
        S = S * alpha[:, :, None, None]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        S = S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
        o_t = torch.einsum('bhd,bhde->bhe', q[:, :, t], S)
        outputs.append(o_t)
    return torch.stack(outputs, dim=2).to(dtype), S


@pytest.mark.parametrize("B,H,N,D,C", [
    (1, 2, 64, 32, 64),
    (1, 2, 128, 32, 64),
    (1, 2, 256, 32, 64),
    (1, 2, 512, 32, 64),
    (2, 4, 1024, 64, 64),
    (1, 2, 129, 32, 64),
])
def test_chunked_matches_recurrent(B, H, N, D, C):
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    gate_logits = torch.randn(B, H, N, device='cuda', dtype=torch.float32) * 0.5
    log_gate = torch.nn.functional.logsigmoid(gate_logits)

    attn = GLAttention(hidden_size=H*D, num_heads=H, head_dim=D, use_fla=False).cuda()
    out_chunked, state_chunked = attn._chunked_gla_forward(q, k, v, log_gate, C)

    out_rec, state_rec = recurrent_gla(q, k, v, log_gate)

    out_rec_f64, _ = recurrent_gla(q.double(), k.double(), v.double(), log_gate.double())
    out_chunked_f64, _ = attn._chunked_gla_forward(q.double(), k.double(), v.double(), log_gate.double(), C)
    math_diff = (out_chunked_f64 - out_rec_f64).abs().max().item()
    assert math_diff < 1e-10, f"Chunked vs recurrent mathematical mismatch in float64: {math_diff}"

    rel_diff = (out_chunked - out_rec).abs().max().item() / out_rec.abs().max().item()
    assert rel_diff < 1e-3, f"Float32 relative error exceeds bound: {rel_diff}"
    print(f"  B={B} H={H} N={N} D={D} C={C}: math_diff(f64)={math_diff:.2e}, rel_diff(f32)={rel_diff:.2e} ✅")
