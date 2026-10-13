import torch
import pytest
from neuralops.attentions.linear.gla_attention import GLAttention


def recurrent_gla(q, k, v, log_gate):
    """Naive recurrent GLA — reference for correctness."""
    B, H, N, D = q.shape
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    outputs = []
    for t in range(N):
        alpha = log_gate[:, :, t].exp()  # (B, H)
        S = S * alpha[:, :, None, None]  # decay
        k_t = k[:, :, t]  # (B, H, D)
        v_t = v[:, :, t]
        S = S + torch.einsum('bhd,bhe->bhde', k_t.float(), v_t.float())
        o_t = torch.einsum('bhd,bhde->bhe', q[:, :, t].float(), S)
        outputs.append(o_t)
    return torch.stack(outputs, dim=2).to(q.dtype), S


@pytest.mark.parametrize("B,H,N,D,C", [
    (1, 2, 64, 32, 64),
    (1, 2, 128, 32, 64),
    (1, 2, 256, 32, 64),
    (1, 2, 512, 32, 64),
    (2, 4, 1024, 64, 64),
    (1, 2, 129, 32, 64),  # non-divisible
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

    diff = (out_chunked - out_rec).abs().max().item()
    assert diff < 1e-4, f"Chunked vs recurrent diff: {diff}"
    print(f"  B={B} H={H} N={N} D={D} C={C}: max_diff={diff:.2e} ✅")
