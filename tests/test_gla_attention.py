import torch
from neuralops.attentions.linear.gla_attention import GLAttention


def test_forward_backward():
    torch.manual_seed(42)
    device = "cuda"
    B, N, D = 2, 128, 256
    H, d = 8, 32

    attn = GLAttention(hidden_size=D, num_heads=H, head_dim=d).to(device)
    x = torch.randn(B, N, D, device=device)

    out, _ = attn(x)
    assert out.shape == (B, N, D), f"shape {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected"
    assert not torch.isinf(out).any(), "Inf detected"

    loss = out.sum()
    loss.backward()

    for name, p in attn.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"

    print("✅ forward + backward OK")
    print(f"   output shape: {out.shape}")
    print(f"   FLA used: {attn._fla_kernel is not None}")


if __name__ == "__main__":
    test_forward_backward()
