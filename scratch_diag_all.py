import torch
from framework.core.triton_kernels.gla_scan import gla_scan_triton
from scratch_test_gla import ref_gla

def rel_norm(a, b):
    return (a - b).norm().item() / b.norm().item()

for (B, H, L, D) in [(2, 4, 32, 16), (2, 4, 64, 32), (2, 8, 128, 64)]:
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
    lam = torch.sigmoid(torch.randn(B, H, L, device='cuda', dtype=torch.float32)).requires_grad_(True)
    y_tri = gla_scan_triton(q, k, v, lam)
    y_ref = ref_gla(q, k, v, lam)
    g = torch.randn_like(y_tri)
    gq_t, gk_t, gv_t, gd_t = torch.autograd.grad(y_tri, (q, k, v, lam), g)
    gq_r, gk_r, gv_r, gd_r = torch.autograd.grad(y_ref, (q, k, v, lam), g)
    print(f"[{B},{H},{L},{D}] Norm rel: fwd={rel_norm(y_tri, y_ref):.2e}, dq={rel_norm(gq_t, gq_r):.2e}, dk={rel_norm(gk_t, gk_r):.2e}, dv={rel_norm(gv_t, gv_r):.2e}, dlam={rel_norm(gd_t, gd_r):.2e}")
