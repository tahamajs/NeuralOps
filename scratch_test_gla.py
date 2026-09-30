import torch
from framework.core.triton_kernels.gla_scan import gla_scan_triton

def ref_gla(q, k, v, lam):
    B, H, L, D = q.shape
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    lam_f = lam.float().clamp(min=1e-4, max=1.0 - 1e-4)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    Z = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        q_t = q_f[:, :, t]
        k_t = k_f[:, :, t]
        v_t = v_f[:, :, t]
        lam_t = lam_f[:, :, t]
        S = lam_t.view(B, H, 1, 1) * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
        Z = lam_t.view(B, H, 1) * Z + k_t
        num = torch.einsum('bhd,bhde->bhe', q_t, S)
        den = torch.einsum('bhd,bhd->bh', q_t, Z).clamp_min(1e-4)
        ys.append(num / den.unsqueeze(-1))
    return torch.stack(ys, dim=2)

def rel_err(a, b, eps=1e-5):
    return ((a - b).abs() / (b.abs() + eps)).max().item()

for (B, H, L, D) in [(2, 4, 32, 16), (2, 4, 64, 32), (2, 8, 128, 64)]:
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
    lam = torch.sigmoid(torch.randn(B, H, L, device='cuda')).requires_grad_(True)
    y_tri = gla_scan_triton(q, k, v, lam)
    y_ref = ref_gla(q, k, v, lam)
    fwd_err = rel_err(y_tri, y_ref)
    g = torch.randn_like(y_tri)
    gq_t, gk_t, gv_t, gd_t = torch.autograd.grad(y_tri, (q, k, v, lam), g)
    gq_r, gk_r, gv_r, gd_r = torch.autograd.grad(y_ref, (q, k, v, lam), g)
    print(f'[{B},{H},{L},{D}] fwd: {fwd_err:.2e}, dq: {rel_err(gq_t, gq_r):.2e}, dk: {rel_err(gk_t, gk_r):.2e}, dv: {rel_err(gv_t, gv_r):.2e}, dlam: {rel_err(gd_t, gd_r):.2e}')
