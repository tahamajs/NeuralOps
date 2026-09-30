import torch
import torch.nn.functional as F

# Triton imports
from framework.core.triton_kernels.fms_scan import fms_scan_triton
from framework.core.triton_kernels.gla_scan import gla_scan_triton
from framework.core.triton_kernels.selective_scan import selective_scan_fn
from framework.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
from framework.core.triton_kernels.gdn_scan import gdn_scan_triton

def rel_norm(a, b):
    b_n = b.norm().item()
    if b_n < 1e-12:
        return (a - b).abs().max().item()
    return (a - b).norm().item() / b_n

def max_rel(a, b, eps=1e-5):
    return ((a - b).abs() / (b.abs() + eps)).max().item()

# ==============================================================================
# 1. FMS TEST
# ==============================================================================
def ref_fms(q, k, v, decay):
    B, H, L, D = q.shape
    S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    Z = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    ys = []
    for t in range(L):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        q_f = F.elu(q_t) + 1.0
        k_f = (F.elu(k_t) + 1.0) / (D ** 0.5)
        S = decay.view(1, H, 1, 1) * S + torch.einsum('bhd,bhe->bhde', k_f, v_t)
        Z = decay.view(1, H, 1) * Z + k_f
        num = torch.einsum('bhd,bhde->bhe', q_f, S)
        den = torch.einsum('bhd,bhd->bh', q_f, Z).clamp_min(1e-5)
        y_t = num / den.unsqueeze(-1)
        ys.append(y_t)
    return torch.stack(ys, dim=2)

def test_fms():
    print("Testing 1/5: FMS (factorised_matrix_state)...")
    max_fwd, max_dq, max_dk, max_dv, max_dd = 0.0, 0.0, 0.0, 0.0, 0.0
    for (B, H, L, D) in [(2, 4, 32, 16), (2, 4, 64, 32), (2, 8, 128, 64)]:
        torch.manual_seed(42)
        q = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
        k = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
        v = torch.randn(B, H, L, D, device='cuda', requires_grad=True)
        decay = torch.sigmoid(torch.randn(H, device='cuda')).requires_grad_(True)

        y_tri = fms_scan_triton(q, k, v, decay)
        y_ref = ref_fms(q, k, v, decay)
        g = torch.randn_like(y_tri)

        gq_t, gk_t, gv_t, gd_t = torch.autograd.grad(y_tri, (q, k, v, decay), g)
        gq_r, gk_r, gv_r, gd_r = torch.autograd.grad(y_ref, (q, k, v, decay), g)

        max_fwd = max(max_fwd, rel_norm(y_tri, y_ref))
        max_dq = max(max_dq, rel_norm(gq_t, gq_r))
        max_dk = max(max_dk, rel_norm(gk_t, gk_r))
        max_dv = max(max_dv, rel_norm(gv_t, gv_r))
        max_dd = max(max_dd, rel_norm(gd_t, gd_r))

    status = "PASS" if max(max_fwd, max_dq, max_dk, max_dv, max_dd) < 1e-3 else "FAIL"
    return {"name": "FMS", "fwd": max_fwd, "dq": max_dq, "dk": max_dk, "dv": max_dv, "dparam": max_dd, "status": status}

# ==============================================================================
# 2. GLA TEST
# ==============================================================================
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

def test_gla():
    print("Testing 2/5: GLA (gated_linear_attention)...")
    max_fwd, max_dq, max_dk, max_dv, max_dl = 0.0, 0.0, 0.0, 0.0, 0.0
    for (B, H, L, D) in [(2, 4, 32, 16), (2, 4, 64, 32), (2, 8, 128, 64)]:
        torch.manual_seed(42)
        q = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
        lam = torch.sigmoid(torch.randn(B, H, L, device='cuda', dtype=torch.float32)).requires_grad_(True)

        y_tri = gla_scan_triton(q, k, v, lam)
        y_ref = ref_gla(q, k, v, lam)
        g = torch.randn_like(y_tri)

        gq_t, gk_t, gv_t, gl_t = torch.autograd.grad(y_tri, (q, k, v, lam), g)
        gq_r, gk_r, gv_r, gl_r = torch.autograd.grad(y_ref, (q, k, v, lam), g)

        max_fwd = max(max_fwd, rel_norm(y_tri, y_ref))
        max_dq = max(max_dq, rel_norm(gq_t, gq_r))
        max_dk = max(max_dk, rel_norm(gk_t, gk_r))
        max_dv = max(max_dv, rel_norm(gv_t, gv_r))
        max_dl = max(max_dl, rel_norm(gl_t, gl_r))

    status = "PASS" if max(max_fwd, max_dq, max_dk, max_dv, max_dl) < 1e-3 else "FAIL"
    return {"name": "GLA", "fwd": max_fwd, "dq": max_dq, "dk": max_dk, "dv": max_dv, "dparam": max_dl, "status": status}

# ==============================================================================
# 3. MAMBA TEST
# ==============================================================================
def ref_selective_scan(u, delta, A, B, C, D):
    B_s, L, D_in = u.shape
    d_state = A.shape[1]
    delta_soft = F.softplus(delta)
    A_bar = torch.exp(delta_soft.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    B_bar = delta_soft.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    h = torch.zeros(B_s, D_in, d_state, device=u.device)
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + B_bar[:, t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1) + D * u[:, t]
        ys.append(y_t)
    return torch.stack(ys, dim=1)

def test_mamba():
    print("Testing 3/5: Mamba (selective_scan)...")
    max_fwd, max_du, max_ddelta, max_dA, max_dBC = 0.0, 0.0, 0.0, 0.0, 0.0
    for (B, L, D_in, d_state) in [(2, 32, 64, 16), (2, 64, 128, 16), (2, 128, 256, 16)]:
        torch.manual_seed(42)
        u = torch.randn(B, L, D_in, device='cuda', requires_grad=True)
        delta = torch.randn(B, L, D_in, device='cuda', requires_grad=True)
        A = -torch.rand(D_in, d_state, device='cuda', requires_grad=True)
        B_mat = torch.randn(B, L, d_state, device='cuda', requires_grad=True)
        C_mat = torch.randn(B, L, d_state, device='cuda', requires_grad=True)
        D_vec = torch.randn(D_in, device='cuda', requires_grad=True)

        y_tri = selective_scan_fn(u, delta, A, B_mat, C_mat, D_vec, delta_softplus=True)
        y_ref = ref_selective_scan(u, delta, A, B_mat, C_mat, D_vec)
        g = torch.randn_like(y_tri)

        gu_t, gdel_t, gA_t, gB_t, gC_t, gD_t = torch.autograd.grad(y_tri, (u, delta, A, B_mat, C_mat, D_vec), g)
        gu_r, gdel_r, gA_r, gB_r, gC_r, gD_r = torch.autograd.grad(y_ref, (u, delta, A, B_mat, C_mat, D_vec), g)

        max_fwd = max(max_fwd, rel_norm(y_tri, y_ref))
        max_du = max(max_du, rel_norm(gu_t, gu_r))
        max_ddelta = max(max_ddelta, rel_norm(gdel_t, gdel_r))
        max_dA = max(max_dA, rel_norm(gA_t, gA_r))
        max_dBC = max(max_dBC, max(rel_norm(gB_t, gB_r), rel_norm(gC_t, gC_r)))

    status = "PASS" if max(max_fwd, max_du, max_ddelta, max_dA, max_dBC) < 1e-3 else "FAIL"
    return {"name": "mamba", "fwd": max_fwd, "dq": max_du, "dk": max_ddelta, "dv": max_dA, "dparam": max_dBC, "status": status}

# ==============================================================================
# 4. HGRN2 TEST
# ==============================================================================
def ref_hgrn2(f, iv):
    B, L, D = f.shape
    h = torch.zeros(B, D, device=f.device, dtype=f.dtype)
    ys = []
    for t in range(L):
        h = f[:, t] * h + iv[:, t]
        ys.append(h)
    return torch.stack(ys, dim=1)

def test_hgrn2():
    print("Testing 4/5: HGRN2 (hgrn2_scan)...")
    max_fwd, max_df, max_div = 0.0, 0.0, 0.0
    for (B, L, D) in [(2, 32, 64), (2, 64, 128), (2, 128, 768)]:
        torch.manual_seed(42)
        f = torch.sigmoid(torch.randn(B, L, D, device='cuda', requires_grad=True))
        iv = torch.randn(B, L, D, device='cuda', requires_grad=True)

        y_tri = hgrn2_scan_triton(f, iv)
        y_ref = ref_hgrn2(f, iv)
        g = torch.randn_like(y_tri)

        gf_t, giv_t = torch.autograd.grad(y_tri, (f, iv), g)
        gf_r, giv_r = torch.autograd.grad(y_ref, (f, iv), g)

        max_fwd = max(max_fwd, rel_norm(y_tri, y_ref))
        max_df = max(max_df, rel_norm(gf_t, gf_r))
        max_div = max(max_div, rel_norm(giv_t, giv_r))

    status = "PASS" if max(max_fwd, max_df, max_div) < 1e-3 else "FAIL"
    return {"name": "hgrn2", "fwd": max_fwd, "dq": max_df, "dk": max_div, "dv": 0.0, "dparam": 0.0, "status": status}

# ==============================================================================
# 5. GDN TEST
# ==============================================================================
def ref_gdn(q, k, v, beta, alpha):
    B, H, L, D = q.shape
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        beta_t = beta[:, :, t]
        alpha_t = alpha[:, :, t]

        inter_v = torch.einsum('bhd,bhde->bhe', k_t, S)
        err = v_t - inter_v
        k_norm_sq = (k_t * k_t).sum(dim=-1, keepdim=True)
        step = alpha_t.unsqueeze(-1) / (1.0 + k_norm_sq).clamp_min(1.0)
        update = step.unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', k_t, err)
        S = beta_t.view(B, H, 1, 1) * S + update
        y_t = torch.einsum('bhd,bhde->bhe', q_t, S)
        ys.append(y_t)
    return torch.stack(ys, dim=2)

def test_gdn():
    print("Testing 5/5: GDN (gated_delta_net)...")
    max_fwd, max_dq, max_dk, max_dv, max_dparam = 0.0, 0.0, 0.0, 0.0, 0.0
    for (B, H, L, D) in [(2, 4, 32, 16), (2, 4, 64, 32), (2, 8, 128, 64)]:
        torch.manual_seed(42)
        q = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, L, D, device='cuda', dtype=torch.float32, requires_grad=True)
        beta = torch.sigmoid(torch.randn(B, H, L, device='cuda', dtype=torch.float32)).requires_grad_(True)
        alpha = torch.sigmoid(torch.randn(B, H, L, device='cuda', dtype=torch.float32)).requires_grad_(True)

        y_tri = gdn_scan_triton(q, k, v, beta, alpha)
        y_ref = ref_gdn(q, k, v, beta, alpha)
        g = torch.randn_like(y_tri)

        gq_t, gk_t, gv_t, gb_t, ga_t = torch.autograd.grad(y_tri, (q, k, v, beta, alpha), g)
        gq_r, gk_r, gv_r, gb_r, ga_r = torch.autograd.grad(y_ref, (q, k, v, beta, alpha), g)

        max_fwd = max(max_fwd, rel_norm(y_tri, y_ref))
        max_dq = max(max_dq, rel_norm(gq_t, gq_r))
        max_dk = max(max_dk, rel_norm(gk_t, gk_r))
        max_dv = max(max_dv, rel_norm(gv_t, gv_r))
        max_dparam = max(max_dparam, max(rel_norm(gb_t, gb_r), rel_norm(ga_t, ga_r)))

    status = "PASS" if max(max_fwd, max_dq, max_dk, max_dv, max_dparam) < 1e-3 else "FAIL"
    return {"name": "GDN", "fwd": max_fwd, "dq": max_dq, "dk": max_dk, "dv": max_dv, "dparam": max_dparam, "status": status}

def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    results = []
    results.append(test_fms())
    results.append(test_gla())
    results.append(test_mamba())
    results.append(test_hgrn2())
    results.append(test_gdn())

    print("\n" + "=" * 80)
    print("MASTER TRITON KERNEL NUMERICAL GRADIENT PROTOCOL REPORT")
    print("=" * 80)
    header = f"| {'Kernel':<8} | {'Forward err':<11} | {'dq err':<8} | {'dk err':<8} | {'dv err':<8} | {'d(other) err':<12} | {'Status':<6} |"
    print(header)
    print("|" + "-"*10 + "|" + "-"*13 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*8 + "|")
    all_passed = True
    for r in results:
        dv_str = f"{r['dv']:.2e}" if r['dv'] > 0 else "N/A"
        dp_str = f"{r['dparam']:.2e}" if r['dparam'] > 0 else "N/A"
        row = f"| {r['name']:<8} | {r['fwd']:<11.2e} | {r['dq']:<8.2e} | {r['dk']:<8.2e} | {dv_str:<8} | {dp_str:<12} | {r['status']:<6} |"
        print(row)
        if r['status'] != "PASS":
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("[OK] ALL 5 TRITON KERNELS PASSED STRICT NUMERICAL VALIDATION!")
        return 0
    else:
        print("[FAIL] AT LEAST ONE TRITON KERNEL FAILED VALIDATION.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
