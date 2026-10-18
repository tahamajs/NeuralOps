import os
import sys
import time
import csv
import math
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels.cross_attention.cross_attn_autograd import cross_attention, cross_attention_pytorch_ref
from neuralops.core.cuda_kernels.tt_contraction.tt_autograd import tt_contraction, tt_contraction_pytorch_ref
from neuralops.core.cuda_kernels.block_diag_gemm.block_diag_autograd import block_diag_matmul, block_diag_matmul_pytorch_ref
from neuralops.core.cuda_kernels.samba_moe.samba_moe_autograd import samba_moe, samba_moe_ref

SEEDS = [404, 777, 999]

def run_stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Multi-Seed Stress Test on Backward Passes (Seeds {SEEDS}) on {device} ({torch.cuda.get_device_name(0)}) ===")

    results = []

    for seed in SEEDS:
        print(f"\n--- Testing Seed {seed} ---")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 1. cross_attention
        B, H, Sq, Skv, D = 2, 4, 32, 48, 64
        q_cuda = torch.randn(B, H, Sq, D, device=device, dtype=torch.float32, requires_grad=True)
        k_cuda = torch.randn(B, H, Skv, D, device=device, dtype=torch.float32, requires_grad=True)
        v_cuda = torch.randn(B, H, Skv, D, device=device, dtype=torch.float32, requires_grad=True)

        q_ref = q_cuda.detach().clone().requires_grad_(True)
        k_ref = k_cuda.detach().clone().requires_grad_(True)
        v_ref = v_cuda.detach().clone().requires_grad_(True)

        y_cuda = cross_attention(q_cuda, k_cuda, v_cuda)
        y_ref = cross_attention_pytorch_ref(q_ref, k_ref, v_ref)

        dout_ca = torch.randn_like(y_cuda)
        y_cuda.backward(dout_ca)
        y_ref.backward(dout_ca)

        fwd_diff_ca = (y_cuda - y_ref).abs().max().item()
        grad_diff_q = (q_cuda.grad - q_ref.grad).abs().max().item()
        grad_diff_k = (k_cuda.grad - k_ref.grad).abs().max().item()
        grad_diff_v = (v_cuda.grad - v_ref.grad).abs().max().item()
        max_grad_ca = max(grad_diff_q, grad_diff_k, grad_diff_v)

        has_nan_ca = any(torch.isnan(g).any().item() for g in [q_cuda.grad, k_cuda.grad, v_cuda.grad])
        has_inf_ca = any(torch.isinf(g).any().item() for g in [q_cuda.grad, k_cuda.grad, v_cuda.grad])

        results.append({
            "kernel": "cross_attention",
            "seed": seed,
            "fwd_max_diff": fwd_diff_ca,
            "max_grad_diff": max_grad_ca,
            "zero_nan": (not has_nan_ca) and (not has_inf_ca),
            "status": "PASS" if max_grad_ca < 1e-4 and not has_nan_ca and not has_inf_ca else "FAIL"
        })
        print(f"  cross_attention | Fwd Diff: {fwd_diff_ca:.2e} | Max Grad Diff: {max_grad_ca:.2e} | Zero-NaN: {not has_nan_ca}")

        # 2. tt_contraction
        p_dims = (4, 4)
        q_dims = (8, 8)
        c0_cuda = torch.randn(4, 8, 16, device=device, dtype=torch.float32, requires_grad=True)
        c1_cuda = torch.randn(16, 4, 8, device=device, dtype=torch.float32, requires_grad=True)

        c0_ref = c0_cuda.detach().clone().requires_grad_(True)
        c1_ref = c1_cuda.detach().clone().requires_grad_(True)

        y_tt_cuda = tt_contraction(c0_cuda, c1_cuda, p_dims, q_dims)
        y_tt_ref = tt_contraction_pytorch_ref(c0_ref, c1_ref, p_dims, q_dims)

        dout_tt = torch.randn_like(y_tt_cuda)
        y_tt_cuda.backward(dout_tt)
        y_tt_ref.backward(dout_tt)

        fwd_diff_tt = (y_tt_cuda - y_tt_ref).abs().max().item()
        grad_diff_c0 = (c0_cuda.grad - c0_ref.grad).abs().max().item()
        grad_diff_c1 = (c1_cuda.grad - c1_ref.grad).abs().max().item()
        max_grad_tt = max(grad_diff_c0, grad_diff_c1)

        has_nan_tt = any(torch.isnan(g).any().item() for g in [c0_cuda.grad, c1_cuda.grad])
        has_inf_tt = any(torch.isinf(g).any().item() for g in [c0_cuda.grad, c1_cuda.grad])

        results.append({
            "kernel": "tt_contraction",
            "seed": seed,
            "fwd_max_diff": fwd_diff_tt,
            "max_grad_diff": max_grad_tt,
            "zero_nan": (not has_nan_tt) and (not has_inf_tt),
            "status": "PASS" if max_grad_tt < 1e-4 and not has_nan_tt and not has_inf_tt else "FAIL"
        })
        print(f"  tt_contraction  | Fwd Diff: {fwd_diff_tt:.2e} | Max Grad Diff: {max_grad_tt:.2e} | Zero-NaN: {not has_nan_tt}")

        # 3. block_diag_gemm
        x_cuda = torch.randn(4, 64, device=device, dtype=torch.float32, requires_grad=True)
        w_cuda = torch.randn(4, 16, 16, device=device, dtype=torch.float32, requires_grad=True)

        x_ref = x_cuda.detach().clone().requires_grad_(True)
        w_ref = w_cuda.detach().clone().requires_grad_(True)

        y_bd_cuda = block_diag_matmul(x_cuda, w_cuda)
        y_bd_ref = block_diag_matmul_pytorch_ref(x_ref, w_ref)

        dout_bd = torch.randn_like(y_bd_cuda)
        y_bd_cuda.backward(dout_bd)
        y_bd_ref.backward(dout_bd)

        fwd_diff_bd = (y_bd_cuda - y_bd_ref).abs().max().item()
        grad_diff_x = (x_cuda.grad - x_ref.grad).abs().max().item()
        grad_diff_w = (w_cuda.grad - w_ref.grad).abs().max().item()
        max_grad_bd = max(grad_diff_x, grad_diff_w)

        has_nan_bd = any(torch.isnan(g).any().item() for g in [x_cuda.grad, w_cuda.grad])
        has_inf_bd = any(torch.isinf(g).any().item() for g in [x_cuda.grad, w_cuda.grad])

        results.append({
            "kernel": "block_diag_gemm",
            "seed": seed,
            "fwd_max_diff": fwd_diff_bd,
            "max_grad_diff": max_grad_bd,
            "zero_nan": (not has_nan_bd) and (not has_inf_bd),
            "status": "PASS" if max_grad_bd < 1e-4 and not has_nan_bd and not has_inf_bd else "FAIL"
        })
        print(f"  block_diag_gemm | Fwd Diff: {fwd_diff_bd:.2e} | Max Grad Diff: {max_grad_bd:.2e} | Zero-NaN: {not has_nan_bd}")

        # 4. samba_moe
        e1_cuda = torch.randn(4, 32, 64, device=device, dtype=torch.float32, requires_grad=True)
        e2_cuda = torch.randn(4, 32, 64, device=device, dtype=torch.float32, requires_grad=True)
        rw_cuda = torch.softmax(torch.randn(4, 32, 2, device=device, dtype=torch.float32), dim=-1).requires_grad_(True)

        e1_ref = e1_cuda.detach().clone().requires_grad_(True)
        e2_ref = e2_cuda.detach().clone().requires_grad_(True)
        rw_ref = rw_cuda.detach().clone().requires_grad_(True)

        y_sm_cuda = samba_moe(e1_cuda, e2_cuda, rw_cuda)
        y_sm_ref = samba_moe_ref(e1_ref, e2_ref, rw_ref)

        dout_sm = torch.randn_like(y_sm_cuda)
        y_sm_cuda.backward(dout_sm)
        y_sm_ref.backward(dout_sm)

        fwd_diff_sm = (y_sm_cuda - y_sm_ref).abs().max().item()
        grad_diff_e1 = (e1_cuda.grad - e1_ref.grad).abs().max().item()
        grad_diff_e2 = (e2_cuda.grad - e2_ref.grad).abs().max().item()
        grad_diff_rw = (rw_cuda.grad - rw_ref.grad).abs().max().item()
        max_grad_sm = max(grad_diff_e1, grad_diff_e2, grad_diff_rw)

        has_nan_sm = any(torch.isnan(g).any().item() for g in [e1_cuda.grad, e2_cuda.grad, rw_cuda.grad])
        has_inf_sm = any(torch.isinf(g).any().item() for g in [e1_cuda.grad, e2_cuda.grad, rw_cuda.grad])

        results.append({
            "kernel": "samba_moe",
            "seed": seed,
            "fwd_max_diff": fwd_diff_sm,
            "max_grad_diff": max_grad_sm,
            "zero_nan": (not has_nan_sm) and (not has_inf_sm),
            "status": "PASS" if max_grad_sm < 1e-4 and not has_nan_sm and not has_inf_sm else "FAIL"
        })
        print(f"  samba_moe       | Fwd Diff: {fwd_diff_sm:.2e} | Max Grad Diff: {max_grad_sm:.2e} | Zero-NaN: {not has_nan_sm}")

    # Output CSV
    base_dir = "/mnt/d/SVD_linear_Attention"
    csv_path = os.path.join(base_dir, "docs/benchmarks/paper_benchmarks_seeds_404_777_999.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "seed", "fwd_max_diff", "max_grad_diff", "zero_nan", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved CSV to: {csv_path}")

if __name__ == "__main__":
    run_stress_test()
