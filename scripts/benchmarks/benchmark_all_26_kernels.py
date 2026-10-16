# -*- coding: utf-8 -*-
"""Direct Triton vs PyTorch physical hardware benchmark for NeuralOps JIT kernel modules."""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Direct Kernel Benchmark on: {device} ({torch.cuda.get_device_name(0)})")
    dtype = torch.float16

    results = []

    def bench_pair(name, shape_str, triton_fn, pytorch_fn, *args, **kwargs):
        # Warmup
        for _ in range(5):
            _ = triton_fn(*args, **kwargs)
            _ = pytorch_fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Measure Triton
        t0 = time.perf_counter()
        for _ in range(30):
            out_t = triton_fn(*args, **kwargs)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 30.0 * 1000.0

        # Measure PyTorch Reference
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            out_p = pytorch_fn(*args, **kwargs)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 30.0 * 1000.0

        if isinstance(out_t, tuple):
            out_t = out_t[0]
        if isinstance(out_p, tuple):
            out_p = out_p[0]

        max_diff = (out_t.float() - out_p.float()).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": name,
            "shape": shape_str,
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"  {name:36s} {shape_str:20s}: Tri={t_tri:6.3f}ms, PT={t_pt:6.3f}ms, Speedup={sp:5.2f}x, Diff={max_diff:.2e}")

    # 1. low_rank: fused_hnc_nullspace_step
    from neuralops.core.triton_kernels.low_rank import (
        fused_hnc_nullspace_step_pytorch,
        fused_hnc_nullspace_step_triton,
    )
    print("\n--- 1. low_rank (HNC Nullspace Step) ---")
    dim = 2048
    w = torch.randn(dim, dim, device=device, dtype=dtype)
    g = torch.randn(dim, dim, device=device, dtype=dtype)
    r = torch.randn(dim, dim, device=device, dtype=dtype)
    bench_pair("fused_hnc_nullspace_step", f"({dim}x{dim})", fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch, w, g, r, 0.05)

    # 2. low_rank: fused_linformer_seq_proj
    from neuralops.core.triton_kernels.low_rank import (
        fused_linformer_seq_proj_pytorch,
        fused_linformer_seq_proj_triton,
    )
    print("\n--- 2. low_rank (Linformer Projection) ---")
    seq_len = 4096
    x_lin = torch.randn(8, 16, seq_len, 64, device=device, dtype=dtype)
    proj_lin = torch.randn(64, seq_len, device=device, dtype=dtype)
    bench_pair("fused_linformer_seq_proj", f"(T={seq_len})", fused_linformer_seq_proj_triton, fused_linformer_seq_proj_pytorch, proj_lin, x_lin)

    # 3. low_rank: fused_kronecker_batched_gemm
    from neuralops.core.triton_kernels.low_rank import (
        fused_kronecker_batched_gemm_pytorch,
        fused_kronecker_batched_gemm_triton,
    )
    print("\n--- 3. low_rank (Kronecker Batched GEMM) ---")
    x_kron = torch.randn(4, 128, 256, device=device, dtype=dtype)
    A_kron = torch.randn(16, 16, device=device, dtype=dtype)
    B_kron = torch.randn(16, 16, device=device, dtype=dtype)
    bench_pair("fused_kronecker_gemm", "(dim=256, r=16)", fused_kronecker_batched_gemm_triton, fused_kronecker_batched_gemm_pytorch, x_kron, A_kron, B_kron)

    # 4. low_rank: fused_nystrom_landmark_pinv
    from neuralops.core.triton_kernels.low_rank import (
        fused_nystrom_landmark_pinv_step_pytorch,
        fused_nystrom_landmark_pinv_step_triton,
    )
    print("\n--- 4. low_rank (Nystrom Landmark Pinv) ---")
    m = 32
    F1 = F.softmax(torch.randn(4, 8, 512, m, device=device, dtype=torch.float32), dim=-1)
    F2_pinv = torch.randn(4, 8, m, m, device=device, dtype=torch.float32)
    F3 = F.softmax(torch.randn(4, 8, m, 512, device=device, dtype=torch.float32), dim=-1)
    V_nys = torch.randn(4, 8, 512, 64, device=device, dtype=torch.float32)
    bench_pair("fused_nystrom_landmark_pinv", f"(m={m}, N=512)", fused_nystrom_landmark_pinv_step_triton, fused_nystrom_landmark_pinv_step_pytorch, F1, F2_pinv, F3, V_nys)

    # 5. sliding_window
    from neuralops.core.triton_kernels.sliding_window import (
        sliding_window_attention,
        sliding_window_pytorch,
    )
    print("\n--- 5. sliding_window (Sliding Window Attention) ---")
    seq_len = 2048
    q_sw = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_sw = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_sw = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    bench_pair("sliding_window_attention", f"(T={seq_len}, W=128)", sliding_window_attention, sliding_window_pytorch, q_sw, k_sw, v_sw, 128)

    # 6. gla_scan
    from neuralops.core.triton_kernels.gla_scan import gla_scan, gla_scan_pytorch
    print("\n--- 6. gla_scan (Gated Linear Attention Scan) ---")
    seq_len = 1024
    q_gla = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_gla = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_gla = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    g_gla = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype)).clamp(1e-4, 1.0 - 1e-4)
    bench_pair("gla_scan", f"(T={seq_len})", gla_scan, gla_scan_pytorch, q_gla, k_gla, v_gla, g_gla)

    # 7. gdn_scan
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan, gdn_scan_pytorch
    print("\n--- 7. gdn_scan (Gated DeltaNet Scan) ---")
    seq_len = 1024
    q_gdn = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_gdn = F.normalize(torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype), dim=-1)
    v_gdn = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    beta_gdn = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype))
    alpha_gdn = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype))
    bench_pair("gdn_scan", f"(T={seq_len})", gdn_scan, gdn_scan_pytorch, q_gdn, k_gdn, v_gdn, beta_gdn, alpha_gdn)

    # 8. hgrn2_scan
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan, hgrn2_scan_pytorch
    print("\n--- 8. hgrn2_scan (HGRN-2 Scan) ---")
    seq_len = 1024
    f_hgrn = torch.sigmoid(torch.randn(2, seq_len, 256, device=device, dtype=dtype))
    iv_hgrn = torch.randn(2, seq_len, 256, device=device, dtype=dtype)
    bench_pair("hgrn2_scan", f"(T={seq_len})", lambda f, iv: hgrn2_scan(f, iv, res_clip=1.0), lambda f, iv: hgrn2_scan_pytorch(f, iv, res_clip=1.0), f_hgrn, iv_hgrn)

    # 9. softplus
    from neuralops.core.triton_kernels.softplus import softplus_fwd, softplus_pytorch
    print("\n--- 9. softplus (Fused Activation) ---")
    num_elements = 4194304
    x_sp = torch.randn(num_elements, device=device, dtype=dtype)
    bench_pair("softplus", f"({num_elements})", softplus_fwd, softplus_pytorch, x_sp)

    # 10. fused_norm_linear
    from neuralops.core.triton_kernels.fused_norm_linear import (
        fused_norm_linear,
        fused_norm_linear_pytorch,
    )
    print("\n--- 10. fused_norm_linear (RMSNorm + Linear) ---")
    dim = 2048
    x_fnl = torch.randn(2, 128, dim, device=device, dtype=dtype)
    w_fnl = torch.randn(dim, dim, device=device, dtype=dtype)
    b_fnl = torch.randn(dim, device=device, dtype=dtype)
    bench_pair("fused_norm_linear", f"({dim})", fused_norm_linear, fused_norm_linear_pytorch, x_fnl, w_fnl, bias=b_fnl)

    # 11. cosformer_triton_kernel
    from neuralops.core.triton_kernels.cosformer_triton_kernel import (
        cosformer_pytorch_fwd,
        cosformer_triton_fwd,
    )
    print("\n--- 11. cosformer (Cosine Reweighted Attention) ---")
    seq_len = 1024
    q_cos = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_cos = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_cos = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    scale_cos = 64.0 ** -0.5
    q_cos_act = F.elu(q_cos.float() * scale_cos) + 1.0
    k_cos_act = F.elu(k_cos.float()) + 1.0
    bench_pair("cosformer", f"(T={seq_len})", lambda q, k, v: cosformer_triton_fwd(q, k, v, is_causal=False), lambda q, k, v: cosformer_pytorch_fwd(q, k, v, is_causal=False), q_cos_act, k_cos_act, v_cos.float())

    # 12. fused_ce
    from neuralops.core.triton_kernels.fused_ce import (
        fused_cross_entropy,
        fused_cross_entropy_pytorch,
    )
    print("\n--- 12. fused_ce (Fused Cross-Entropy) ---")
    vocab = 50257
    logits_ce = torch.randn(8, 256, vocab, device=device, dtype=dtype)
    targets_ce = torch.randint(0, vocab, (8, 256), device=device)
    bench_pair("fused_ce", f"(V={vocab})", fused_cross_entropy, fused_cross_entropy_pytorch, logits_ce, targets_ce)

    # 13. fused_kd
    from neuralops.core.triton_kernels.fused_kd import (
        fused_kd_loss,
        fused_kd_loss_pytorch,
    )
    print("\n--- 13. fused_kd (Fused KL Divergence) ---")
    vocab = 16384
    s_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
    t_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
    bench_pair("fused_kd", f"(V={vocab})", fused_kd_loss, fused_kd_loss_pytorch, s_logits, t_logits, 2.0)

    # 14. selective_scan
    from neuralops.core.triton_kernels.selective_scan import (
        selective_scan_fn,
        selective_scan_ref,
    )
    print("\n--- 14. selective_scan (Mamba State-Space Scan) ---")
    seq_len = 512
    u_ssm = torch.randn(2, 64, seq_len, device=device, dtype=dtype)
    delta_ssm = F.softplus(torch.randn(2, 64, seq_len, device=device, dtype=dtype)) * 0.05
    A_ssm = -torch.rand(64, 16, device=device, dtype=dtype) - 0.1
    B_ssm = torch.randn(2, 16, seq_len, device=device, dtype=dtype) * 0.1
    C_ssm = torch.randn(2, 16, seq_len, device=device, dtype=dtype) * 0.1
    bench_pair("selective_scan", f"(T={seq_len})", selective_scan_fn, selective_scan_ref, u_ssm, delta_ssm, A_ssm, B_ssm, C_ssm)

    # 15. fms_scan
    from neuralops.core.triton_kernels.fms_scan import fms_scan, fms_scan_pytorch
    print("\n--- 15. fms_scan (Factorized Matrix State) ---")
    seq_len = 1024
    q_fms = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_fms = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_fms = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    decay_fms = torch.sigmoid(torch.randn(4, device=device, dtype=dtype))
    bench_pair("fms_scan", f"(T={seq_len})", fms_scan, fms_scan_pytorch, q_fms, k_fms, v_fms, decay_fms)

    # 16. paged_attention
    from neuralops.core.triton_kernels.paged_attention import (
        paged_attention_kernel,
        paged_attention_pytorch,
    )
    print("\n--- 16. paged_attention (Paged KV Block Cache) ---")
    num_blocks, block_size, num_heads, head_dim = 128, 16, 4, 64
    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(2, 64)
    context_lens = torch.full((2,), 64 * block_size, device=device, dtype=torch.int32)
    q_pa = torch.randn(2, num_heads, head_dim, device=device, dtype=dtype)
    bench_pair("paged_attention", "(B=2, H=4, d=64)", paged_attention_kernel, paged_attention_pytorch, q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size)

    # 17. gqa: fused_mqa_bias_scale
    from neuralops.core.triton_kernels.gqa import (
        fused_mqa_bias_scale_triton,
        fused_mqa_bias_scale_pytorch,
    )
    print("\n--- 17. gqa (Fused MQA Bias & Scaling) ---")
    scores_gqa = torch.randn(4, 8, 256, 256, device=device, dtype=torch.float32)
    bench_pair("fused_mqa_bias_scale", "(256x256, causal)", fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch, scores_gqa, 0.25, is_causal=True, past_len=4)

    # 18. sparse_attn: fused_hamming_scores
    from neuralops.core.triton_kernels.sparse_attn import (
        fused_hamming_scores_triton,
        fused_hamming_scores_pytorch,
    )
    print("\n--- 18. sparse_attn (Fused Hamming Bit Scores) ---")
    Q_ham = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
    K_ham = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
    lora_q = torch.randn(8, device=device, dtype=torch.float32)
    lora_k = torch.randn(8, device=device, dtype=torch.float32)
    temp = torch.tensor([1.2], device=device, dtype=torch.float32)
    bench_pair("fused_hamming_scores", "(256x64)", fused_hamming_scores_triton, fused_hamming_scores_pytorch, Q_ham, K_ham, lora_q, lora_k, temp)

    # 19. elementwise: fused_favor_plus_feature_map
    from neuralops.core.triton_kernels.elementwise import (
        fused_favor_plus_feature_map_triton,
        fused_favor_plus_feature_map_pytorch,
    )
    print("\n--- 19. elementwise (FAVOR+ Feature Map) ---")
    x_fav = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
    omega_fav = torch.randn(128, 64, device=device, dtype=torch.float32)
    bench_pair("fused_favor_plus", "(256x64, M=128)", fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch, x_fav, omega_fav, num_features=128, kernel_type="favor_plus")

    # 20. fused_rope: native Triton JIT kernel
    from neuralops.core.triton_kernels.fused_rope import fused_rope_fwd_kernel, _apply_rope_rotation
    import triton
    print("\n--- 20. fused_rope (Direct Triton JIT RoPE) ---")
    B_r, H_r, N_r, D_r = 4, 8, 1024, 64
    q_r = torch.randn(B_r, H_r, N_r, D_r, device=device, dtype=dtype)
    k_r = torch.randn(B_r, H_r, N_r, D_r, device=device, dtype=dtype)
    cos_r = torch.cos(torch.randn(N_r, D_r // 2, device=device, dtype=dtype))
    sin_r = torch.sin(torch.randn(N_r, D_r // 2, device=device, dtype=dtype))

    def tri_rope_bench(q, k, cos, sin):
        out_q = torch.empty_like(q)
        BLOCK_M = 32
        grid = (triton.cdiv(N_r, BLOCK_M), B_r * H_r)
        fused_rope_fwd_kernel[grid](
            q, k, cos, sin, out_q, out_q,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            0, cos.stride(0), cos.stride(1),
            0, sin.stride(0), sin.stride(1),
            out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
            0, 0, 0, 0,
            H_r, N_r, D_r, D_r // 2,
            BLOCK_M=BLOCK_M,
        )
        return out_q

    bench_pair("fused_rope", f"(T={N_r}, d={D_r})", tri_rope_bench, lambda q, k, c, s: _apply_rope_rotation(q, c, s), q_r, k_r, cos_r, sin_r)

    # 21. block_sparse: fused_moba_block_router_combine
    from neuralops.core.triton_kernels.block_sparse import (
        fused_moba_block_router_combine_triton,
        fused_moba_block_router_combine_pytorch,
    )
    print("\n--- 21. block_sparse (MoBA Block Router) ---")
    Qb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
    Kb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
    Vb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
    bench_pair("fused_moba_block_router", "(16x16x32)", fused_moba_block_router_combine_triton, fused_moba_block_router_combine_pytorch, Qb_moba, Kb_moba, Vb_moba, 0.25, top_k=2, is_causal=True)

    # 22. linear_attn: fused_rala_augmented_scan
    from neuralops.core.triton_kernels.linear_attn import (
        fused_rala_augmented_scan_triton,
        fused_rala_augmented_scan_pytorch,
    )
    print("\n--- 22. linear_attn (RALA Augmented Scan) ---")
    q_rala = F.softplus(torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32))
    k_rala = F.softplus(torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32))
    v_rala = torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32)
    lora_U = torch.randn(4, 32, 4, device=device, dtype=torch.float32)
    lora_V = torch.randn(4, 32, 4, device=device, dtype=torch.float32)
    bench_pair("fused_rala_augmented_scan", "(128x32, r=4)", fused_rala_augmented_scan_triton, fused_rala_augmented_scan_pytorch, q_rala, k_rala, v_rala, lora_U, lora_V, is_causal=True)

    # 23. state_update: fused_recurrent_linear_step
    from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
    print("\n--- 23. state_update (Recurrent Linear Step) ---")
    q_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
    k_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
    v_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
    kv_state_rec = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
    z_state_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)

    def py_rec_step(q_t, k_t, v_t, kv_s, z_s, eps=1e-6):
        kv = kv_s + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float()).to(kv_s.dtype)
        z = z_s + k_t
        num = torch.einsum("bhk,bhkv->bhv", q_t.float(), kv.float())
        den = torch.einsum("bhk,bhk->bh", q_t.float(), z.float()).unsqueeze(-1) + eps
        return (num / den).to(q_t.dtype), kv, z

    bench_pair("fused_recurrent_linear_step", "(B=2, H=4, D=64)", fused_recurrent_linear_step, py_rec_step, q_rec, k_rec, v_rec, kv_state_rec.clone(), z_state_rec.clone())

    # 24. selective_scan: fused_jamba_hybrid_gate
    from neuralops.core.triton_kernels.selective_scan import (
        fused_jamba_hybrid_ssm_attn_gate_triton,
        fused_jamba_hybrid_ssm_attn_gate_pytorch,
    )
    print("\n--- 24. selective_scan (Jamba Hybrid Gate) ---")
    attn_out_j = torch.randn(8, 512, 1024, device=device, dtype=torch.float32)
    ssm_out_j = torch.randn(8, 512, 1024, device=device, dtype=torch.float32)
    alpha_j = torch.randn(16, device=device, dtype=torch.float32)
    bench_pair("fused_jamba_hybrid_gate", "(512x1024, H=16)", fused_jamba_hybrid_ssm_attn_gate_triton, fused_jamba_hybrid_ssm_attn_gate_pytorch, attn_out_j, ssm_out_j, alpha_j)

    # 25. fms_scan: fused_goldfinch_hopfield_lookup
    from neuralops.core.triton_kernels.fms_scan import (
        fused_goldfinch_hopfield_lookup_triton,
        fused_goldfinch_hopfield_lookup_pytorch,
    )
    print("\n--- 25. fms_scan (Goldfinch Hopfield Lookup) ---")
    g_gf = torch.sigmoid(torch.randn(4, 512, 128, device=device, dtype=torch.float32))
    v_gf = torch.tanh(torch.randn(4, 512, 128, device=device, dtype=torch.float32))
    bench_pair("fused_goldfinch_hopfield", "(512x128, chunk=16)", fused_goldfinch_hopfield_lookup_triton, fused_goldfinch_hopfield_lookup_pytorch, g_gf, v_gf, chunk_size=16)

    # 26. flash_decoding: flash_decode_kernel
    from neuralops.core.triton_kernels.flash_decoding import (
        flash_decode_kernel,
        flash_decoding_pytorch,
    )
    print("\n--- 26. flash_decoding (Flash Decode Attention) ---")
    q_fd = torch.randn(2, 4, 1, 64, device=device, dtype=dtype)
    k_fd = torch.randn(2, 4, 1024, 64, device=device, dtype=dtype)
    v_fd = torch.randn(2, 4, 1024, 64, device=device, dtype=dtype)
    bench_pair("flash_decode", "(N_kv=1024, H=4, d=64)", flash_decode_kernel, flash_decoding_pytorch, q_fd, k_fd, v_fd, chunk_size=64)

    out_path = Path("results/triton_all_26_kernels_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")
    faster_count = sum(1 for r in results if r["speedup"] >= 1.0)
    print(f"Summary: {faster_count}/{len(results)} configurations faster or equal to PyTorch reference")


if __name__ == "__main__":
    main()
