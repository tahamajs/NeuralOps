# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Subprocess-Isolated Master Benchmark across ALL 108 Native CUDA C++ Modules.

Runs each CUDA kernel test inside an independent isolated Python subprocess.
This guarantees:
1. Complete isolation: A single kernel failure or illegal memory address never
   corrupts the CUDA context for subsequent kernels.
2. Direct invocation: Loads `_get_module()` directly, verifying true CUDA C++ compilation
   and execution without fallback masking.
3. Detailed metrics: CUDA latency, reference latency, speedup, and numerical difference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

WORKER_CODE = """
import sys, os, time, json, inspect, importlib.util, math
from pathlib import Path
import torch

repo_root = Path("{repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

ALIASES = {aliases_json}

def time_callable(fn, warmup=3, iters=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0

def generate_args(mod_name, sig, device, dtype, B=4, H=8, N=512, D=64):
    args = []
    kwargs = {{}}

    if mod_name == "kronecker_gemm":
        x = torch.randn(B * 8, 16 * 16, device=device, dtype=dtype) * 0.1
        A = torch.randn(16, 16, device=device, dtype=dtype) * 0.1
        B_mat = torch.randn(16, 16, device=device, dtype=dtype) * 0.1
        return [x, A, B_mat], {{}}
    elif mod_name == "bitonic_topk":
        x = torch.randn(B * 8, 1024, device=device, dtype=dtype)
        return [x, 32], {{}}
    elif mod_name == "block_diag_gemm":
        G = 4
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        w = torch.randn(G, D // G, D // G, device=device, dtype=dtype) * 0.1
        return [x, w], {{}}
    elif mod_name == "cluster_attention":
        K_clust = 32
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        c_cen = torch.randn(B, H, K_clust, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, K_clust, D, device=device, dtype=dtype) * 0.1
        c_idx = torch.randint(0, K_clust, (B, H, N), device=device, dtype=torch.int32)
        return [q, c_cen, v, c_idx], {{}}
    elif mod_name == "latent_kv_attn":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        c_kv = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        u_k = torch.randn(D, D, device=device, dtype=dtype) * 0.1
        u_v = torch.randn(D, D, device=device, dtype=dtype) * 0.1
        return [q, c_kv, u_k, u_v], {{}}
    elif mod_name == "hadamard_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.05
        w_p = torch.randint(0, 255, (N_dim, K_dim // 2), device=device, dtype=torch.uint8)
        sc = torch.full((N_dim, K_dim // 32), 0.01, device=device, dtype=dtype)
        zr = torch.zeros(N_dim, K_dim // 32, device=device, dtype=dtype)
        return [x_gemm, w_p, sc, zr], {{"group_size": 32}}
    elif mod_name == "ternary_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.1
        w_p = torch.randint(0, 255, (N_dim, K_dim // 4), device=device, dtype=torch.uint8)
        sc = torch.full((N_dim,), 0.01, device=device, dtype=dtype)
        return [x_gemm, w_p, sc], {{}}
    elif mod_name == "lut_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.1
        w_p = torch.randint(0, 15, (N_dim, K_dim // 2), device=device, dtype=torch.uint8)
        cb = torch.randn(K_dim // 64, 16, device=device, dtype=dtype) * 0.1
        return [x_gemm, w_p, cb], {{"group_size": 64}}
    elif mod_name == "rotational_svd":
        M_dim, N_dim, K_dim, R = 64, 128, 128, 32
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.1
        r_in = torch.randn(K_dim, K_dim, device=device, dtype=dtype) * 0.1
        v_r = torch.randn(R, K_dim, device=device, dtype=dtype) * 0.1
        u_r = torch.randn(N_dim, R, device=device, dtype=dtype) * 0.1
        return [x_gemm, r_in, v_r, u_r], {{}}
    elif mod_name == "rpca_sparse_svd":
        M_dim, N_dim, K_dim, R = 64, 128, 128, 32
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.1
        v_r = torch.randn(K_dim, R, device=device, dtype=dtype) * 0.1
        u_r = torch.randn(R, N_dim, device=device, dtype=dtype) * 0.1
        return [x_gemm, v_r, u_r], {{}}
    elif mod_name == "layernorm":
        x = torch.randn(B * 8, D * 8, device=device, dtype=dtype) * 0.1
        w = torch.randn(D * 8, device=device, dtype=dtype) * 0.1
        b = torch.randn(D * 8, device=device, dtype=dtype) * 0.1
        return [x, w, b], {{}}
    elif mod_name == "rmsnorm":
        x = torch.randn(B * 8, D * 8, device=device, dtype=dtype) * 0.1
        w = torch.randn(D * 8, device=device, dtype=dtype) * 0.1
        return [x, w], {{}}
    elif mod_name == "channel_shuffle":
        x = torch.randn(B, 16, 32, 32, device=device, dtype=dtype) * 0.1
        return [x, 4], {{}}
    elif mod_name == "moe_routing":
        tokens = torch.randn(B * 8, D, device=device, dtype=dtype) * 0.1
        gate_logits = torch.randn(B * 8, 8, device=device, dtype=dtype)
        expert = torch.nn.Linear(D, D, device=device, dtype=dtype)
        return [tokens, gate_logits, expert], {{"top_k": 2}}
    elif mod_name == "selective_scan":
        u = torch.randn(B, 64, N, device=device, dtype=dtype) * 0.1
        delta = torch.randn(B, 64, N, device=device, dtype=dtype) * 0.1
        A = torch.randn(64, 16, device=device, dtype=torch.float32) * 0.1
        B_t = torch.randn(B, 16, N, device=device, dtype=dtype) * 0.1
        C_t = torch.randn(B, 16, N, device=device, dtype=dtype) * 0.1
        D_t = torch.randn(64, device=device, dtype=torch.float32) * 0.1
        return [u, delta, A, B_t, C_t, D_t], {{}}
    elif mod_name == "selective_ssm":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        dt = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        A = torch.full((D,), 0.95, device=device, dtype=dtype)
        B_vec = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        C_vec = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, dt, A, B_vec, C_vec], {{}}
    elif mod_name == "simple_ssm":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, N, D), 0.95, device=device, dtype=dtype)
        scale = torch.ones(D, device=device, dtype=dtype)
        return [x, decay, scale], {{}}
    elif mod_name == "quantized_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        num_g = K_dim // 128
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.05
        w_p = torch.randint(0, 255, (K_dim, N_dim // 2), device=device, dtype=torch.uint8)
        sc = torch.full((num_g, N_dim), 0.01, device=device, dtype=dtype)
        zr = torch.zeros(num_g, N_dim, device=device, dtype=dtype)
        return [x_gemm, w_p, sc, zr], {{"group_size": 128}}
    elif mod_name == "sparse_attention":
        bs = 64
        nb = N // bs
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        mask_2d = torch.ones(nb, nb, device=device, dtype=torch.int32)
        return [q, k, v, mask_2d], {{"block_size": bs}}
    elif mod_name == "decay_linear_attention":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        gamma = torch.full((H, D), 0.95, device=device, dtype=dtype)
        return [q, k, v, gamma], {{}}
    elif mod_name == "linear_attention":
        q = torch.nn.functional.elu(torch.randn(B, H, N, D, device=device, dtype=dtype)) + 1.0
        k = torch.nn.functional.elu(torch.randn(B, H, N, D, device=device, dtype=dtype)) + 1.0
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        return [q, k, v], {{}}
    elif mod_name == "causal_conv1d":
        x = torch.randn(B, D, N, device=device, dtype=dtype) * 0.1
        weight = torch.randn(D, 4, device=device, dtype=dtype) * 0.1
        return [x, weight], {{}}
    elif mod_name == "dynamic_conv1d":
        K_conv = 4
        N_conv = 128
        v_full = torch.randn(B, H, N_conv + K_conv - 1, D, device=device, dtype=dtype) * 0.1
        dyn_w = torch.randn(B, H, N_conv, K_conv, device=device, dtype=dtype) * 0.1
        return [v_full, dyn_w], {{}}
    elif mod_name == "flash_decode_stage2":
        num_splits = 8
        mid_out = torch.randn(B, H, num_splits, D, device=device, dtype=dtype) * 0.1
        mid_lse = torch.randn(B, H, num_splits, device=device, dtype=torch.float32)
        return [mid_out, mid_lse], {{}}
    elif mod_name == "goldfinch_ahn":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        beta = torch.full((B, H, N, D), 0.95, device=device, dtype=dtype)
        return [q, k, v, beta], {{}}
    elif mod_name == "griffin_ssd":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        r = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, N, D), 0.95, device=device, dtype=dtype)
        return [x, r, decay], {{}}
    elif mod_name == "fused_rope":
        x = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        cos = torch.randn(1, 1, N, D // 2, device=device, dtype=dtype)
        sin = torch.randn(1, 1, N, D // 2, device=device, dtype=dtype)
        return [x, cos, sin], {{}}
    elif mod_name == "gdn_scan":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        alpha = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype)) * 0.95
        beta = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype)) * 0.95
        return [q, k, v, alpha, beta], {{}}
    elif mod_name == "gla_scan":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        alpha = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype)) * 0.95
        return [q, k, v, alpha], {{}}
    elif mod_name == "recurrent_hgrn2_step":
        x = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        g = torch.sigmoid(torch.randn(B, H, D, device=device, dtype=dtype)) * 0.95
        alpha = torch.sigmoid(torch.randn(B, H, D, device=device, dtype=dtype)) * 0.95
        prev_state = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        return [x, g, alpha, prev_state], {{}}
    elif mod_name == "gla_unnormalized_scan":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, H, N, D), -0.05, device=device, dtype=dtype)
        return [q, k, v, decay], {{}}
    elif mod_name == "recurrent_gla_step":
        q = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        s_state = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.1
        alpha = torch.full((B, H), 0.95, device=device, dtype=dtype)
        return [q, k, v, s_state, alpha], {{}}
    elif mod_name == "varlen_attention":
        total_toks = 512
        seq_len = 128
        q = torch.randn(total_toks, H, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(total_toks, H, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(total_toks, H, D, device=device, dtype=dtype) * 0.1
        cu_seqlens = torch.arange(0, total_toks + 1, seq_len, device=device, dtype=torch.int32)
        return [q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len], {{"is_causal": False}}
    elif mod_name == "hgrn2_spatial_scan":
        x = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        g = torch.sigmoid(torch.randn(B, H, N, D, device=device, dtype=dtype)) * 0.95
        return [x, g], {{}}
    elif mod_name == "hgrn2_scan":
        x = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        g = torch.sigmoid(torch.randn(B, H, N, D, device=device, dtype=dtype)) * 0.95
        return [x, g], {{}}
    elif mod_name == "linear_rnn_attention":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        a = torch.full((B, N, D), 0.95, device=device, dtype=dtype)
        b = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, a, b], {{}}
    elif mod_name == "nasc_attention":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, H, N, D), 0.95, device=device, dtype=dtype)
        return [q, k, v, decay], {{}}
    elif mod_name == "nystrom_attention":
        q_l = torch.randn(B, H, 32, D, device=device, dtype=dtype) * 0.1
        k_l = torch.randn(B, H, 32, D, device=device, dtype=dtype) * 0.1
        v_l = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.1
        p_l = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.1
        return [q_l, k_l, v_l, p_l], {{}}
    elif mod_name == "paged_attention":
        num_seqs, num_heads, num_kv_heads, head_dim, block_sz, max_blocks = 4, 8, 2, 64, 16, 8
        q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype) * 0.1
        num_total_blocks = num_seqs * max_blocks
        k_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=dtype) * 0.1
        v_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=dtype) * 0.1
        context_lens = torch.tensor([16, 32, 48, 64], device=device, dtype=torch.int32)
        block_tables = torch.zeros(num_seqs, max_blocks, device=device, dtype=torch.int32)
        for i in range(num_seqs):
            nb = int(context_lens[i].item()) // block_sz
            block_tables[i, :nb] = torch.arange(i * max_blocks, i * max_blocks + nb, device=device)
        return [q, k_cache, v_cache, block_tables, context_lens], {{"block_size": block_sz}}
    elif mod_name == "paged_kv_cache":
        nb, num_h, head_d, block_sz = 16, 8, 64, 16
        src_k = torch.randn(nb, num_h, head_d, block_sz, device=device, dtype=dtype) * 0.1
        src_v = torch.randn(nb, num_h, head_d, block_sz, device=device, dtype=dtype) * 0.1
        dst_k = torch.zeros_like(src_k)
        dst_v = torch.zeros_like(src_v)
        b_map = torch.tensor([[0, 2], [1, 5], [3, 7]], device=device, dtype=torch.int32)
        return [dst_k, dst_v, src_k, src_v, b_map, block_sz, num_h, head_d], {{}}
    elif mod_name == "paged_kv_evict":
        k_in = torch.randn(64, D * 8, device=device, dtype=dtype) * 0.1
        v_in = torch.randn(64, D * 8, device=device, dtype=dtype) * 0.1
        keep = torch.arange(32, device=device, dtype=torch.int64)
        return [k_in, v_in, keep], {{}}
    elif mod_name == "paged_recurrent_step":
        q = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        page_pool = torch.randn(32, H, D, D, device=device, dtype=dtype) * 0.1
        block_tables = torch.arange(B, device=device, dtype=torch.int64)
        return [q, k, v, page_pool, block_tables], {{"decay": 0.95}}
    elif mod_name == "rfa_attention":
        phi_q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        phi_k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        return [phi_q, phi_k, v], {{}}
    elif mod_name == "rwkv6":
        r = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        w = torch.full((B, H, N, D), -0.05, device=device, dtype=dtype)
        u = torch.randn(H, D, device=device, dtype=dtype) * 0.1
        return [r, k, v, w, u], {{}}
    elif mod_name == "spectraformer_attention":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        c_basis = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        s_basis = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, c_basis, s_basis], {{}}
    elif mod_name == "spectral_mamba":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        freq = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, freq], {{"threshold": 0.01}}
    elif mod_name == "spectral_shrinkage":
        x_re = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        x_im = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        w_re = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        w_im = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        tau = torch.full((B, N, D), 0.01, device=device, dtype=dtype)
        return [x_re, x_im, w_re, w_im, tau], {{}}
    elif mod_name == "tt_contraction":
        core0 = torch.randn(1, 8, 8, 8, device=device, dtype=dtype) * 0.1
        core1 = torch.randn(8, 8, 8, 1, device=device, dtype=dtype) * 0.1
        return [core0, core1, (8, 8), (8, 8)], {{}}
    elif mod_name == "upi_stuffed_mamba":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        upi_tok = torch.randn(B, 16, D, device=device, dtype=dtype) * 0.1
        return [x, upi_tok], {{"blend_weight": 0.2}}
    elif mod_name == "vspec":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        spec_gate = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, N, D), 0.95, device=device, dtype=dtype)
        return [x, spec_gate, decay], {{}}
    elif mod_name == "hymba_transx":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        meta_tok = torch.randn(B, 16, D, device=device, dtype=dtype) * 0.1
        cross_gate = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, meta_tok, cross_gate], {{}}
    elif mod_name == "jamba_info":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        bw = torch.randn(D, D, device=device, dtype=dtype) * 0.1
        return [x, bw], {{}}
    elif mod_name == "state_update":
        q = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        kv_state = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.1
        z_state = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
        return [q, k, v, kv_state, z_state], {{"eps": 1e-6}}
    elif mod_name == "samba_moe":
        e1 = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        e2 = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        rw = torch.softmax(torch.randn(B, N, 2, device=device, dtype=dtype), dim=-1)
        return [e1, e2, rw], {{}}
    elif mod_name == "samba_m3":
        m1 = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        m2 = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        la = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        g3 = torch.randn(B, N, 3, device=device, dtype=dtype)
        return [m1, m2, la, g3], {{}}
    elif mod_name == "nexus":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.05
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.05
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.05
        return [q, k, v], {{"tau": 1.0}}
    elif mod_name == "orion":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.05
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.05
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.05
        return [q, k, v], {{"temperature": 1.0}}
    elif mod_name == "differential_attention":
        q1 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k1 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        q2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        return [q1, k1, q2, k2, v], {{"lambda_val": 0.8}}
    elif mod_name == "adasplash_attention":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        return [q, k, v], {{"threshold": 0.1, "window_size": 64}}
    elif mod_name == "fms_scan":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((H,), 0.95, device=device, dtype=torch.float32)
        return [q, k, v, decay], {{}}
    elif mod_name == "multi_scale_retention":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        gammas = torch.full((H,), 0.95, device=device, dtype=torch.float32)
        return [q, k, v, gammas], {{}}
    elif mod_name == "linformer_attention":
        K_proj = 64
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k_proj = torch.randn(B, H, K_proj, D, device=device, dtype=dtype) * 0.1
        v_proj = torch.randn(B, H, K_proj, D, device=device, dtype=dtype) * 0.1
        return [q, k_proj, v_proj], {{}}
    elif mod_name == "entropy_gate":
        M_dim, N_dim, K_dim, R = 64, 128, 128, 32
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.1
        v_r = torch.randn(K_dim, R, device=device, dtype=dtype) * 0.1
        u_r = torch.randn(R, N_dim, device=device, dtype=dtype) * 0.1
        return [x_gemm, v_r, u_r, 16], {{}}
    elif mod_name == "fused_dropout_add":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        residual = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, residual], {{"p": 0.0, "training": False}}
    elif mod_name == "fused_norm_linear":
        x_gemm = torch.randn(B * 8, D * 8, device=device, dtype=dtype) * 0.1
        linear_w = torch.randn(D * 8, D * 8, device=device, dtype=dtype) * 0.1
        return [x_gemm, linear_w], {{}}
    elif mod_name == "fwht_transform":
        x = torch.randn(B, 1024, device=device, dtype=dtype) * 0.1
        return [x], {{}}
    elif mod_name == "rms_variance_match":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        target = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, target], {{}}
    elif mod_name == "low_rank_svd":
        M_dim, N_dim, K_dim, R = 64, 128, 128, 32
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.1
        v_r = torch.randn(R, K_dim, device=device, dtype=dtype) * 0.1
        u_r = torch.randn(N_dim, R, device=device, dtype=dtype) * 0.1
        alpha = torch.randn(R, device=device, dtype=dtype) * 0.1
        return [x_gemm, v_r, u_r, alpha], {{}}
    elif mod_name == "mamba_block":
        S_dim = 16
        u = torch.randn(B, D, device=device, dtype=dtype) * 0.1
        delta = torch.randn(B, D, device=device, dtype=dtype) * 0.1
        A = -torch.rand(D, S_dim, device=device, dtype=torch.float32)
        B_mat = torch.randn(B, S_dim, device=device, dtype=dtype) * 0.1
        C_mat = torch.randn(B, S_dim, device=device, dtype=dtype) * 0.1
        prev_state = torch.randn(B, D, S_dim, device=device, dtype=dtype) * 0.1
        return [u, delta, A, B_mat, C_mat, prev_state], {{}}
    elif mod_name == "mamba2_block":
        S_dim = 16
        u = torch.randn(B, D, N, device=device, dtype=dtype) * 0.1
        delta = torch.randn(B, D, N, device=device, dtype=dtype) * 0.1
        A = torch.randn(D, device=device, dtype=dtype) * 0.1
        B_mat = torch.randn(B, S_dim, N, device=device, dtype=dtype) * 0.1
        C_mat = torch.randn(B, S_dim, N, device=device, dtype=dtype) * 0.1
        return [u, delta, A, B_mat, C_mat], {{}}
    elif mod_name == "liquid_attention":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        delta = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        A = torch.randn(D, device=device, dtype=dtype) * 0.1
        return [x, delta, A], {{}}
    elif mod_name == "swiglu":
        x = torch.randn(B * 8, D * 8, device=device, dtype=dtype) * 0.1
        return [x], {{}}
    elif mod_name == "sliding_window":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        return [q, k, v], {{"window_size": 128, "num_sinks": 4}}
    elif mod_name == "split_head_norm":
        x = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        w = torch.randn(D, device=device, dtype=dtype) * 0.1
        return [x, w], {{"eps": 1e-5}}
    elif mod_name == "mamba_svd_flash":
        R_rank = 16
        u = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        svd_u = torch.randn(D, R_rank, device=device, dtype=dtype) * 0.1
        svd_vt = torch.randn(R_rank, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, N, D), 0.95, device=device, dtype=dtype)
        return [u, svd_u, svd_vt, decay], {{}}
    elif mod_name == "phantom":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        gate = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        return [x, gate], {{"threshold": 0.5}}
    elif mod_name == "spectra_v2":
        x = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        freq = torch.randn(D, device=device, dtype=dtype) * 0.1
        return [x, freq], {{"threshold": 0.05}}
    elif mod_name == "act_scale_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.05
        s_inv = torch.randn(K_dim, device=device, dtype=dtype) * 0.1
        w_mat = torch.randn(N_dim, K_dim, device=device, dtype=dtype) * 0.05
        return [x_gemm, s_inv, w_mat], {{}}
    elif mod_name == "adaptive_span":
        spans = torch.full((H,), 32.0, device=device, dtype=torch.float32)
        q_pos = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N)
        k_pos = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N)
        return [spans, q_pos, k_pos, 16.0], {{"is_causal": True}}
    elif mod_name == "entmax_bisect":
        scores = torch.randn(B, H, N, device=device, dtype=dtype) * 0.1
        return [scores], {{"n_iter": 15}}
    elif mod_name == "unified_rope_offset":
        x = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
        cos = torch.randn(N + 64, D // 2, device=device, dtype=torch.float32)
        sin = torch.randn(N + 64, D // 2, device=device, dtype=torch.float32)
        return [x, cos, sin, D], {{"offset": 0}}
    elif mod_name == "fp4_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype) * 0.05
        w_p = torch.randint(0, 255, (N_dim, K_dim // 2), device=device, dtype=torch.uint8)
        sc = torch.full((N_dim,), 0.01, device=device, dtype=dtype)
        return [x_gemm, w_p, sc], {{}}
    elif mod_name == "aegis_gate_norm":
        b0 = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        b1 = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        weights = torch.softmax(torch.randn(B, N, 2, device=device, dtype=dtype), dim=-1)
        return [b0, b1, weights], {{"eps": 1e-5}}
    elif mod_name == "gdn_delta_rule":
        N_delta = 64
        q = torch.randn(B, H, N_delta, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, H, N_delta, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, H, N_delta, D, device=device, dtype=dtype) * 0.1
        beta = torch.full((B, H, N_delta), 0.5, device=device, dtype=dtype)
        return [q, k, v, beta], {{}}
    elif mod_name == "hstu_attention":
        u = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, N, D, device=device, dtype=dtype) * 0.1
        decay = torch.full((B, N, D), 0.95, device=device, dtype=dtype)
        return [u, v, decay], {{"eps": 1e-6}}
    elif mod_name == "favor_plus":
        tot_toks, f_dim = B * N, D * 2
        proj = torch.randn(tot_toks, f_dim, device=device, dtype=dtype) * 0.1
        diag_sq = torch.rand(tot_toks, device=device, dtype=torch.float32) + 0.1
        norm_factor = 0.5
        return [proj, diag_sq, norm_factor], {{}}

    for param_name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            anno = str(param.annotation)
            if "Tensor" not in anno or "Optional" in anno:
                continue

        p_name = param_name.lower()
        anno = str(param.annotation)

        if any(k in p_name for k in ["q1", "k1", "q2", "k2"]):
            args.append(torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
        elif p_name in ["q", "k", "v", "query", "key", "value", "qk"]:
            args.append(torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
        elif p_name == "h_states":
            args.append(torch.randn(B, N, D * 4, device=device, dtype=dtype) * 0.1)
        elif p_name in ["weights", "route_weights"]:
            if "tpa" in mod_name or "weights" == p_name:
                args.append(torch.randn(B, N, device=device, dtype=dtype) * 0.1)
            else:
                args.append(torch.rand(B, H, N, device=device, dtype=dtype) * 0.1)
        elif p_name == "slopes":
            args.append(torch.randn(H, device=device, dtype=torch.float32) * 0.1)
        elif p_name == "gamma":
            args.append(torch.full((H,), 0.95, device=device, dtype=torch.float32))
        elif p_name == "hash_buckets":
            args.append(torch.randint(0, 8, (B, N), device=device, dtype=torch.int32))
        elif p_name == "cluster_centroids":
            args.append(torch.randn(B, H, 32, D, device=device, dtype=dtype) * 0.1)
        elif p_name == "cluster_indices":
            args.append(torch.randint(0, 32, (B, H, N), device=device, dtype=torch.int32))
        elif p_name == "spans":
            args.append(torch.full((B, H), 32.0, device=device, dtype=torch.float32))
        elif p_name in ["q_pos", "k_pos"]:
            args.append(torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N))
        elif p_name == "ramp_size":
            args.append(16.0)
        elif p_name in ["b0", "b1"]:
            args.append(torch.randn(B, N, D * 4, device=device, dtype=dtype) * 0.1)
        elif p_name == "w_packed":
            args.append(torch.randint(0, 255, (D * 8, (D * 8) // 2), device=device, dtype=torch.uint8))
        elif p_name in ["scales", "s_inv"]:
            args.append(torch.full((D * 8,), 0.01, device=device, dtype=dtype))
        elif p_name in ["weight", "weights_gemm", "w"]:
            args.append(torch.randn(D * 8, D * 8, device=device, dtype=dtype) * 0.05)
        elif p_name in ["x", "inp", "input"]:
            args.append(torch.randn(B * 8, D * 8, device=device, dtype=dtype) * 0.1)
        elif "int" in anno:
            if "topk" in p_name or p_name == "k": args.append(16)
            elif "window" in p_name: args.append(64)
            elif "group" in p_name: args.append(2)
            else: args.append(32)
        elif "float" in anno:
            args.append(0.1)
        elif "bool" in anno:
            args.append(False)
        elif "Tensor" in anno:
            args.append(torch.randn(B, N, D, device=device, dtype=dtype) * 0.1)
        else:
            args.append(torch.randn(B, N, D, device=device, dtype=dtype) * 0.1)

    return args, kwargs

def run_test(mod_name):
    device = torch.device("cuda:0")
    dtype = torch.float16
    cuda_dir = repo_root / "neuralops" / "core" / "cuda_kernels" / mod_name
    py_files = list(cuda_dir.glob("*autograd*.py"))
    if not py_files:
        return {{"status": "NO_AUTOGRAD_FILE"}}

    spec = importlib.util.spec_from_file_location(mod_name, py_files[0])
    py_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(py_mod)

    entry_name = ALIASES.get(mod_name, mod_name)
    fn = getattr(py_mod, entry_name, None)
    if fn is None or not callable(fn):
        return {{"status": "NO_ENTRYPOINT", "entry": entry_name}}
    assert callable(fn), f"Dynamically loaded kernel '{entry_name}' in {mod_name} must be callable"

    if hasattr(py_mod, "_get_module"):
        py_mod._get_module()

    ref_fn = (
        getattr(py_mod, f"{{entry_name}}_ref", None)
        or getattr(py_mod, f"{{entry_name}}_pytorch_ref", None)
        or getattr(py_mod, f"{{entry_name}}_pytorch", None)
        or getattr(py_mod, f"{{mod_name}}_ref", None)
        or getattr(py_mod, f"{{mod_name}}_pytorch_ref", None)
        or getattr(py_mod, f"{{mod_name}}_pytorch", None)
        or getattr(py_mod, f"{{mod_name}}_pytorch_reference", None)
    )
    if ref_fn is None:
        candidates = [
            getattr(py_mod, k)
            for k in dir(py_mod)
            if callable(getattr(py_mod, k))
            and any(term in k.lower() for term in ["_ref", "pytorch", "reference"])
            and not k.startswith("_")
            and k != entry_name
            and "triton" not in k.lower()
        ]
        if candidates:
            ref_fn = candidates[0]

    sig = inspect.signature(fn)
    args, kwargs = generate_args(mod_name, sig, device, dtype)

    out = fn(*args, **kwargs)
    cuda_ms = time_callable(lambda: fn(*args, **kwargs), warmup=2, iters=5)

    ref_ms = 0.0
    speedup = 1.0
    max_diff = float("nan")
    ref_out = None
    cuda_out = out

    if ref_fn:
        try:
            ref_args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
            ref_kwargs = {{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}}
            use_kwargs = True
            try:
                ref_out = ref_fn(*ref_args, **ref_kwargs)
            except (TypeError, Exception):
                ref_out = ref_fn(*ref_args)
                use_kwargs = False

            if use_kwargs:
                ref_ms = time_callable(lambda: ref_fn(*ref_args, **ref_kwargs), warmup=2, iters=5)
            else:
                ref_ms = time_callable(lambda: ref_fn(*ref_args), warmup=2, iters=5)
            speedup = ref_ms / max(cuda_ms, 1e-5)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            ref_ms = 0.0
            ref_out = None

    ref_ok = (ref_out is not None) and (ref_ms > 0.0)
    if not ref_ok:
        max_diff = float("nan")
        status = "REF_FAILED"
        reason = "reference did not execute or produced no output"
    elif cuda_out is None:
        max_diff = float("nan")
        status = "CUDA_FAILED"
        reason = "CUDA path raised or returned None"
    else:
        if isinstance(cuda_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
            assert cuda_out.data_ptr() != ref_out.data_ptr(), f"{{mod_name}}: cuda_out and ref_out alias the same storage"
        elif isinstance(cuda_out, (tuple, list)) and isinstance(ref_out, (tuple, list)):
            for o_elem, r_elem in zip(cuda_out, ref_out):
                if isinstance(o_elem, torch.Tensor) and isinstance(r_elem, torch.Tensor):
                    assert o_elem.data_ptr() != r_elem.data_ptr(), f"{{mod_name}}: cuda_out and ref_out tuple elements alias the same storage"

        if mod_name == "bitonic_topk":
            val_out = torch.gather(args[0], -1, cuda_out.long()).sort(dim=-1).values
            val_ref = torch.gather(args[0], -1, ref_out.long()).sort(dim=-1).values
            max_diff = (val_out.float() - val_ref.float()).abs().max().item()
        elif isinstance(cuda_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
            max_diff = (cuda_out.float() - ref_out.float()).abs().max().item()
        elif isinstance(cuda_out, (tuple, list)) and isinstance(ref_out, (tuple, list)):
            diffs = []
            for o_elem, r_elem in zip(cuda_out, ref_out):
                if isinstance(o_elem, torch.Tensor) and isinstance(r_elem, torch.Tensor):
                    diffs.append((o_elem.float() - r_elem.float()).abs().max().item())
            max_diff = max(diffs) if diffs else 0.0

        is_bit_identical = False
        if max_diff == 0.0:
            if isinstance(cuda_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
                if torch.equal(cuda_out, ref_out):
                    is_bit_identical = True
            elif isinstance(cuda_out, (tuple, list)) and isinstance(ref_out, (tuple, list)):
                if all(isinstance(o, torch.Tensor) and isinstance(r, torch.Tensor) and torch.equal(o, r) for o, r in zip(cuda_out, ref_out)):
                    is_bit_identical = True

        tolerance = 0.05
        if is_bit_identical:
            status = "DISPATCH_FAILURE"
            reason = f"{{mod_name}}: max_diff=0.0 but tensors are bit-identical — dispatch fallback suspected"
        elif math.isnan(max_diff):
            status = "FAIL_NAN"
            reason = "max_diff is NaN"
        elif max_diff >= tolerance:
            status = "FAIL"
            reason = f"max_diff={{max_diff:.4f}} >= {{tolerance}}"
        elif cuda_ms > ref_ms * 0.95:
            status = "PASS_SLOW"
            reason = f"speedup={{speedup:.2f}}x < 1.05x — slower than reference"
        elif max_diff >= 1e-3:
            status = "PASS_HIGH_DIFF"
            reason = f"max_diff={{max_diff:.4e}} >= 1e-3"
        else:
            status = "PASS_FAST"
            reason = f"speedup={{speedup:.2f}}x, max_diff={{max_diff:.4e}}"


    return {{
        "status": status,
        "reason": reason,
        "cuda_ms": round(cuda_ms, 4),
        "ref_ms": round(ref_ms, 4),
        "speedup": round(speedup, 2),
        "max_diff": max_diff,
    }}

if __name__ == "__main__":
    mod_name = sys.argv[1]
    res = run_test(mod_name)
    print("RESULT_JSON:" + json.dumps(res))
"""

ALIASES = {
    "act_scale_gemm": "act_scale_quant_gemm",
    "block_diag_gemm": "block_diag_matmul",
    "cosformer": "cosformer_linear_attention",
    "entropy_gate": "spectral_entropy_svd_gate",
    "flash_decoding": "flash_decoding_attention",
    "fused_ce": "fused_cross_entropy",
    "fused_kd": "fused_kd_loss",
    "fused_rope": "fused_rotary_emb",
    "gdn_scan": "chunked_delta_gate_scan",
    "hadamard_gemm": "hadamard_w4a16_gemm",
    "hgrn2_scan": "hgrn2_fused_bwd_scan",
    "kronecker_gemm": "kronecker_matmul",
    "latent_kv_attn": "latent_kv_cache_attention",
    "low_rank_svd": "fused_lowrank_gated_gemm",
    "lut_gemm": "lut_codebook_dequant_gemm",
    "moe_routing": "moe_dispatch_gather",
    "paged_kv_cache": "copy_blocks_cow",
    "quantized_gemm": "w4a16_gemm",
    "rpca_sparse_svd": "rpca_sparse_svd_gemm",
    "sliding_window": "sliding_window_attn",
    "softplus": "fused_softplus",
    "sparse_attention": "block_sparse_attn",
    "spectraformer_attention": "spectraformer",
    "state_update": "recurrent_linear_step",
}


def main():
    """Execute main routine."""
    parser = argparse.ArgumentParser(description="Subprocess-Isolated Master Benchmark for CUDA Modules")
    parser.add_argument("--classify-strict", action="store_true", help="Apply strict PASS/FAIL classification")
    parser.add_argument(
        "--modules", type=str, default=None, help="Limit number of modules (e.g. 3) or comma-separated names"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of modules to run")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    print("=" * 115)
    print("SUBPROCESS-ISOLATED MASTER BENCHMARK ACROSS ALL 108 NATIVE CUDA MODULES (PHASE 0 HARNESS)")
    print("=" * 115)

    cuda_dir = repo_root / "neuralops" / "core" / "cuda_kernels"
    subdirs = sorted([d for d in cuda_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))])

    if args.modules:
        if args.modules.isdigit():
            subdirs = subdirs[: int(args.modules)]
        else:
            mod_filter = {m.strip() for m in args.modules.split(",")}
            subdirs = [d for d in subdirs if d.name in mod_filter]
    elif args.limit:
        subdirs = subdirs[: args.limit]

    artifact_dir = repo_root / "scratch"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    worker_path = artifact_dir / "_isolated_worker.py"
    worker_content = WORKER_CODE.format(
        repo_root=str(repo_root),
        aliases_json=json.dumps(ALIASES),
    )
    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(worker_content)

    results = []
    stats = {}

    print(
        f"{'Idx':<4} | {'Module Name':<26} | {'CUDA (ms)':<10} | {'Ref (ms)':<10} | {'Speedup':<9} | {'Diff':<10} | {'Classified Status'}"
    )
    print("-" * 115)

    python_bin = sys.executable

    for idx, d in enumerate(subdirs, 1):
        mod_name = d.name
        cmd = [python_bin, str(worker_path), mod_name]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stdout = p.stdout
            stderr = p.stderr

            if "RESULT_JSON:" in stdout:
                res_str = stdout.split("RESULT_JSON:")[1].strip().split("\n")[0]
                res_data = json.loads(res_str)
                cls_status = res_data.get("status", "FAIL")
                reason = res_data.get("reason", "")
                cuda_ms = res_data.get("cuda_ms", 0.0)
                ref_ms = res_data.get("ref_ms", 0.0)
                speedup = res_data.get("speedup", 1.0)
                diff = res_data.get("max_diff")

                stats[cls_status] = stats.get(cls_status, 0) + 1

                ref_str = f"{ref_ms:.4f}" if (ref_ms > 0 and cls_status != "REF_FAILED") else "N/A"
                sp_str = f"{speedup:.2f}x" if (ref_ms > 0 and cls_status != "REF_FAILED") else "1.00x"
                diff_str = f"{diff:.2e}" if (diff is not None and str(diff) != "nan" and diff != 999.0) else "N/A"
                print(
                    f"{idx:<4} | {mod_name:<26} | {cuda_ms:<10.4f} | {ref_str:<10} | {sp_str:<9} | {diff_str:<10} | {cls_status}",
                    flush=True,
                )
                results.append({"module": mod_name, **res_data, "status": cls_status, "reason": reason})
            else:
                err = (stderr or stdout).strip().split("\n")[-1][:50]
                stats["CRASH"] = stats.get("CRASH", 0) + 1
                print(
                    f"{idx:<4} | {mod_name:<26} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | CRASH: {err}",
                    flush=True,
                )
                results.append({"module": mod_name, "status": "CRASH", "error": err})
        except subprocess.TimeoutExpired:
            stats["TIMEOUT"] = stats.get("TIMEOUT", 0) + 1
            print(
                f"{idx:<4} | {mod_name:<26} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | TIMEOUT (120s)", flush=True
            )
            results.append({"module": mod_name, "status": "TIMEOUT"})
        except Exception as e:
            stats["CRASH"] = stats.get("CRASH", 0) + 1
            print(
                f"{idx:<4} | {mod_name:<26} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | SUBPROC ERR: {str(e)[:30]}",
                flush=True,
            )
            results.append({"module": mod_name, "status": "CRASH", "error": str(e)})

    print("=" * 115, flush=True)
    print("Totals (108 modules):", flush=True)
    print(f"  PASS_FAST      (>=1.0x, diff<1e-3) : {stats.get('PASS_FAST', 0)}", flush=True)
    print(f"  PASS_SLOW      (<1.0x)             : {stats.get('PASS_SLOW', 0)}   <- must be 0 for release", flush=True)
    print(
        f"  PASS_HIGH_DIFF (diff>=1e-3)        : {stats.get('PASS_HIGH_DIFF', 0)}   <- must be 0 for release",
        flush=True,
    )
    print(f"  FAIL           (diff>=tol)         : {stats.get('FAIL', 0)}", flush=True)
    print(f"  REF_FAILED     (ref missing)       : {stats.get('REF_FAILED', 0)}   <- must be 0 for release", flush=True)
    print(f"  CUDA_FAILED    (kernel raised)     : {stats.get('CUDA_FAILED', 0)}", flush=True)
    print(f"  DISPATCH_FAIL  (bit-identical ref) : {stats.get('DISPATCH_FAILURE', 0)}", flush=True)
    print(f"  CRASH          (import/exec error) : {stats.get('CRASH', 0)}   <- must be 0 for release", flush=True)
    print("=" * 115, flush=True)

    out_file = artifact_dir / "master_108_cuda_modules_benchmark.json"
    if args.output:
        custom_out = Path(args.output)
        if not custom_out.is_absolute():
            custom_out = repo_root / custom_out
        custom_out.parent.mkdir(parents=True, exist_ok=True)
        with open(custom_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results also saved to: {custom_out}", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Master benchmark results saved to: {out_file}", flush=True)

    csv_file = artifact_dir / "master_108_cuda_modules_benchmark.csv"
    import csv

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Module", "CUDA (ms)", "Ref (ms)", "Speedup", "Max Abs Diff", "Status", "Reason"])
        for r in results:
            mod = r.get("module", "")
            stat = r.get("status", "")
            reason = r.get("reason", "")
            c_ms = r.get("cuda_ms", "")
            r_ms = r.get("ref_ms", "")
            sp = f"{r.get('speedup', '')}x" if r.get("speedup") else ""
            diff = r.get("max_diff", "")
            writer.writerow([mod, c_ms, r_ms, sp, diff, stat, reason])
    print(f"Master benchmark CSV saved to: {csv_file}", flush=True)


if __name__ == "__main__":
    main()
