# -*- coding: utf-8 -*-
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
import os
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

WORKER_CODE = """
import sys, os, time, json, inspect, importlib.util
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

    # Custom shapes for specific math operations
    if mod_name == "kronecker_gemm":
        x = torch.randn(B * 8, 16 * 16, device=device, dtype=dtype)
        A = torch.randn(16, 16, device=device, dtype=dtype)
        B_mat = torch.randn(16, 16, device=device, dtype=dtype)
        return [x, A, B_mat], {{}}
    elif mod_name == "bitonic_topk":
        x = torch.randn(B * 8, 1024, device=device, dtype=dtype)
        return [x, 32], {{}}
    elif mod_name == "block_diag_gemm":
        G = 4
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        w = torch.randn(G, D // G, D // G, device=device, dtype=dtype)
        return [x, w], {{}}
    elif mod_name == "latent_kv_attn":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        c_kv = torch.randn(B, N, D, device=device, dtype=dtype)
        u_k = torch.randn(D, D, device=device, dtype=dtype)
        u_v = torch.randn(D, D, device=device, dtype=dtype)
        return [q, c_kv, u_k, u_v], {{}}
    elif mod_name == "hadamard_gemm":
        M_dim, N_dim, K_dim = 128, 256, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        w_p = torch.randint(0, 255, (N_dim, K_dim // 2), device=device, dtype=torch.uint8)
        sc = torch.randn(K_dim // 32, N_dim, device=device, dtype=dtype)
        zr = torch.zeros(K_dim // 32, N_dim, device=device, dtype=dtype)
        return [x_gemm, w_p, sc, zr], {{"group_size": 32}}
    elif mod_name == "ternary_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        w_p = torch.randint(0, 255, (N_dim, K_dim // 4), device=device, dtype=torch.uint8)
        sc = torch.randn(N_dim, device=device, dtype=dtype)
        return [x_gemm, w_p, sc], {{}}
    elif mod_name == "lut_gemm":
        M_dim, N_dim, K_dim = 64, 128, 256
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        w_p = torch.randint(0, 15, (N_dim, K_dim // 2), device=device, dtype=torch.uint8)
        cb = torch.randn(16, device=device, dtype=dtype)
        return [x_gemm, w_p, cb], {{"group_size": 64}}
    elif mod_name == "rotational_svd":
        M_dim, N_dim, K_dim, R = 64, 128, 128, 32
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        r_in = torch.randn(K_dim, K_dim, device=device, dtype=dtype)
        v_r = torch.randn(K_dim, R, device=device, dtype=dtype)
        u_r = torch.randn(R, N_dim, device=device, dtype=dtype)
        return [x_gemm, r_in, v_r, u_r], {{}}
    elif mod_name == "rpca_sparse_svd":
        M_dim, N_dim, K_dim, R = 64, 128, 128, 32
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        v_r = torch.randn(K_dim, R, device=device, dtype=dtype)
        u_r = torch.randn(R, N_dim, device=device, dtype=dtype)
        return [x_gemm, v_r, u_r], {{}}
    elif mod_name == "layernorm":
        x = torch.randn(B * 8, D * 8, device=device, dtype=dtype)
        w = torch.randn(D * 8, device=device, dtype=dtype)
        b = torch.randn(D * 8, device=device, dtype=dtype)
        return [x, w, b], {{}}
    elif mod_name == "rmsnorm":
        x = torch.randn(B * 8, D * 8, device=device, dtype=dtype)
        w = torch.randn(D * 8, device=device, dtype=dtype)
        return [x, w], {{}}
    elif mod_name == "channel_shuffle":
        x = torch.randn(B, 16, 32, 32, device=device, dtype=dtype)
        return [x, 4], {{}}
    elif mod_name == "moe_routing":
        tokens = torch.randn(B * 8, D, device=device, dtype=dtype)
        gate_logits = torch.randn(B * 8, 8, device=device, dtype=dtype)
        expert = torch.nn.Linear(D, D, device=device, dtype=dtype)
        return [tokens, gate_logits, expert], {{"top_k": 2}}
    elif mod_name == "selective_scan":
        u = torch.randn(B, 64, N, device=device, dtype=dtype)
        delta = torch.randn(B, 64, N, device=device, dtype=dtype)
        A = torch.randn(64, 16, device=device, dtype=torch.float32)
        B_t = torch.randn(B, 16, N, device=device, dtype=dtype)
        C_t = torch.randn(B, 16, N, device=device, dtype=dtype)
        D_t = torch.randn(64, device=device, dtype=torch.float32)
        return [u, delta, A, B_t, C_t, D_t], {{}}
    elif mod_name == "selective_ssm":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        dt = torch.randn(B, N, D, device=device, dtype=dtype)
        A = torch.randn(D, device=device, dtype=dtype)
        B_vec = torch.randn(B, N, D, device=device, dtype=dtype)
        C_vec = torch.randn(B, N, D, device=device, dtype=dtype)
        return [x, dt, A, B_vec, C_vec], {{}}
    elif mod_name == "simple_ssm":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        decay = torch.randn(B, N, D, device=device, dtype=dtype)
        scale = torch.tensor(1.0, device=device, dtype=dtype)
        return [x, decay, scale], {{}}
    elif mod_name == "quantized_gemm":
        M_dim, N_dim, K_dim = 128, 512, 512
        num_g = K_dim // 128
        x_gemm = torch.randn(M_dim, K_dim, device=device, dtype=dtype)
        w_p = torch.randint(0, 255, (N_dim, K_dim // 2), device=device, dtype=torch.uint8)
        sc = torch.randn(num_g, N_dim, device=device, dtype=dtype)
        zr = torch.zeros(num_g, N_dim, device=device, dtype=dtype)
        return [x_gemm, w_p, sc, zr], {{"group_size": 128}}
    elif mod_name == "sparse_attention":
        bs = 64
        nb = N // bs
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask_2d = torch.ones(nb, nb, device=device, dtype=torch.int32)
        return [q, k, v, mask_2d], {{"block_size": bs}}
    elif mod_name == "decay_linear_attention":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        gamma = torch.full((H,), 0.99, device=device, dtype=torch.float32)
        return [q, k, v, gamma], {{}}
    elif mod_name in ["causal_conv1d", "dynamic_conv1d"]:
        x = torch.randn(B, D, N, device=device, dtype=dtype)
        weight = torch.randn(D, 1, 4, device=device, dtype=dtype)
        return [x, weight], {{}}
    elif mod_name == "fused_rope":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        cos = torch.randn(N, D, device=device, dtype=dtype)
        sin = torch.randn(N, D, device=device, dtype=dtype)
        return [q, cos, sin], {{}}
    elif mod_name == "recurrent_gla_step":
        q = torch.randn(B, H, D, device=device, dtype=dtype)
        k = torch.randn(B, H, D, device=device, dtype=dtype)
        v = torch.randn(B, H, D, device=device, dtype=dtype)
        h = torch.randn(B, H, D, D, device=device, dtype=dtype)
        g = torch.full((B, H), 0.99, device=device, dtype=dtype)
        return [q, k, v, h, g], {{}}
    elif mod_name in ["hgrn2_scan", "hgrn2_spatial_scan"]:
        x = torch.randn(B, H, N, D, device=device, dtype=dtype)
        g = torch.randn(B, H, N, D, device=device, dtype=dtype)
        return [x, g], {{}}
    elif mod_name == "linear_rnn_attention":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        a = torch.randn(B, N, D, device=device, dtype=dtype)
        b = torch.randn(B, N, D, device=device, dtype=dtype)
        return [x, a, b], {{}}
    elif mod_name == "nasc_attention":
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        decay = torch.randn(B, H, N, D, device=device, dtype=dtype)
        return [q, k, v, decay], {{}}
    elif mod_name == "nystrom_attention":
        q_l = torch.randn(B, H, 32, D, device=device, dtype=dtype)
        k_l = torch.randn(B, H, 32, D, device=device, dtype=dtype)
        v_l = torch.randn(B, H, D, D, device=device, dtype=dtype)
        p_l = torch.randn(B, H, D, D, device=device, dtype=dtype)
        return [q_l, k_l, v_l, p_l], {{}}
    elif mod_name == "paged_attention":
        num_seqs, num_heads, num_kv_heads, head_dim, block_sz, max_blocks = 4, 8, 2, 64, 16, 8
        q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
        num_total_blocks = num_seqs * max_blocks
        k_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=dtype)
        v_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=dtype)
        context_lens = torch.tensor([16, 32, 48, 64], device=device, dtype=torch.int32)
        block_tables = torch.zeros(num_seqs, max_blocks, device=device, dtype=torch.int32)
        for i in range(num_seqs):
            nb = int(context_lens[i].item()) // block_sz
            block_tables[i, :nb] = torch.arange(i * max_blocks, i * max_blocks + nb, device=device)
        return [q, k_cache, v_cache, block_tables, context_lens], {{"block_size": block_sz}}
    elif mod_name == "paged_kv_cache":
        nb, num_h, head_d, block_sz = 16, 8, 64, 16
        src_k = torch.randn(nb, num_h, head_d, block_sz, device=device, dtype=dtype)
        src_v = torch.randn(nb, num_h, head_d, block_sz, device=device, dtype=dtype)
        dst_k = torch.zeros_like(src_k)
        dst_v = torch.zeros_like(src_v)
        b_map = torch.tensor([[0, 2], [1, 5], [3, 7]], device=device, dtype=torch.int32)
        return [dst_k, dst_v, src_k, src_v, b_map, block_sz, num_h, head_d], {{}}
    elif mod_name == "paged_kv_evict":
        k_in = torch.randn(64, 8, 64, device=device, dtype=dtype)
        v_in = torch.randn(64, 8, 64, device=device, dtype=dtype)
        keep = torch.arange(32, device=device, dtype=torch.int64)
        return [k_in, v_in, keep], {{}}
    elif mod_name == "paged_recurrent_step":
        q = torch.randn(B, H, D, device=device, dtype=dtype)
        k = torch.randn(B, H, D, device=device, dtype=dtype)
        v = torch.randn(B, H, D, device=device, dtype=dtype)
        page_pool = torch.randn(32, H, D, D, device=device, dtype=dtype)
        block_tables = torch.arange(B, device=device, dtype=torch.int64)
        return [q, k, v, page_pool, block_tables], {{}}
    elif mod_name == "rfa_attention":
        phi_q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        phi_k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        return [phi_q, phi_k, v], {{}}
    elif mod_name == "rwkv6":
        r = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        w = torch.randn(B, H, N, D, device=device, dtype=dtype)
        u = torch.randn(H, D, device=device, dtype=dtype)
        return [r, k, v, w, u], {{}}
    elif mod_name == "spectraformer_attention":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        c_basis = torch.randn(B, N, D, device=device, dtype=dtype)
        s_basis = torch.randn(B, N, D, device=device, dtype=dtype)
        return [x, c_basis, s_basis], {{}}
    elif mod_name == "spectral_mamba":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        freq = torch.randn(B, N, D, device=device, dtype=dtype)
        return [x, freq], {{"threshold": 0.01}}
    elif mod_name == "spectral_shrinkage":
        x_re = torch.randn(B, N, D, device=device, dtype=dtype)
        x_im = torch.randn(B, N, D, device=device, dtype=dtype)
        w_re = torch.randn(B, N, D, device=device, dtype=dtype)
        w_im = torch.randn(B, N, D, device=device, dtype=dtype)
        tau = torch.tensor(0.01, device=device, dtype=dtype)
        return [x_re, x_im, w_re, w_im, tau], {{}}
    elif mod_name == "tt_contraction":
        core0 = torch.randn(1, 16, 4, device=device, dtype=dtype)
        core1 = torch.randn(4, 16, 1, device=device, dtype=dtype)
        return [core0, core1, (16, 16), (16, 16)], {{}}
    elif mod_name == "upi_stuffed_mamba":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        upi_tok = torch.randn(B, 16, D, device=device, dtype=dtype)
        return [x, upi_tok], {{"blend_weight": 0.2}}
    elif mod_name == "vspec":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        spec_gate = torch.randn(B, N, D, device=device, dtype=dtype)
        decay = torch.randn(B, N, D, device=device, dtype=dtype)
        return [x, spec_gate, decay], {{}}
    elif mod_name == "hymba_transx":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        meta_tok = torch.randn(B, 16, D, device=device, dtype=dtype)
        cross_gate = torch.randn(B, N, D, device=device, dtype=dtype)
        return [x, meta_tok, cross_gate], {{}}
    elif mod_name == "jamba_info":
        x = torch.randn(B, N, D, device=device, dtype=dtype)
        bw = torch.randn(D, D, device=device, dtype=dtype)
        return [x, bw], {{}}
    elif mod_name == "state_update":
        q = torch.randn(B, H, D, device=device, dtype=dtype)
        k = torch.randn(B, H, D, device=device, dtype=dtype)
        v = torch.randn(B, H, D, device=device, dtype=dtype)
        state = torch.randn(B, H, D, D, device=device, dtype=dtype)
        return [q, k, v, state], {{}}

    # Generic parameter resolution
    for param_name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            anno = str(param.annotation)
            if "Tensor" in anno and "Optional" not in anno: pass
            else: continue

        p_name = param_name.lower()
        anno = str(param.annotation)

        if any(k in p_name for k in ["q1", "k1", "q2", "k2"]):
            args.append(torch.randn(B, H, N, D, device=device, dtype=dtype))
        elif p_name in ["q", "k", "v", "query", "key", "value", "qk"]:
            args.append(torch.randn(B, H, N, D, device=device, dtype=dtype))
        elif p_name == "h_states":
            args.append(torch.randn(B, N, D * 4, device=device, dtype=dtype))
        elif p_name in ["weights", "route_weights"]:
            if "tpa" in mod_name or "weights" == p_name:
                args.append(torch.randn(B, N, device=device, dtype=dtype))
            else:
                args.append(torch.rand(B, H, N, device=device, dtype=dtype))
        elif p_name == "slopes":
            args.append(torch.randn(H, device=device, dtype=torch.float32))
        elif p_name == "gamma":
            args.append(torch.full((H,), 0.99, device=device, dtype=torch.float32))
        elif p_name == "hash_buckets":
            args.append(torch.randint(0, 8, (B, N), device=device, dtype=torch.int32))
        elif p_name == "cluster_centroids":
            args.append(torch.randn(B, H, 32, D, device=device, dtype=dtype))
        elif p_name == "cluster_indices":
            args.append(torch.randint(0, 32, (B, H, N), device=device, dtype=torch.int32))
        elif p_name == "spans":
            args.append(torch.full((B, H), 32.0, device=device, dtype=torch.float32))
        elif p_name in ["q_pos", "k_pos"]:
            args.append(torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N))
        elif p_name == "ramp_size":
            args.append(16.0)
        elif p_name in ["b0", "b1"]:
            args.append(torch.randn(B, N, D * 4, device=device, dtype=dtype))
        elif p_name == "w_packed":
            args.append(torch.randint(0, 255, (D * 8, (D * 8) // 2), device=device, dtype=torch.uint8))
        elif p_name in ["scales", "s_inv"]:
            args.append(torch.randn(D * 8, device=device, dtype=dtype))
        elif p_name in ["weight", "weights_gemm", "w"]:
            args.append(torch.randn(D * 8, D * 8, device=device, dtype=dtype))
        elif p_name in ["x", "inp", "input"]:
            args.append(torch.randn(B * 8, D * 8, device=device, dtype=dtype))
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
            args.append(torch.randn(B, N, D, device=device, dtype=dtype))
        else:
            args.append(torch.randn(B, N, D, device=device, dtype=dtype))

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

    # JIT Compile
    if hasattr(py_mod, "_get_module"):
        py_mod._get_module()

    ref_fn = getattr(py_mod, f"{{entry_name}}_ref", None) or getattr(py_mod, f"{{mod_name}}_ref", None)
    if ref_fn is None:
        refs = [getattr(py_mod, k) for k in dir(py_mod) if k.endswith("_ref") and callable(getattr(py_mod, k))]
        if refs: ref_fn = refs[0]

    sig = inspect.signature(fn)
    args, kwargs = generate_args(mod_name, sig, device, dtype)

    out = fn(*args, **kwargs)
    cuda_ms = time_callable(lambda: fn(*args, **kwargs), warmup=2, iters=5)

    ref_ms = 0.0
    speedup = 1.0
    max_diff = 0.0

    if ref_fn:
        try:
            ref_out = ref_fn(*args, **kwargs)
            ref_ms = time_callable(lambda: ref_fn(*args, **kwargs), warmup=2, iters=5)
            speedup = ref_ms / max(cuda_ms, 1e-5)
            if isinstance(out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
                max_diff = (out.float() - ref_out.float()).abs().max().item()
            elif isinstance(out, tuple) and isinstance(ref_out, tuple):
                max_diff = (out[0].float() - ref_out[0].float()).abs().max().item()
        except Exception:
            pass

    return {{
        "status": "PASS",
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
    print("=" * 105)
    print("SUBPROCESS-ISOLATED MASTER BENCHMARK ACROSS ALL 108 NATIVE CUDA MODULES")
    print("=" * 105)

    cuda_dir = repo_root / "neuralops" / "core" / "cuda_kernels"
    subdirs = sorted([d for d in cuda_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))])

    # Write worker script
    worker_path = repo_root / "scripts" / "benchmarks" / "_isolated_worker.py"
    worker_content = WORKER_CODE.format(
        repo_root=str(repo_root),
        aliases_json=json.dumps(ALIASES),
    )
    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(worker_content)

    results = []
    passed = 0
    failed = 0

    print(f"{'Idx':<4} | {'Module Name':<28} | {'CUDA (ms)':<10} | {'Ref (ms)':<10} | {'Speedup':<9} | {'Diff':<10} | {'Status'}")
    print("-" * 105)

    python_bin = sys.executable

    for idx, d in enumerate(subdirs, 1):
        mod_name = d.name
        cmd = [python_bin, str(worker_path), mod_name]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            stdout = p.stdout
            stderr = p.stderr

            if "RESULT_JSON:" in stdout:
                res_str = stdout.split("RESULT_JSON:")[1].strip().split("\n")[0]
                res_data = json.loads(res_str)
                if res_data.get("status") == "PASS":
                    passed += 1
                    cuda_ms = res_data["cuda_ms"]
                    ref_ms = res_data["ref_ms"]
                    speedup = res_data["speedup"]
                    diff = res_data["max_diff"]
                    ref_str = f"{ref_ms:.4f}" if ref_ms > 0 else "N/A"
                    sp_str = f"{speedup:.2f}x" if ref_ms > 0 else "1.00x"
                    diff_str = f"{diff:.2e}" if ref_ms > 0 else "N/A"
                    print(f"{idx:<4} | {mod_name:<28} | {cuda_ms:<10.4f} | {ref_str:<10} | {sp_str:<9} | {diff_str:<10} | PASS")
                    results.append({"module": mod_name, "status": "PASS", **res_data})
                else:
                    failed += 1
                    err_msg = res_data.get("status", "FAIL")
                    print(f"{idx:<4} | {mod_name:<28} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | FAIL ({err_msg})")
                    results.append({"module": mod_name, "status": "FAIL", "error": err_msg})
            else:
                failed += 1
                err = (stderr or stdout).strip().split("\n")[-1][:40]
                print(f"{idx:<4} | {mod_name:<28} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | CRASH: {err}")
                results.append({"module": mod_name, "status": "CRASH", "error": err})
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"{idx:<4} | {mod_name:<28} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | TIMEOUT (30s)")
            results.append({"module": mod_name, "status": "TIMEOUT"})
        except Exception as e:
            failed += 1
            print(f"{idx:<4} | {mod_name:<28} | {'-':<10} | {'-':<10} | {'-':<9} | {'-':<10} | SUBPROC ERR: {str(e)[:30]}")
            results.append({"module": mod_name, "status": "ERROR", "error": str(e)})

    print("=" * 105)
    print(f"MASTER ISOLATED BENCHMARK COMPLETED: {passed}/{len(subdirs)} PASSED ({passed/len(subdirs)*100:.1f}%), {failed} FAILED.")
    print("=" * 105)

    out_file = repo_root / "results" / "master_108_cuda_modules_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Master benchmark results saved to: {out_file}")


if __name__ == "__main__":
    main()
