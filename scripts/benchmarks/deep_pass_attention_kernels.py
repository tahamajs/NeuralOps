"""Deep pass attention kernels module.

Author: Taha Majlesi
"""

import sys
import time
import math
import importlib.util
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

targets = [
    "flash_gqa",
    "latent_kv_attn",
    "sparse_attention",
    "differential_attention",
    "local_attention",
    "routing_attention",
    "flash_decode_stage2",
    "flash_decoding",
    "varlen_attention",
    "cosformer",
    "alibi_attention",
    "bigbird",
    "cross_attention",
    "galerkin_attention",
    "hydra_attention",
    "hymba_transx",
    "longformer",
    "reformer",
    "tpa_attention",
    "upi_stuffed_mamba",
    "hstu_attention",
    "linformer_attention",
    "favor_plus",
    "nystrom_attention",
    "cluster_attention",
    "multi_scale_retention",
    "fms_scan",
    "adaptive_span",
]


def benchmark_fn(fn, warmup=5, iters=30):
    """Execute benchmark fn routine."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def run_deep_pass():
    """Execute run deep pass routine."""
    device = torch.device("cuda:0")
    print("=" * 110)
    print("DEEP VERIFICATION PASS: ATTENTION CUDA MODULES (SM 8.9 RTX 4070)")
    print("=" * 110)
    print("idx | module                    | cuda_ms    | ref_ms     | speedup   | max_diff     | nan_check | status")
    print("-" * 110)

    all_pass = True
    results = []

    B, H, N, D = 4, 8, 512, 64

    for idx, name in enumerate(targets, 1):
        try:
            p = repo_root / "neuralops" / "core" / "cuda_kernels" / name
            py_files = list(p.glob("*autograd*.py"))
            if not py_files:
                print(f"{idx:<3} | {name:<25} | NO AUTOGRAD FILE")
                continue

            spec = importlib.util.spec_from_file_location(name, py_files[0])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Generate inputs & callables
            cuda_fn = None
            ref_fn = None
            if name == "flash_gqa":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                scale = 1.0 / math.sqrt(D)
                cuda_fn = lambda: mod.flash_gqa(q, k, v, False, scale)
                ref_fn = lambda: mod.flash_gqa_pytorch_ref(q, k, v, False, scale)

            elif name == "latent_kv_attn":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                c_kv = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                u_k = torch.randn(D, D, device=device, dtype=torch.float16) * 0.1
                u_v = torch.randn(D, D, device=device, dtype=torch.float16) * 0.1
                scale = 1.0 / math.sqrt(D)
                cuda_fn = lambda: mod.latent_kv_cache_attention(q, c_kv, u_k, u_v, scale)
                ref_fn = lambda: mod.latent_kv_cache_attention_pytorch_ref(q, c_kv, u_k, u_v, scale)

            elif name == "sparse_attention":
                bs = 16
                nb = N // bs
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 0.1
                bmask = (torch.rand(nb, nb, device=device) > 0.5).to(torch.int32)
                scale = 1.0 / math.sqrt(D)
                cuda_fn = lambda: mod.block_sparse_attn(q, k, v, bmask, bs, scale)
                ref_fn = lambda: mod.block_sparse_attn_pytorch_ref(q, k, v, bmask, bs, scale)

            elif name == "differential_attention":
                q1 = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k1 = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                q2 = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k2 = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.differential_attention(q1, k1, q2, k2, v, 0.8)
                ref_fn = lambda: mod.differential_attention_ref(q1, k1, q2, k2, v, 0.8)

            elif name == "local_attention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.local_attention(q, k, v, 64)
                ref_fn = lambda: mod.local_attention_ref(q, k, v, 64)

            elif name == "routing_attention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                rw = torch.rand(B, H, N, device=device, dtype=torch.float16)
                cuda_fn = lambda: mod.routing_attention(q, k, v, rw, 32)
                ref_fn = lambda: mod.routing_attention_ref(q, k, v, rw, 32)

            elif name == "flash_decode_stage2":
                num_splits = 8
                mid_out = torch.randn(B, H, num_splits, D, device=device, dtype=torch.float16) * 0.1
                mid_lse = torch.randn(B, H, num_splits, device=device, dtype=torch.float32)
                cuda_fn = lambda: mod.flash_decode_stage2(mid_out, mid_lse)
                ref_fn = lambda: mod.flash_decode_stage2_pytorch_ref(mid_out, mid_lse)

            elif name == "flash_decoding":
                q = torch.randn(B, H, 1, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.flash_decoding_attention(q, k, v)
                ref_fn = lambda: mod.flash_decoding_attention_pytorch_ref(q, k, v)

            elif name == "varlen_attention":
                q = torch.randn(N, H, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(N, H, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(N, H, D, device=device, dtype=torch.float16) * 0.1
                cu = torch.tensor([0, 128, 256, 384, 512], device=device, dtype=torch.int32)
                cuda_fn = lambda: mod.varlen_attention(q, k, v, cu, cu, 128, 128, False)
                ref_fn = lambda: mod.varlen_attention_pytorch_ref(q, k, v, cu, cu, 128, 128, False)

            elif name == "cosformer":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.cosformer_linear_attention(q, k, v, False)
                ref_fn = lambda: mod.cosformer_linear_attention_pytorch_ref(q, k, v, False)

            elif name == "alibi_attention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                slopes = torch.randn(H, device=device, dtype=torch.float32) * 0.1
                cuda_fn = lambda: mod.alibi_attention(q, k, v, slopes, False)
                ref_fn = lambda: mod.alibi_attention_pytorch_ref(q, k, v, slopes, False)

            elif name == "bigbird":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.bigbird(q, k, v, 32, 16, 2)
                ref_fn = lambda: mod.bigbird_ref(q, k, v, 32, 16, 2)

            elif name == "cross_attention":
                q = torch.randn(B, H, 128, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.cross_attention(q, k, v)
                ref_fn = lambda: mod.cross_attention_pytorch_ref(q, k, v)

            elif name == "galerkin_attention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.galerkin_attention(q, k, v, 0.1)
                ref_fn = lambda: mod.galerkin_attention_ref(q, k, v, 0.1)

            elif name == "hydra_attention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.hydra_attention(q, k, v)
                ref_fn = lambda: mod.hydra_attention_ref(q, k, v)

            elif name == "hymba_transx":
                x = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                meta = torch.randn(B, 32, D, device=device, dtype=torch.float16) * 0.1
                gate = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.hymba_transx(x, meta, gate)
                ref_fn = lambda: mod.hymba_transx_ref(x, meta, gate)

            elif name == "longformer":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.longformer(q, k, v, 64, 2)
                ref_fn = lambda: mod.longformer_ref(q, k, v, 64, 2)

            elif name == "reformer":
                qk = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                hb = torch.randint(0, 8, (B, N), device=device, dtype=torch.int32)
                cuda_fn = lambda: mod.reformer(qk, v, hb)
                ref_fn = lambda: mod.reformer_ref(qk, v, hb)

            elif name == "tpa_attention":
                h_states = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                weights = torch.randn(B, N, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.tpa_attention(h_states, weights)
                ref_fn = lambda: mod.tpa_attention_ref(h_states, weights)

            elif name == "upi_stuffed_mamba":
                x = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                upi = torch.randn(B, 16, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.upi_stuffed_mamba(x, upi, 0.5)
                ref_fn = lambda: mod.upi_stuffed_mamba_ref(x, upi, 0.5)

            elif name == "hstu_attention":
                u = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, N, D, device=device, dtype=torch.float16) * 0.1
                decay = torch.sigmoid(torch.randn(B, N, D, device=device, dtype=torch.float16)) * 0.95
                cuda_fn = lambda: mod.hstu_attention(u, v, decay, 1e-4)
                ref_fn = lambda: mod.hstu_attention_ref(u, v, decay, 1e-4)

            elif name == "linformer_attention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                kp = torch.randn(B, H, 64, D, device=device, dtype=torch.float16) * 0.1
                vp = torch.randn(B, H, 64, D, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.linformer_attention(q, kp, vp)
                ref_fn = lambda: mod.linformer_attention_ref(q, kp, vp)

            elif name == "favor_plus":
                proj = torch.randn(B * N, 64, device=device, dtype=torch.float16) * 0.1
                diag_sq = torch.sum(proj**2, dim=-1) * 0.5
                cuda_fn = lambda: mod.favor_plus(proj, diag_sq, 1.0)
                ref_fn = lambda: mod.favor_plus_pytorch_ref(proj, diag_sq, 1.0)

            elif name == "nystrom_attention":
                M = 32
                q_l = torch.randn(B, H, N, M, device=device, dtype=torch.float16) * 0.1
                k_l = torch.randn(B, H, M, D, device=device, dtype=torch.float16) * 0.1
                v_l = torch.randn(B, H, M, D, device=device, dtype=torch.float16) * 0.1
                pinv = torch.randn(B, H, M, M, device=device, dtype=torch.float16) * 0.1
                cuda_fn = lambda: mod.nystrom_attention(q_l, k_l, v_l, pinv)
                ref_fn = lambda: mod.nystrom_attention_ref(q_l, k_l, v_l, pinv)

            elif name == "cluster_attention":
                K_c = 32
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                cen = torch.randn(B, H, K_c, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, K_c, D, device=device, dtype=torch.float16) * 0.1
                c_idx = torch.randint(0, K_c, (B, H, N), device=device, dtype=torch.int32)
                cuda_fn = lambda: mod.cluster_attention(q, cen, v, c_idx)
                ref_fn = lambda: mod.cluster_attention_ref(q, cen, v, c_idx)

            elif name == "multi_scale_retention":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                gammas = torch.full((H,), 0.95, device=device, dtype=torch.float32)
                cuda_fn = lambda: mod.multi_scale_retention(q, k, v, gammas)
                ref_fn = lambda: mod.multi_scale_retention_pytorch_ref(q, k, v, gammas)

            elif name == "fms_scan":
                q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
                decay = torch.full((H,), 0.95, device=device, dtype=torch.float32)
                cuda_fn = lambda: mod.fms_scan(q, k, v, decay)
                ref_fn = lambda: mod.fms_scan_pytorch_ref(q, k, v, decay)

            elif name == "adaptive_span":
                spans = torch.full((H,), 32.0, device=device, dtype=torch.float32)
                q_pos = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N)
                k_pos = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N)
                cuda_fn = lambda: mod.adaptive_span(spans, q_pos, k_pos, 16.0, True)
                ref_fn = lambda: mod.adaptive_span_pytorch_ref(spans, q_pos, k_pos, 16.0, True)

            if cuda_fn is None or ref_fn is None:
                continue

            # Test numerical parity & finite check
            out_cuda = cuda_fn()
            out_ref = ref_fn()

            has_nan = torch.isnan(out_cuda).any().item() or torch.isinf(out_cuda).any().item()
            diff = (out_cuda.float() - out_ref.float()).abs().max().item()

            t_cuda = benchmark_fn(cuda_fn)
            t_ref = benchmark_fn(ref_fn)
            speedup = t_ref / max(t_cuda, 1e-5)

            is_pass = (not has_nan) and (diff <= 0.05) and (t_cuda <= 20.0)
            if not is_pass:
                all_pass = False

            status_str = "PASS" if is_pass else "FAIL"
            nan_str = "CLEAN" if not has_nan else "NAN/INF"

            print(
                f"{idx:<3} | {name:<25} | {t_cuda:<10.4f} | {t_ref:<10.4f} | {speedup:<9.2f}x | {diff:<12.2e} | {nan_str:<9} | {status_str}",
                flush=True,
            )
            results.append(
                {
                    "module": name,
                    "cuda_ms": t_cuda,
                    "ref_ms": t_ref,
                    "speedup": speedup,
                    "diff": diff,
                    "has_nan": has_nan,
                    "pass": is_pass,
                }
            )
        except Exception as e:
            all_pass = False
            print(f"{idx:<3} | {name:<25} | EXCEPTION: {str(e)[:40]}", flush=True)

    print("=" * 110)
    print(f"DEEP PASS OVERALL STATUS: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 110)


if __name__ == "__main__":
    run_deep_pass()
