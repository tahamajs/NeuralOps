"""
Head-to-head FLA-style benchmark on NVIDIA RTX 4070.
Evaluates Forward (ms), Forward+Backward (ms), and Peak VRAM (MB)
across sequence lengths T in [512, 1024, 2048, 4096, 8192] for:
1. PyTorch SDPA (Standard Scaled Dot-Product Attention)
2. Chunk GLA (Gated Linear Attention)
3. Chunk GDN / GDN-2 (Gated DeltaNet 2)
4. PGDN (Preconditioned Gated DeltaNet)
5. DeltaProduct / PaTH Attention (Householder Reflection)
6. Cosformer (Linear Attention)
7. Standard Cross-Entropy vs Fused Linear Cross-Entropy
8. Standard KL-Divergence vs Fused Linear KL-Divergence
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

import neuralops
import neuralops.attentions
import neuralops.attentions.gated.flagship_fla
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.triton_kernels.fused_ce import FusedLinearCrossEntropy
from neuralops.core.triton_kernels.fused_kd import FusedLinearKLDivergence
from transformers import GPT2Config

B = 2
D = 768
H = 12
seq_lens = [512, 1024, 2048, 4096, 8192]

models_to_test = [
    ("PyTorch SDPA", "full_attention"),
    ("Chunk GLA", "gla_attention"),
    ("Chunk GDN", "gated_delta_net"),
    ("GDN-2", "gdn2_attention"),
    ("PGDN", "preconditioned_gdn_attention"),
    ("DeltaProduct / PaTH", "delta_product_attention"),
    ("Cosformer", "cosformer"),
]

def benchmark_attention(name, key, seq_len):
    cls = ATTENTION_REGISTRY.get(key)
    cfg = GPT2Config(n_embd=D, n_head=H, hidden_size=D, num_attention_heads=H)
    try:
        mod = cls(config=cfg)
    except TypeError:
        try:
            mod = cls(hidden_size=D, num_heads=H)
        except TypeError:
            mod = cls()

    mod = mod.to(device=device, dtype=torch.bfloat16).train()
    x = torch.randn(B, seq_len, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Warmup
    for _ in range(3):
        try:
            out = mod(x)
        except Exception:
            out = mod(x, x, x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = out.sum()
        loss.backward()
        if x.grad is not None:
            x.grad.zero_()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Forward benchmark
    t0 = time.perf_counter()
    iters = 3 if seq_len >= 4096 else 5
    for _ in range(iters):
        try:
            out = mod(x)
        except Exception:
            out = mod(x, x, x)
    torch.cuda.synchronize()
    fwd_time_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Fwd + Bwd benchmark
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        try:
            out = mod(x)
        except Exception:
            out = mod(x, x, x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = out.sum()
        loss.backward()
        if x.grad is not None:
            x.grad.zero_()
    torch.cuda.synchronize()
    fwd_bwd_time_ms = (time.perf_counter() - t0) * 1000.0 / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return fwd_time_ms, fwd_bwd_time_ms, peak_vram_mb

print("\n" + "=" * 95, flush=True)
print(f"{'Mechanism':<24} | {'SeqLen':<7} | {'Fwd (ms)':<10} | {'Fwd+Bwd (ms)':<14} | {'Peak VRAM (MB)':<14}", flush=True)
print("-" * 95, flush=True)

results = []
for name, key in models_to_test:
    for T in seq_lens:
        try:
            fwd, fwd_bwd, vram = benchmark_attention(name, key, T)
            print(f"{name:<24} | {T:<7} | {fwd:10.2f} | {fwd_bwd:14.2f} | {vram:14.1f}", flush=True)
            results.append((name, T, fwd, fwd_bwd, vram))
        except torch.cuda.OutOfMemoryError:
            print(f"{name:<24} | {T:<7} | {'OOM':<10} | {'OOM':<14} | {'OOM':<14}", flush=True)
            results.append((name, T, "OOM", "OOM", "OOM"))
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name:<24} | {T:<7} | {'ERR':<10} | {'ERR':<14} | {str(e)[:14]}", flush=True)
            torch.cuda.empty_cache()

# Loss benchmarking
print("\n" + "=" * 95)
print("LOSS BENCHMARKS: Standard vs Fused Linear Modules (Vocab=50,257)")
print("=" * 95)
print(f"{'Loss Type':<30} | {'SeqLen':<7} | {'Fwd+Bwd (ms)':<14} | {'Peak VRAM (MB)':<14}")
print("-" * 95)

V = 50257
for T in [512, 1024, 2048, 4096]:
    # Standard CE
    try:
        h = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(V, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        targets = torch.randint(0, V, (B, T), device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            logits = F.linear(h, w)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            loss.backward()
            h.grad = None
            w.grad = None
        torch.cuda.synchronize()
        std_ce_ms = (time.perf_counter() - t0) * 1000.0 / 5
        std_ce_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{'Standard Linear CE':<30} | {T:<7} | {std_ce_ms:14.2f} | {std_ce_vram:14.1f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{'Standard Linear CE':<30} | {T:<7} | {'OOM':<14} | {'OOM':<14}")
        torch.cuda.empty_cache()

    # Fused Linear CE
    try:
        fused_ce = FusedLinearCrossEntropy().to(device)
        h = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(V, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        targets = torch.randint(0, V, (B, T), device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            loss = fused_ce(h, w, targets)
            loss.backward()
            h.grad = None
            w.grad = None
        torch.cuda.synchronize()
        fused_ce_ms = (time.perf_counter() - t0) * 1000.0 / 5
        fused_ce_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{'Fused Linear CE (FLA-style)':<30} | {T:<7} | {fused_ce_ms:14.2f} | {fused_ce_vram:14.1f}")
    except Exception as e:
        print(f"{'Fused Linear CE (FLA-style)':<30} | {T:<7} | {'ERR':<14} | {str(e)[:14]}")
        torch.cuda.empty_cache()

    # Standard KD
    try:
        s_h = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        t_h = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        s_w = torch.randn(V, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        t_w = torch.randn(V, D, device=device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            s_log = F.log_softmax(F.linear(s_h, s_w), dim=-1)
            t_prob = F.softmax(F.linear(t_h, t_w), dim=-1)
            loss = F.kl_div(s_log, t_prob, reduction="batchmean")
            loss.backward()
            s_h.grad = None
            s_w.grad = None
        torch.cuda.synchronize()
        std_kd_ms = (time.perf_counter() - t0) * 1000.0 / 5
        std_kd_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{'Standard Linear KL-Div':<30} | {T:<7} | {std_kd_ms:14.2f} | {std_kd_vram:14.1f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{'Standard Linear KL-Div':<30} | {T:<7} | {'OOM':<14} | {'OOM':<14}")
        torch.cuda.empty_cache()

    # Fused Linear KD
    try:
        fused_kd = FusedLinearKLDivergence().to(device)
        s_h = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        t_h = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
        s_w = torch.randn(V, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        t_w = torch.randn(V, D, device=device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            loss = fused_kd(s_h, s_w, t_h, t_w)
            loss.backward()
            s_h.grad = None
            s_w.grad = None
        torch.cuda.synchronize()
        fused_kd_ms = (time.perf_counter() - t0) * 1000.0 / 5
        fused_kd_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{'Fused Linear KL-Div (FLA)':<30} | {T:<7} | {fused_kd_ms:14.2f} | {fused_kd_vram:14.1f}")
    except Exception as e:
        print(f"{'Fused Linear KL-Div (FLA)':<30} | {T:<7} | {'ERR':<14} | {str(e)[:14]}")
        torch.cuda.empty_cache()
