"""Eval concurrent offloading jitter module.

Author: Taha Majlesi
"""

import gc
import time
import json
import threading
import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from neuralops.utils.replace_attention import replace_attention


def print_log(msg):
    """Execute print log routine."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_log(f"Concurrent Generation & Background KV Offloading Benchmark on: {torch.cuda.get_device_name(0)}")

model_path = "/mnt/d/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Inductive reasoning prompt
reasoning_prompt = (
    "Question: Let a sequence be defined by S_0 = 1, and for all k >= 1, S_k = 3 * S_{k-1} + 2. "
    "Derive the step-by-step reasoning chain for S_1, S_2, S_3, S_4, and establish the general closed form: "
    "Step 1: Compute S_1 = 3 * 1 + 2 = 5. "
    "Step 2: Compute S_2 = 3 * 5 + 2 = 17. "
    "Step 3: Compute S_3 = 3 * 17 + 2 = 53. "
    "Step 4: Compute S_4 = 3 * 53 + 2 = 161. "
    "Step 5: Deduce the general recurrence relation:"
)

input_ids = tokenizer(reasoning_prompt, return_tensors="pt").input_ids.to(device)
TOTAL_GEN_STEPS = 128


class BackgroundWorker:
    """Simulates realistic asynchronous background operations (KV offloading to host pinned memory or SVD compression)."""

    def __init__(self, mode="offload"):
        """Execute init routine."""
        self.mode = mode
        self.stop_event = threading.Event()
        self.stream = torch.cuda.Stream(device=device)
        self.iterations = 0
        self.thread = None

    def _run(self):
        """Execute run routine."""
        with torch.cuda.stream(self.stream):
            while not self.stop_event.is_set():
                if self.mode == "offload":
                    # Pinned memory host-device copy stream
                    tensor_gpu = torch.randn((12, 16, 64, 64), device=device, dtype=torch.float16)
                    tensor_cpu = torch.empty((12, 16, 64, 64), dtype=torch.float16, pin_memory=True)
                    tensor_cpu.copy_(tensor_gpu, non_blocking=True)
                    del tensor_gpu, tensor_cpu
                elif self.mode == "compression":
                    # Background SVD matrix compression
                    mat = torch.randn((256, 256), device=device, dtype=torch.float32)
                    U, S, V = torch.linalg.svd(mat, full_matrices=False)
                    recon = (U[:, :64] * S[:64]) @ V[:64, :]
                    del mat, U, S, V, recon
                self.iterations += 1
                time.sleep(0.005)  # Yield

    def start(self):
        """Execute start routine."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Execute stop routine."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


architectures = [("GPT-2 Baseline (MHA)", None), ("GPT-2 + SVD Linear Attention", "svd_linear_attention")]

concurrency_modes = ["none", "kv_offload", "svd_compression"]

benchmark_matrix = {}

for mode in concurrency_modes:
    print_log(f"\n==================== Testing Concurrency Mode: [{mode.upper()}] ====================")
    benchmark_matrix[mode] = {}

    for name, key in architectures:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        m = GPT2LMHeadModel.from_pretrained(model_path)
        if key:
            replace_attention(m, attention_key=key)
        m = m.to(device).eval()

        # Warmup
        with torch.no_grad():
            _ = m(input_ids)
        torch.cuda.synchronize()

        # Start background worker if applicable
        worker = None
        if mode == "kv_offload":
            worker = BackgroundWorker(mode="offload")
            worker.start()
        elif mode == "svd_compression":
            worker = BackgroundWorker(mode="compression")
            worker.start()

        step_latencies_ms = []
        curr_ids = input_ids.clone()

        torch.cuda.synchronize()
        t_start_total = time.perf_counter()

        with torch.no_grad():
            for step in range(TOTAL_GEN_STEPS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = m(curr_ids)
                next_t = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_t], dim=-1)
                torch.cuda.synchronize()
                step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t_total_elapsed = time.perf_counter() - t_start_total
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        if worker is not None:
            worker.stop()
            print_log(f"Background worker completed {worker.iterations} concurrent iterations.")

        lat_arr = np.array(step_latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        throughput_tok_s = TOTAL_GEN_STEPS / t_total_elapsed

        has_nan = torch.isnan(out.logits).any().item()
        has_inf = torch.isinf(out.logits).any().item()

        print_log(
            f"[{name}] | Mean: {mean_lat:.2f} ms | p50: {p50:.2f} ms | p95: {p95:.2f} ms | "
            f"p99: {p99:.2f} ms | Std: {std_lat:.2f} ms | Throughput: {throughput_tok_s:.2f} tok/s | "
            f"p99 <= 45ms: {p99 <= 45.0} | Peak VRAM: {peak_vram_mb:.2f} MB"
        )

        benchmark_matrix[mode][name] = {
            "mode": mode,
            "architecture": name,
            "throughput_tok_s": round(throughput_tok_s, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "std_latency_ms": round(std_lat, 2),
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "p99_bounded_45ms": p99 <= 45.0,
            "peak_vram_mb": round(peak_vram_mb, 2),
            "zero_nan_inf": not has_nan and not has_inf,
        }

        del m, curr_ids
        torch.cuda.empty_cache()
        gc.collect()

out_file = "/mnt/d/SVD_linear_Attention/concurrent_offloading_jitter_results.json"
with open(out_file, "w") as f:
    json.dump(benchmark_matrix, f, indent=2)

print_log(f"All concurrent generation benchmarks completed! Written to {out_file}")
