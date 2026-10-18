import os
import gc
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer
from neuralops.utils.replace_attention import replace_attention

def print_log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_log(f"Token Latency Jitter & 128-step CoT Suite running on: {torch.cuda.get_device_name(0)}")

model_path = '/mnt/d/models/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Inductive Chain-of-Thought reasoning prompt
reasoning_prompt = (
    "Question: Let a sequence be defined by S_0 = 1, and for all k >= 1, S_k = 3 * S_{k-1} + 2. "
    "Derive the step-by-step reasoning chain for S_1, S_2, S_3, S_4, and establish the general closed form: "
    "Step 1: Compute S_1 = 3 * 1 + 2 = 5. "
    "Step 2: Compute S_2 = 3 * 5 + 2 = 17. "
    "Step 3: Compute S_3 = 3 * 17 + 2 = 53. "
    "Step 4: Compute S_4 = 3 * 53 + 2 = 161. "
    "Step 5: Deduce the general recurrence relation:"
)

input_ids = tokenizer(reasoning_prompt, return_tensors='pt').input_ids.to(device)
print_log(f"Prompt length: {input_ids.shape[1]} tokens")

TOTAL_GEN_STEPS = 128

architectures = [
    ("GPT-2 Baseline (MHA)", None),
    ("GPT-2 + SVD Linear Attention", "svd_linear_attention"),
    ("GPT-2 + Gated Delta Net", "gated_delta_net")
]

results = {}

for name, key in architectures:
    print_log(f"\n==================== Benchmarking: {name} ====================")
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
    
    # 128 consecutive generation steps with per-token latency profiling
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
    
    # Calculate jitter percentiles
    lat_arr = np.array(step_latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    throughput_tok_s = TOTAL_GEN_STEPS / t_total_elapsed
    
    # Decode text
    generated_text = tokenizer.decode(curr_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    has_nan = torch.isnan(out.logits).any().item()
    has_inf = torch.isinf(out.logits).any().item()
    
    print_log(
        f"Mean: {mean_lat:.2f} ms | p50: {p50:.2f} ms | p95: {p95:.2f} ms | p99: {p99:.2f} ms | "
        f"Std: {std_lat:.2f} ms | Throughput: {throughput_tok_s:.2f} tok/s | Peak VRAM: {peak_vram_mb:.2f} MB"
    )
    print_log(f"Sample output: {generated_text[:100]}...")
    
    results[name] = {
        'total_steps': TOTAL_GEN_STEPS,
        'throughput_tok_s': round(throughput_tok_s, 2),
        'mean_latency_ms': round(mean_lat, 2),
        'std_latency_ms': round(std_lat, 2),
        'p50_ms': round(p50, 2),
        'p90_ms': round(p90, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2),
        'peak_vram_mb': round(peak_vram_mb, 2),
        'zero_nan_inf': not has_nan and not has_inf,
        'generated_cot_sample': generated_text.strip()
    }
    
    del m
    torch.cuda.empty_cache()
    gc.collect()

out_file = '/mnt/d/SVD_linear_Attention/cot_jitter_128steps_results.json'
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)

print_log(f"Jitter benchmark finished! Written to {out_file}")
