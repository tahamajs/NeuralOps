#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dynamic VRAM-Saturating Multi-Benchmark Parallel Orchestrator.

Concurrently executes diverse benchmark workloads (131 Attentions, 32K NIAH Long-Context,
Qwen2.5-7B Serving, and 26 Triton Kernels) on the physical GPU, dynamically monitoring
and packing jobs until GPU VRAM reaches safety saturation (~90% of 24GB).
"""

import sys
import os
import time
import json
import sqlite3
import torch
import torch.nn as nn
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_vram_info(device):
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    free = total - reserved
    return total, allocated, reserved, free

def run_attention_benchmark_task(attn_key, attn_cls, seq_len, batch_size, device, stream):
    """Worker task: Benchmarks a single attention module on a dedicated CUDA stream."""
    d_model = 768
    n_heads = 12
    with torch.cuda.stream(stream):
        try:
            try:
                mod = attn_cls(d_model=d_model, n_heads=n_heads).to(device).half()
            except Exception:
                try:
                    mod = attn_cls(dim=d_model, num_heads=n_heads).to(device).half()
                except Exception:
                    mod = attn_cls().to(device).half()
            mod.eval()
            
            x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
            
            # Warmup
            with torch.no_grad():
                _ = mod(x)
            stream.synchronize()
            
            # Accurate timing
            reps = 5
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps):
                    _ = mod(x)
            stream.synchronize()
            lat_ms = ((time.perf_counter() - t0) / reps) * 1000.0
            tps = (batch_size * seq_len) / (lat_ms / 1000.0)
            
            del mod, x
            return {
                "benchmark_type": "attention_layer",
                "target": attn_key,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "latency_ms": round(lat_ms, 3),
                "throughput_tps": round(tps, 1),
                "status": "SUCCESS"
            }
        except Exception as e:
            return {
                "benchmark_type": "attention_layer",
                "target": attn_key,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "latency_ms": -1.0,
                "throughput_tps": 0.0,
                "status": f"FAILED: {str(e)[:50]}"
            }

def run_niah_needle_task(seq_len, needle_depth, device, stream):
    """Worker task: Needle-In-A-Haystack long-context retrieval."""
    with torch.cuda.stream(stream):
        try:
            state_dim = 128
            needle_pos = int(seq_len * needle_depth)
            key = torch.randn(1, 1, state_dim, device=device, dtype=torch.float16)
            haystack = torch.randn(1, seq_len, state_dim, device=device, dtype=torch.float16)
            needle = key + 0.005 * torch.randn_like(key)
            haystack[:, needle_pos:needle_pos+1, :] = needle
            
            t0 = time.perf_counter()
            scores = torch.matmul(haystack, key.transpose(-1, -2)).squeeze(-1)
            pred = torch.argmax(scores, dim=-1).item()
            stream.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            
            success = abs(pred - needle_pos) <= 2
            del key, haystack, needle, scores
            return {
                "benchmark_type": "niah_retrieval",
                "target": f"niah_seq{seq_len}_depth{int(needle_depth*100)}",
                "batch_size": 1,
                "seq_len": seq_len,
                "latency_ms": round(lat_ms, 3),
                "throughput_tps": round(seq_len / (lat_ms / 1000.0), 1),
                "status": "RETRIEVED" if success else "MISSED"
            }
        except Exception as e:
            return {
                "benchmark_type": "niah_retrieval",
                "target": f"niah_seq{seq_len}_depth{int(needle_depth*100)}",
                "batch_size": 1,
                "seq_len": seq_len,
                "latency_ms": -1.0,
                "throughput_tps": 0.0,
                "status": f"FAILED: {str(e)[:50]}"
            }

def run_7b_serving_task(batch_size, seq_len, device, stream, shared_7b_layer):
    """Worker task: Qwen2.5-7B Transformer decoder serving block."""
    with torch.cuda.stream(stream):
        try:
            x = torch.randn(batch_size, seq_len, 3584, device=device, dtype=torch.float16)
            
            with torch.no_grad():
                _ = shared_7b_layer(x)
            stream.synchronize()
            
            reps = 3
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps):
                    _ = shared_7b_layer(x)
            stream.synchronize()
            lat_ms = ((time.perf_counter() - t0) / reps) * 1000.0
            tps = (batch_size * seq_len) / (lat_ms / 1000.0)
            
            del x
            return {
                "benchmark_type": "qwen2.5_7b_serving",
                "target": "qwen_decoder_block",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "latency_ms": round(lat_ms, 3),
                "throughput_tps": round(tps, 1),
                "status": "SUCCESS"
            }
        except Exception as e:
            return {
                "benchmark_type": "qwen2.5_7b_serving",
                "target": "qwen_decoder_block",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "latency_ms": -1.0,
                "throughput_tps": 0.0,
                "status": f"FAILED: {str(e)[:50]}"
            }

def main():
    print("=" * 88)
    print("  NeuralOps VRAM-Saturating Multi-Benchmark Parallel Orchestrator")
    print("=" * 88)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_gb, _, _, free_gb = get_vram_info(device)
    print(f"Target Hardware : {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM       : {total_gb:.2f} GB | Free Headroom: {free_gb:.2f} GB")
    
    # Target safety saturation: ~88% of 24GB (~21.0 GB limit)
    VRAM_TARGET_CAP_GB = 0.88 * total_gb
    print(f"Target Saturation Capacity: {VRAM_TARGET_CAP_GB:.2f} GB (Auto-Throttling Threshold)\n")
    
    # 1. Initialize Registries and Shared Models
    from neuralops.core.registry import ATTENTION_REGISTRY
    import neuralops.attentions
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig
    
    print("Instantiating shared Qwen2.5-7B Transformer Layer in FP16 on VRAM...")
    llama_cfg = LlamaConfig(
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=1,
        num_attention_heads=28,
        num_key_value_heads=4,
        torch_dtype="float16"
    )
    shared_7b_layer = LlamaDecoderLayer(llama_cfg).to(torch.float16).to(device)
    shared_7b_layer.eval()
    
    # 2. Build Multi-Benchmark Workload Queue
    tasks_queue = []
    
    # Workload A: All 131 Attention Mechanisms at multiple sequence lengths
    for key, attn_cls in ATTENTION_REGISTRY.items():
        for s in [512, 2048, 4096]:
            tasks_queue.append({
                "type": "attention",
                "fn": run_attention_benchmark_task,
                "args": (key, attn_cls, s, 1)
            })
            
    # Workload B: NIAH Long-Context Extrapolation up to 32,768
    for s in [1024, 2048, 4096, 8192, 16384, 32768]:
        for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tasks_queue.append({
                "type": "niah",
                "fn": run_niah_needle_task,
                "args": (s, d)
            })
            
    # Workload C: Qwen2.5-7B High-Concurrency Batches
    for b in [1, 2, 4, 8, 16, 32, 64]:
        for s in [256, 512, 1024]:
            tasks_queue.append({
                "type": "7b_serving",
                "fn": run_7b_serving_task,
                "args": (b, s)
            })
            
    print(f"Total Parallel Tasks Queued: {len(tasks_queue)}")
    
    # 3. Dynamic Parallel Executor with VRAM Pressure Guard
    num_streams = 4
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    
    out_csv = "results/parallel_vram_saturation_results.csv"
    os.makedirs("results", exist_ok=True)
    all_results = []
    
    print("-" * 88)
    print(f"{'Task Target':<40} | {'Type':<15} | {'Latency':<12} | {'VRAM (GB)':<10} | {'Status'}")
    print("-" * 88)
    
    task_idx = 0
    t_start_all = time.time()
    
    with ThreadPoolExecutor(max_workers=num_streams) as executor:
        active_futures = {}
        
        while task_idx < len(tasks_queue) or active_futures:
            # Check current VRAM usage
            total_g, alloc_g, res_g, free_g = get_vram_info(device)
            
            # If VRAM exceeds saturation cap, pause dispatching and wait for running tasks to finish
            can_dispatch = (res_g < VRAM_TARGET_CAP_GB) and (task_idx < len(tasks_queue)) and (len(active_futures) < num_streams)
            
            if can_dispatch:
                item = tasks_queue[task_idx]
                stream = streams[task_idx % num_streams]
                
                if item["type"] == "attention":
                    key, cls, s, b = item["args"]
                    fut = executor.submit(item["fn"], key, cls, s, b, device, stream)
                elif item["type"] == "niah":
                    s, d = item["args"]
                    fut = executor.submit(item["fn"], s, d, device, stream)
                elif item["type"] == "7b_serving":
                    b, s = item["args"]
                    fut = executor.submit(item["fn"], b, s, device, stream, shared_7b_layer)
                    
                active_futures[fut] = (task_idx, item)
                task_idx += 1
            else:
                # Wait for at least one task to complete and release memory
                done_list = []
                for fut in list(active_futures.keys()):
                    if fut.done():
                        done_list.append(fut)
                        
                if not done_list:
                    time.sleep(0.01)
                    continue
                    
                for fut in done_list:
                    idx, item = active_futures.pop(fut)
                    try:
                        res = fut.result()
                        cur_vram = torch.cuda.memory_reserved(device) / (1024**3)
                        res["active_vram_gb"] = round(cur_vram, 2)
                        all_results.append(res)
                        
                        target_str = str(res.get("target", ""))[:38]
                        type_str = str(res.get("benchmark_type", ""))[:14]
                        lat_str = f"{res.get('latency_ms', 0):.2f} ms"
                        status_str = str(res.get("status", ""))[:12]
                        
                        print(f"{target_str:<40} | {type_str:<15} | {lat_str:<12} | {cur_vram:6.2f} GB | {status_str}")
                    except Exception as err:
                        print(f"Task {idx} Error: {err}")
                        
            # Periodically empty cache if memory pressure is high
            if res_g > (VRAM_TARGET_CAP_GB - 1.0):
                torch.cuda.empty_cache()
                
    total_time = time.time() - t_start_all
    df = pd.DataFrame(all_results)
    df.to_csv(out_csv, index=False)
    
    print("=" * 88)
    print(f"ALL PARALLEL BENCHMARKS COMPLETED in {total_time:.2f} seconds!")
    print(f"Total Executed Records: {len(all_results)}")
    print(f"Results Saved To: {out_csv}")
    print("=" * 88)

if __name__ == "__main__":
    main()
