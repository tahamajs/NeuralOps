import sys
import os
import gc
import json
import time
import glob
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig

print("================================================================================")
print("  NeuralOps HEAVY LLM Benchmark on Qwen2.5-7B (NVIDIA RTX 3090 - 24GB)")
print("================================================================================")

model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct")
print("Loading model weights from:", model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
print("Target GPU: {} ({}) | Total VRAM: {:.2f} GB".format(device, torch.cuda.get_device_name(device), vram_total))

with open(os.path.join(model_path, "config.json"), "r") as f:
    cfg_dict = json.load(f)

llama_cfg = LlamaConfig(
    vocab_size=cfg_dict.get("vocab_size", 152064),
    hidden_size=cfg_dict.get("hidden_size", 3584),
    intermediate_size=cfg_dict.get("intermediate_size", 18944),
    num_hidden_layers=cfg_dict.get("num_hidden_layers", 28),
    num_attention_heads=cfg_dict.get("num_attention_heads", 28),
    num_key_value_heads=cfg_dict.get("num_key_value_heads", 4),
    rms_norm_eps=cfg_dict.get("rms_norm_eps", 1e-6),
    rope_theta=cfg_dict.get("rope_theta", 1000000.0),
    torch_dtype="float16"
)

print("\nModel Architecture Specifications:")
print(" - Hidden Size:        {}".format(llama_cfg.hidden_size))
print(" - Num Hidden Layers:  {}".format(llama_cfg.num_hidden_layers))
print(" - Num Attention Heads: {}".format(llama_cfg.num_attention_heads))
print(" - Num KV Heads (GQA): {}".format(llama_cfg.num_key_value_heads))
print(" - Intermediate Size:  {}".format(llama_cfg.intermediate_size))
print(" - Total Parameters:   ~7.61 Billion Parameters")

print("\nInitializing model skeleton in float16...")
torch.cuda.empty_cache()
t0 = time.perf_counter()
model = LlamaForCausalLM(llama_cfg).to(torch.float16).to(device)

print("Directly streaming safetensor weight shards into RTX 3090 GDDR6X VRAM...")
weight_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
print("Found {} safetensor shards.".format(len(weight_files)))

model_state = model.state_dict()
for wf in weight_files:
    fname = os.path.basename(wf)
    print("  Loading shard {} into GPU VRAM...".format(fname), end="", flush=True)
    t_shard = time.perf_counter()
    shard_tensors = load_file(wf, device=str(device))
    for k, v in shard_tensors.items():
        if k in model_state:
            model_state[k].copy_(v)
    del shard_tensors
    torch.cuda.empty_cache()
    print(" Done in {:.2f}s".format(time.perf_counter() - t_shard))

load_time = time.perf_counter() - t0
vram_allocated = torch.cuda.memory_allocated(device) / (1024**3)
print("\n>>> 7B Model fully loaded in {:.2f}s | VRAM Consumed: {:.2f} GB / {:.2f} GB".format(
    load_time, vram_allocated, vram_total
))

# Benchmark generation & forward pass under realistic context lengths
batch_size = 1
seq_lens = [256, 512, 1024, 2048]
print("\n" + "=" * 80)
print("  Running Real-Time Forward Pass & Attention Benchmark Across Contexts")
print("=" * 80)
print("{:<10} | {:<20} | {:<18} | {:<15}".format("SeqLen", "Full 7B Latency", "Throughput (T/s)", "Allocated VRAM"))
print("-" * 80)

model.eval()
for s in seq_lens:
    input_ids = torch.randint(100, 10000, (batch_size, s), device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()
    
    # Timing
    reps = 3
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            out = model(input_ids)
    torch.cuda.synchronize()
    lat_ms = ((time.perf_counter() - t_start) / reps) * 1000.0
    tps = s / (lat_ms / 1000.0)
    mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
    
    print("{:<10} | {:<17.2f} ms | {:<18.2f} | {:.2f} GB".format(s, lat_ms, tps, mem_gb))

print("=" * 80)
print("HEAVY 7B BENCHMARK COMPLETED SUCCESSFULLY ON NVIDIA GEFORCE RTX 3090")
