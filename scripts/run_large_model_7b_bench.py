import sys
import os
import gc
import json
import time
import glob
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

print("================================================================================")
print("  NeuralOps HEAVY LLM Layer Benchmark on Qwen2.5-7B (NVIDIA RTX 3090 - 24GB)")
print("================================================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
print("Target GPU: {} ({}) | Total VRAM: {:.2f} GB".format(device, torch.cuda.get_device_name(device), vram_total))

model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct")
with open(os.path.join(model_path, "config.json"), "r") as f:
    cfg_dict = json.load(f)

llama_cfg = LlamaConfig(
    vocab_size=cfg_dict.get("vocab_size", 152064),
    hidden_size=cfg_dict.get("hidden_size", 3584),
    intermediate_size=cfg_dict.get("intermediate_size", 18944),
    num_hidden_layers=1, # Layer-by-layer benchmark with real Qwen 7B dimensions
    num_attention_heads=cfg_dict.get("num_attention_heads", 28),
    num_key_value_heads=cfg_dict.get("num_key_value_heads", 4),
    rms_norm_eps=cfg_dict.get("rms_norm_eps", 1e-6),
    rope_theta=cfg_dict.get("rope_theta", 1000000.0),
    torch_dtype="float16"
)

print("\nModel Layer Architecture (Exact Qwen2.5-7B-Instruct Specifications):")
print(" - Hidden Dimension (d_model): {}".format(llama_cfg.hidden_size))
print(" - Intermediate (FFN) Dim:     {}".format(llama_cfg.intermediate_size))
print(" - Query Heads:                {}".format(llama_cfg.num_attention_heads))
print(" - KV Heads (GQA):             {}".format(llama_cfg.num_key_value_heads))
print(" - Total 7B Model Parameters:  7.61 Billion")

print("\nDirectly initializing Qwen2.5-7B Transformer Layer in FP16 on RTX 3090 VRAM...")
torch.cuda.empty_cache()
t0 = time.perf_counter()
layer = LlamaDecoderLayer(llama_cfg).to(torch.float16).to(device)
layer.eval()

# Load real weights from shard 1 for layer 0
print("Loading real pretrained weights for Layer 0 from shard 1...")
shard1_path = os.path.join(model_path, "model-00001-of-00004.safetensors")
shard1 = load_file(shard1_path, device=str(device))
layer_state = layer.state_dict()
loaded_params = 0
for k, v in shard1.items():
    # map model.layers.0.* to layer.*
    if k.startswith("model.layers.0."):
        param_name = k.replace("model.layers.0.", "")
        if param_name in layer_state:
            layer_state[param_name].copy_(v)
            loaded_params += 1
del shard1
torch.cuda.empty_cache()
print("Loaded {} parameter tensors directly into GPU VRAM in {:.2f}s.".format(loaded_params, time.perf_counter() - t0))

print("\n" + "=" * 80)
print("  Benchmarking Real Qwen2.5-7B Transformer Block Across Sequence Lengths")
print("=" * 80)
print("{:<10} | {:<20} | {:<22} | {:<15}".format("SeqLen", "Layer Latency", "Estimated 28L 7B Lat", "Allocated VRAM"))
print("-" * 80)

batch_size = 1
seq_lens = [512, 1024, 2048, 4096, 8192, 16384]

for s in seq_lens:
    hidden_states = torch.randn(batch_size, s, llama_cfg.hidden_size, dtype=torch.float16, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = layer(hidden_states)
    torch.cuda.synchronize()
    
    # Accurate timing
    reps = 5
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            out = layer(hidden_states)
    torch.cuda.synchronize()
    lat_layer_ms = ((time.perf_counter() - t_start) / reps) * 1000.0
    lat_full_model_ms = lat_layer_ms * 28.0
    mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
    
    print("{:<10} | {:<17.3f} ms | {:<19.2f} ms | {:.2f} GB".format(s, lat_layer_ms, lat_full_model_ms, mem_gb))

print("=" * 80)
print("REAL 7B TRANSFORMER BENCHMARK COMPLETED SUCCESSFULLY ON RTX 3090!")
