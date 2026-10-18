import os
import gc
import sys
import time
import math
import json
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from neuralops.utils.replace_attention import replace_attention

def print_log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_log(f"Reasoning & Long-Context Evaluation Suite running on: {torch.cuda.get_device_name(0)}")

model_path = '/mnt/d/models/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def build_model(attention_type=None, max_pos=4096):
    """Load GPT-2 base and surgically adapt position embeddings and attention."""
    base = GPT2LMHeadModel.from_pretrained(model_path)
    config = base.config
    config.max_position_embeddings = max_pos
    model = GPT2LMHeadModel(config)
    
    model.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    model.lm_head.load_state_dict(base.lm_head.state_dict())
    model.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())
    
    orig_wpe = base.transformer.wpe.weight.data
    new_wpe = torch.nn.functional.interpolate(
        orig_wpe.unsqueeze(0).transpose(1, 2),
        size=max_pos,
        mode='linear',
        align_corners=True
    ).squeeze(0).transpose(0, 1)
    model.transformer.wpe.weight.data.copy_(new_wpe)
    
    for i in range(len(base.transformer.h)):
        model.transformer.h[i].load_state_dict(base.transformer.h[i].state_dict())
        
    del base
    torch.cuda.empty_cache()
    
    if attention_type is not None:
        replace_attention(model, attention_key=attention_type)
        
    return model.to(device).eval()

# Multi-step reasoning evaluation prompt (Mathematical Chain-of-Thought)
reasoning_prompt = (
    "Problem: In a sequence of reasoning steps, let x_0 = 3. At each step t >= 1, x_t = 2 * x_{t-1} + 1. "
    "Compute x_1, x_2, x_3, x_4, and verify the deductive formula. "
    "Step 1: x_1 = 2 * 3 + 1 = 7. "
    "Step 2: x_2 = 2 * 7 + 1 = 15. "
    "Step 3: x_3 = 2 * 15 + 1 = 31. "
    "Step 4: x_4 = 2 * 31 + 1 = 63. "
    "Step 5: Therefore,"
)

# Benchmark 1: Context Lengths Sweep over L in [128, 512, 1024, 2048]
CONTEXT_LENGTHS = [128, 512, 1024, 2048]
mechanisms = [
    ("GPT-2 Baseline (MHA)", None),
    ("GPT-2 + SVD Linear Attention", "svd_linear_attention"),
    ("GPT-2 + Gated Delta Net", "gated_delta_net"),
    ("GPT-2 + Chunk GLA", "gla_attention")
]

benchmark_report = {}

print_log("=== Step 1: Long-Context Scaling & Numerical Validation over L=128..2048 ===")

for name, key in mechanisms:
    print_log(f"\nEvaluating Mechanism: [{name}]")
    model = build_model(attention_type=key, max_pos=4096)
    mech_results = {'context_scaling': {}, 'generation_test': {}}
    
    for L in CONTEXT_LENGTHS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test input sequence
        inp = torch.randint(100, 1000, (1, L), device=device)
        
        # Warmup
        with torch.no_grad():
            _ = model(inp[:, :min(64, L)])
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(inp)
        torch.cuda.synchronize()
        fwd_latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        nan_count = torch.isnan(out.logits).sum().item()
        inf_count = torch.isinf(out.logits).sum().item()
        
        # Loss & Perplexity on synthetic validation segment
        loss_val = None
        ppl_val = None
        with torch.no_grad():
            loss_t = model(inp, labels=inp).loss
            loss_val = round(loss_t.item(), 4)
            ppl_val = round(math.exp(min(loss_val, 20.0)), 2)
            
        print_log(
            f"L={L:4d} | Fwd Latency: {fwd_latency_ms:6.2f} ms | Peak VRAM: {peak_vram_mb:7.2f} MB | "
            f"Loss: {loss_val:6.4f} | PPL: {ppl_val:8.2f} | NaNs: {nan_count} | Infs: {inf_count}"
        )
        
        mech_results['context_scaling'][L] = {
            'latency_ms': round(fwd_latency_ms, 2),
            'peak_vram_mb': round(peak_vram_mb, 2),
            'loss': loss_val,
            'perplexity': ppl_val,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'status': 'PASS' if nan_count == 0 and inf_count == 0 else 'FAIL'
        }
        
    print_log(f"=== Step 2: Autoregressive CoT Generation & KV State Retention over 64 Steps for [{name}] ===")
    prompt_ids = tokenizer(reasoning_prompt, return_tensors='pt').input_ids.to(device)
    
    # 64-step unrolling
    curr_ids = prompt_ids.clone()
    gen_tokens = 64
    mems = []
    
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        for s in range(gen_tokens):
            out = model(curr_ids)
            next_t = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_t], dim=-1)
            torch.cuda.synchronize()
            mems.append(torch.cuda.memory_allocated() / (1024**2))
            
    t_gen_total = time.perf_counter() - t_gen_start
    gen_throughput = gen_tokens / t_gen_total
    mem_drift = mems[-1] - mems[0]
    
    # Verify outputs
    out_text = tokenizer.decode(curr_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
    all_nan = torch.isnan(out.logits).any().item()
    all_inf = torch.isinf(out.logits).any().item()
    
    print_log(
        f"Generation: {gen_throughput:.2f} tok/s ({t_gen_total:.3f} s) | "
        f"Mem Drift: {mem_drift:.3f} MB | Zero NaN/Inf: {not all_nan and not all_inf}"
    )
    print_log(f"Generated CoT Sample: {out_text[:90]}...")
    
    mech_results['generation_test'] = {
        'throughput_tok_s': round(gen_throughput, 2),
        'total_time_s': round(t_gen_total, 4),
        'initial_mem_mb': round(mems[0], 2),
        'final_mem_mb': round(mems[-1], 2),
        'memory_drift_mb': round(mem_drift, 3),
        'zero_nan_inf': not all_nan and not all_inf,
        'cot_text': out_text.strip()
    }
    
    benchmark_report[name] = mech_results
    del model
    torch.cuda.empty_cache()
    gc.collect()

out_path = '/mnt/d/SVD_linear_Attention/reasoning_evaluation_full_metrics.json'
with open(out_path, 'w') as f:
    json.dump(benchmark_report, f, indent=2)

print_log(f"Reasoning & Long-Context benchmark complete! Written to {out_path}")
