#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Eval reasoning long context module.
"""

import gc
import os
import time
import math
import json
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from neuralops.utils.replace_attention import replace_attention


def print_log(msg):
    """Execute print log routine."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_log(f"Running on device: {torch.cuda.get_device_name(0)}")

model_path = os.environ.get("NEURALOPS_MODEL_PATH", "gpt2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def get_adapted_model(attention_type=None):
    """Execute get adapted model routine."""
    base = GPT2LMHeadModel.from_pretrained(model_path)
    config = base.config
    config.max_position_embeddings = 4096
    model = GPT2LMHeadModel(config)

    model.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    model.lm_head.load_state_dict(base.lm_head.state_dict())
    model.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())

    orig_wpe = base.transformer.wpe.weight.data
    new_wpe = (
        torch.nn.functional.interpolate(
            orig_wpe.unsqueeze(0).transpose(1, 2), size=4096, mode="linear", align_corners=True
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    model.transformer.wpe.weight.data.copy_(new_wpe)

    for i in range(len(base.transformer.h)):
        model.transformer.h[i].load_state_dict(base.transformer.h[i].state_dict())

    del base
    torch.cuda.empty_cache()

    if attention_type is not None:
        replace_attention(model, attention_key=attention_type)

    model = model.to(device).eval()
    return model


eval_text = (
    "Artificial intelligence and deep learning models have achieved remarkable success in natural language processing. "
    "The multi-head attention mechanism enables pairwise token interactions across the entire sequence context. "
    "Standard attention incurs quadratic computational complexity and memory footprint, limiting scalability to long documents. "
    "Linear attention formulations reformulate self-attention as a linear recurrent state-space or kernelized feature map, "
    "achieving linear computational scaling and strictly constant per-token state memory during autoregressive inference. "
    "Low-rank SVD projections compress parameter representations while retaining expressive representational power. "
    "Empirical validation confirms stable numerical training, finite gradient backpropagation, and sustained multi-step coherence."
)
eval_tokens = tokenizer(eval_text, return_tensors="pt").input_ids.to(device)

reasoning_prompt = (
    "Problem: In a sequence of reasoning steps, let x_0 = 3. At each step t >= 1, x_t = 2 * x_{t-1} + 1. "
    "Compute x_1, x_2, x_3, x_4, and verify the induction. "
    "Step 1: x_1 = 2 * 3 + 1 = 7. "
    "Step 2: x_2 = 2 * 7 + 1 = 15. "
    "Step 3: x_3 = 2 * 15 + 1 = 31. "
    "Step 4: x_4 = 2 * 31 + 1 = 63. "
    "Step 5: Therefore,"
)
reasoning_tokens = tokenizer(reasoning_prompt, return_tensors="pt").input_ids.to(device)


def eval_perplexity(model):
    """Execute eval perplexity routine."""
    with torch.no_grad():
        out = model(eval_tokens, labels=eval_tokens)
        loss = out.loss.item()
        ppl = math.exp(loss) if loss < 20 else float("inf")
    return float(round(loss, 4)), float(round(ppl, 2))


def eval_long_context_scaling(attention_key=None, seq_lens=None):
    """Execute eval long context scaling routine."""
    seq_lens = [512, 1024, 2048, 4096] if seq_lens is None else seq_lens
    scaling_stats = {}
    for L in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = get_adapted_model(attention_key)
        inp = torch.randint(100, 1000, (1, L), device=device)

        try:
            with torch.no_grad():
                _ = model(inp[:, : min(128, L)])
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp)
            torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            has_nan = torch.isnan(out.logits).any().item()
            has_inf = torch.isinf(out.logits).any().item()

            scaling_stats[L] = {
                "latency_ms": round(t_ms, 2),
                "peak_mem_mb": round(peak_mb, 2),
                "nan_or_inf": has_nan or has_inf,
                "status": "PASS",
            }
        except torch.cuda.OutOfMemoryError:
            scaling_stats[L] = {"status": "OOM"}
        except Exception as e:
            scaling_stats[L] = {"status": f"ERR: {str(e)}"}
        finally:
            del model, inp
            torch.cuda.empty_cache()
            gc.collect()

    return scaling_stats


def eval_o1_cache_state(attention_key=None, steps=32):
    """Execute eval o1 cache state routine."""
    torch.cuda.empty_cache()
    model = get_adapted_model(attention_key)

    curr_ids = reasoning_tokens.clone()
    mems = []

    with torch.no_grad():
        for s in range(steps):
            out = model(curr_ids)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            torch.cuda.synchronize()
            mems.append(torch.cuda.memory_allocated() / (1024**2))

    gen_text = tokenizer.decode(curr_ids[0][reasoning_tokens.shape[1] :], skip_special_tokens=True)

    del model, curr_ids
    torch.cuda.empty_cache()
    gc.collect()

    delta = mems[-1] - mems[0]
    return {
        "initial_mem_mb": round(mems[0], 2),
        "step_16_mem_mb": round(mems[15], 2),
        "step_32_mem_mb": round(mems[-1], 2),
        "memory_drift_mb": round(delta, 3),
        "generated_cot": gen_text.strip(),
    }


models_to_test = [
    ("GPT-2 Baseline (MHA)", None),
    ("GPT-2 + SVD Linear Attention", "svd_linear_attention"),
    ("GPT-2 + Gated Delta Net", "gated_delta_net"),
    ("GPT-2 + Chunk GLA", "gla_attention"),
]

final_report_data = {}

for name, key in models_to_test:
    print_log(f"=== Testing {name} ===")
    m = get_adapted_model(key)
    loss, ppl = eval_perplexity(m)
    del m
    torch.cuda.empty_cache()
    gc.collect()
    print_log(f"{name} -> Loss: {loss}, PPL: {ppl}")

    print_log(f"Benchmarking long-context scaling for {name}...")
    scaling = eval_long_context_scaling(key, [512, 1024, 2048, 4096])
    print_log(f"{name} Scaling: {scaling}")

    print_log(f"Evaluating O(1) cache memory and CoT coherence for {name}...")
    o1_cot = eval_o1_cache_state(key, steps=32)
    print_log(f"{name} CoT: {o1_cot['generated_cot']}")
    print_log(f"{name} Memory Drift over 32 steps: {o1_cot['memory_drift_mb']} MB")

    final_report_data[name] = {"loss": loss, "perplexity": ppl, "scaling": scaling, "generation": o1_cot}

result_path = os.path.join(os.environ.get("NEURALOPS_OUTPUT_DIR", "."), "reasoning_long_context_benchmark_results.json")
with open(result_path, "w") as f:
    json.dump(final_report_data, f, indent=2)

print_log(
    f"All benchmarks successfully completed! Output saved to {result_path}"
)
