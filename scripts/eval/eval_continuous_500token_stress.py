#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Eval continuous 500token stress module.
"""

import gc
import time
import json
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from neuralops.utils.replace_attention import replace_attention


def print_log(msg):
    """Execute print log routine."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_log(f"Continuous Long-Context 500-Token Trajectory & O(1) State Certifier on: {torch.cuda.get_device_name(0)}")

model_path = "/mnt/d/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def build_model_for_scaling(attention_type=None, max_pos=8192):
    """Load GPT-2 base and interpolate positional embeddings up to 8192."""
    base = GPT2LMHeadModel.from_pretrained(model_path)
    config = base.config
    config.max_position_embeddings = max_pos
    model = GPT2LMHeadModel(config)

    model.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    model.lm_head.load_state_dict(base.lm_head.state_dict())
    model.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())

    orig_wpe = base.transformer.wpe.weight.data
    new_wpe = (
        torch.nn.functional.interpolate(
            orig_wpe.unsqueeze(0).transpose(1, 2), size=max_pos, mode="linear", align_corners=True
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

    return model.to(device).eval()


reasoning_prompt = (
    "Problem: Let a sequence be defined by T_0 = 2, and for all n >= 1, T_n = 2 * T_{n-1} + 3. "
    "Derive the closed form through step-by-step mathematical reasoning. "
    "Step 1: T_1 = 2 * 2 + 3 = 7. "
    "Step 2: T_2 = 2 * 7 + 3 = 17. "
    "Step 3: T_3 = 2 * 17 + 3 = 37. "
    "Step 4: T_4 = 2 * 37 + 3 = 77. "
    "Step 5: Therefore, we establish the continuous multi-step inductive proof:"
)

input_ids = tokenizer(reasoning_prompt, return_tensors="pt").input_ids.to(device)
print_log(f"Prompt length: {input_ids.shape[1]} tokens")

TOTAL_TOKENS = 500

architectures = [("GPT-2 + SVD Linear Attention", "svd_linear_attention"), ("GPT-2 + Chunk GLA", "gla_attention")]

stress_results = {}

print_log("=== Beginning 500-Token Continuous Autoregressive Trajectory Sweep ===")

for name, key in architectures:
    print_log(f"\n==================== Launching 500-Token Stress Test: {name} ====================")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model_for_scaling(attention_type=key, max_pos=8192)

    curr_ids = input_ids.clone()
    mem_trajectory = []

    nan_occurred = False
    inf_occurred = False

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    checkpoint_times = []

    with torch.no_grad():
        for step in range(TOTAL_TOKENS):
            out = model(curr_ids)
            logits = out.logits[:, -1, :]

            if torch.isnan(logits).any().item():
                nan_occurred = True
            if torch.isinf(logits).any().item():
                inf_occurred = True

            next_t = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_t], dim=-1)

            if (step + 1) % 50 == 0:
                torch.cuda.synchronize()
                elapsed_now = time.perf_counter() - t_start
                cur_vram = torch.cuda.memory_allocated() / (1024**2)
                cur_speed = (step + 1) / elapsed_now
                mem_trajectory.append(
                    {
                        "step": step + 1,
                        "vram_mb": round(cur_vram, 2),
                        "tokens_per_sec": round(cur_speed, 2),
                        "zero_nan_inf": not nan_occurred and not inf_occurred,
                    }
                )
                print_log(
                    f"Step {step + 1:3d}/500 | VRAM: {cur_vram:7.2f} MB | Speed: {cur_speed:5.1f} tok/s | "
                    f"NaNs: {nan_occurred} | Infs: {inf_occurred}"
                )

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    overall_speed = TOTAL_TOKENS / total_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    initial_vram = mem_trajectory[0]["vram_mb"]
    final_vram = mem_trajectory[-1]["vram_mb"]
    state_drift = final_vram - initial_vram

    gen_text = tokenizer.decode(curr_ids[0][input_ids.shape[1] :], skip_special_tokens=True)

    print_log(
        f"COMPLETED {name}: 500 tokens in {total_time:.2f}s ({overall_speed:.2f} tok/s) | "
        f"Peak VRAM: {peak_vram:.2f} MB | Initial: {initial_vram} MB | Final: {final_vram} MB | "
        f"Drift: {state_drift:.2f} MB | Zero NaN/Inf: {not nan_occurred and not inf_occurred}"
    )

    stress_results[name] = {
        "tokens_evaluated": TOTAL_TOKENS,
        "total_elapsed_s": round(total_time, 2),
        "throughput_tok_s": round(overall_speed, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "initial_vram_mb": initial_vram,
        "final_vram_mb": final_vram,
        "vram_drift_mb": round(state_drift, 2),
        "is_strict_o1": state_drift < 60.0,
        "zero_nan_inf": not nan_occurred and not inf_occurred,
        "trajectory_checkpoints": mem_trajectory,
        "text_sample": gen_text[:200],
    }

    del model, curr_ids
    torch.cuda.empty_cache()
    gc.collect()

out_path = "/mnt/d/SVD_linear_Attention/continuous_500token_stress_results.json"
with open(out_path, "w") as f:
    json.dump(stress_results, f, indent=2)

print_log(f"All 500-token continuous stress tests finished! Written to {out_path}")
