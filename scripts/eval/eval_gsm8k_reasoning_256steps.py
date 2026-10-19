#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Eval gsm8k reasoning 256steps module.
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
print_log(f"GSM8K-style Multi-Step Arithmetic & 256-token Numerical Stability on: {torch.cuda.get_device_name(0)}")

model_path = "/mnt/d/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

gsm8k_prompts = [
    {
        "id": "gsm8k_01",
        "topic": "Compound Interest & Budgeting",
        "prompt": (
            "Question: A warehouse starts with 450 crates of supplies. In month 1, they receive 120 new crates and ship out 85 crates. "
            "In month 2, they receive 150 crates and ship out 110 crates. In month 3, they sell 40% of the remaining crates. "
            "How many crates remain in the warehouse at the end of month 3? "
            "Step 1: Calculate the crates after month 1: 450 + 120 - 85 = 485. "
            "Step 2: Calculate the crates after month 2: 485 + 150 - 110 = 525. "
            "Step 3: Calculate 40% of 525: 0.40 * 525 = 210 crates sold. "
            "Step 4: Subtract the sold crates: 525 - 210 = 315. "
            "Therefore, the final answer is"
        ),
        "target_number": "315",
    },
    {
        "id": "gsm8k_02",
        "topic": "Multi-Step Geometric Progression",
        "prompt": (
            "Question: A bacteria culture starts with 16 cells and doubles every 3 hours. How many cells will be present after 15 hours? "
            "Step 1: Determine the number of doubling periods: 15 / 3 = 5 periods. "
            "Step 2: Period 1: 16 * 2 = 32. "
            "Step 3: Period 2: 32 * 2 = 64. "
            "Step 4: Period 3: 64 * 2 = 128. "
            "Step 5: Period 4: 128 * 2 = 256. "
            "Step 6: Period 5: 256 * 2 = 512. "
            "Therefore, the final number of cells is"
        ),
        "target_number": "512",
    },
    {
        "id": "gsm8k_03",
        "topic": "Linear Recurrence Sequence",
        "prompt": (
            "Question: Consider a sequence defined by u_0 = 5 and u_n = 2 * u_{n-1} - 3 for n >= 1. Compute u_1, u_2, u_3, u_4: "
            "Step 1: u_1 = 2 * 5 - 3 = 7. "
            "Step 2: u_2 = 2 * 7 - 3 = 11. "
            "Step 3: u_3 = 2 * 11 - 3 = 19. "
            "Step 4: u_4 = 2 * 19 - 3 = 35. "
            "Therefore, u_5 is computed as 2 * 35 - 3 ="
        ),
        "target_number": "67",
    },
    {
        "id": "gsm8k_04",
        "topic": "Rate & Work Proportion",
        "prompt": (
            "Question: Machine A produces 40 parts per hour, while Machine B produces 60 parts per hour. "
            "Together they operate for 4 hours, and then Machine A runs alone for another 3 hours. "
            "How many total parts were produced? "
            "Step 1: Combined production rate: 40 + 60 = 100 parts/hour. "
            "Step 2: Production in first 4 hours: 100 * 4 = 400 parts. "
            "Step 3: Production by Machine A alone: 40 * 3 = 120 parts. "
            "Step 4: Total parts produced: 400 + 120 = 520. "
            "Therefore, the total output is"
        ),
        "target_number": "520",
    },
]

architectures = [
    ("GPT-2 Baseline (MHA)", None),
    ("GPT-2 + SVD Linear Attention", "svd_linear_attention"),
    ("GPT-2 + Gated Delta Net", "gated_delta_net"),
]

MAX_GEN_STEPS = 256

evaluation_matrix = []

print_log("=== Step 1: Evaluating Multi-Step Arithmetic Reasoning up to 256 Tokens ===")

for p_data in gsm8k_prompts:
    p_id = p_data["id"]
    p_topic = p_data["topic"]
    p_text = p_data["prompt"]
    p_target = p_data["target_number"]

    input_ids = tokenizer(p_text, return_tensors="pt").input_ids.to(device)
    print_log(f"\n--- Testing Problem: [{p_id}] ({p_topic}) | Prompt Tokens: {input_ids.shape[1]} ---")

    for name, key in architectures:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        m = GPT2LMHeadModel.from_pretrained(model_path)
        if key:
            replace_attention(m, attention_key=key)
        m = m.to(device).eval()

        curr_ids = input_ids.clone()
        mem_checkpoints = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        nan_detected = False
        inf_detected = False

        with torch.no_grad():
            for step in range(MAX_GEN_STEPS):
                out = m(curr_ids)
                logits = out.logits[:, -1, :]

                if torch.isnan(logits).any().item():
                    nan_detected = True
                if torch.isinf(logits).any().item():
                    inf_detected = True

                next_t = torch.argmax(logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_t], dim=-1)

                if (step + 1) in [64, 128, 192, 256]:
                    torch.cuda.synchronize()
                    mem_checkpoints.append(round(torch.cuda.memory_allocated() / (1024**2), 2))

        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        tok_s = MAX_GEN_STEPS / elapsed_s
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        gen_tokens = curr_ids[0][input_ids.shape[1] :]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        first_emitted_token = tokenizer.decode(gen_tokens[:2], skip_special_tokens=True).strip()

        is_exact_number = p_target in first_emitted_token or p_target in gen_text[:20]

        zero_drift = (not nan_detected) and (not inf_detected)

        print_log(
            f"[{name}] -> Speed: {tok_s:5.1f} tok/s | Elapsed: {elapsed_s:5.2f}s | "
            f"Zero Drift: {zero_drift} (NaNs={nan_detected}, Infs={inf_detected}) | "
            f"First Output: '{first_emitted_token}' | Matches Target ({p_target}): {is_exact_number}"
        )
        print_log(f"VRAM Checkpoints (64, 128, 192, 256 tokens): {mem_checkpoints}")

        record = {
            "problem_id": p_id,
            "topic": p_topic,
            "architecture": name,
            "prompt_length": input_ids.shape[1],
            "tokens_generated": MAX_GEN_STEPS,
            "throughput_tok_s": round(tok_s, 2),
            "total_elapsed_s": round(elapsed_s, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "vram_checkpoints_mb": mem_checkpoints,
            "zero_numerical_drift": zero_drift,
            "nan_detected": nan_detected,
            "inf_detected": inf_detected,
            "first_emitted_token": first_emitted_token,
            "target_number": p_target,
            "target_match": is_exact_number,
            "output_preview": gen_text[:120],
        }
        evaluation_matrix.append(record)

        del m, curr_ids
        torch.cuda.empty_cache()
        gc.collect()

out_file = "/mnt/d/SVD_linear_Attention/gsm8k_256step_reasoning_results.json"
with open(out_file, "w") as f:
    json.dump(evaluation_matrix, f, indent=2)

print_log(f"All GSM8K 256-step evaluations completed! Written to {out_file}")
