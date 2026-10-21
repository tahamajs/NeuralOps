#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Synthetic arithmetic continuation stress test; not a GSM8K evaluation.
"""


def main() -> None:
    """Run this standalone evaluation experiment."""

    import gc
    import os
    import re
    import time
    import json
    import torch
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from neuralops.utils.replace_attention import replace_attention

    if not torch.cuda.is_available():
        raise RuntimeError("This arithmetic continuation stress experiment requires CUDA.")


    def print_log(msg):
        """Execute print log routine."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_log(f"Synthetic arithmetic continuation stress on: {torch.cuda.get_device_name(0)}")

    model_path = os.environ.get("NEURALOPS_MODEL_PATH", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    arithmetic_prompts = [
        {
            "id": "synthetic_arithmetic_01",
            "topic": "Compound Interest & Budgeting",
            "prompt": (
                "A warehouse starts with 450 crates. During month 1 it receives 120 and ships 85. "
                "During month 2 it receives 150 and ships 110. During month 3 it sells 40 percent "
                "of its remaining crates. How many crates remain after month 3? Give the final number."
            ),
            "target_number": "315",
        },
        {
            "id": "synthetic_arithmetic_02",
            "topic": "Multi-Step Geometric Progression",
            "prompt": (
                "A bacteria culture starts with 16 cells and doubles every 3 hours. "
                "How many cells will be present after 15 hours? Give the final number."
            ),
            "target_number": "512",
        },
        {
            "id": "synthetic_arithmetic_03",
            "topic": "Linear Recurrence Sequence",
            "prompt": (
                "A sequence is defined by u_0 = 5 and u_n = 2u_(n-1) - 3 for n >= 1. "
                "What is u_5? Give the final number."
            ),
            "target_number": "67",
        },
        {
            "id": "synthetic_arithmetic_04",
            "topic": "Rate & Work Proportion",
            "prompt": (
                "Machine A produces 40 parts per hour and Machine B produces 60 parts per hour. "
                "They work together for 4 hours, after which Machine A works alone for 3 hours. "
                "How many parts are produced in total? Give the final number."
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

    print_log("=== Synthetic Arithmetic Continuation Stress: up to 256 generated tokens ===")

    for p_data in arithmetic_prompts:
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

            exact_target_found = re.search(rf"(?<!\d){re.escape(p_target)}(?!\d)", gen_text) is not None

            zero_drift = (not nan_detected) and (not inf_detected)

            print_log(
                f"[{name}] -> Speed: {tok_s:5.1f} tok/s | Elapsed: {elapsed_s:5.2f}s | "
                f"Zero Drift: {zero_drift} (NaNs={nan_detected}, Infs={inf_detected}) | "
                f"First Output: '{first_emitted_token}' | Exact target appears in output: {exact_target_found}"
            )
            print_log(f"VRAM Checkpoints (64, 128, 192, 256 tokens): {mem_checkpoints}")

            record = {
                "problem_id": p_id,
                "evaluation_protocol": "synthetic arithmetic continuation; not GSM8K accuracy",
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
                "exact_target_found_in_generated_text": exact_target_found,
                "output_preview": gen_text[:120],
            }
            evaluation_matrix.append(record)

            del m, curr_ids
            torch.cuda.empty_cache()
            gc.collect()

    out_file = os.path.join(
        os.environ.get("NEURALOPS_OUTPUT_DIR", "."), "arithmetic_continuation_256step_results.json"
    )
    with open(out_file, "w") as f:
        json.dump(evaluation_matrix, f, indent=2)

    print_log(f"Synthetic arithmetic continuation run completed. Results written to {out_file}")



if __name__ == "__main__":
    main()
