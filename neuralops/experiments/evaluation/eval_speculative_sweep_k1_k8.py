#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Eval speculative sweep k1 k8 module.
"""


def main() -> None:
    """Run this standalone evaluation experiment."""

    import time
    import os
    import json
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
    from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder
    from neuralops.core.cuda_kernels import speculative_verify
    from neuralops.utils.replace_attention import replace_attention


    def print_log(msg):
        """Execute print log routine."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_log(f"Speculative Decoding Empirical Benchmark on: {torch.cuda.get_device_name(0)}")

    d_mock = torch.tensor([[1, 2, 3, 4]], device=device)
    t_mock = torch.tensor([[1, 2, 5, 4]], device=device)
    mask, accepted_cnt = speculative_verify(d_mock, t_mock)
    print_log(f"Native CUDA speculative_verify kernel check: mask={mask.tolist()}, accepted_cnt={accepted_cnt.item()}")

    model_path = os.environ.get("NEURALOPS_MODEL_PATH", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        (
            "Reasoning_Induction",
            "Problem: In a sequence of reasoning steps, let x_0 = 3. At each step t >= 1, x_t = 2 * x_{t-1} + 1. Compute x_1, x_2, x_3, and explain: ",
        ),
        ("Speculative_Theory", "The primary advantages of speculative decoding in large language models are "),
        (
            "Math_Proof",
            "Step-by-step mathematical proof that the sum of first n positive integers is n(n+1)/2: First, consider ",
        ),
        (
            "System_Design",
            "Deep neural networks achieve efficient inference through model compression, singular value decomposition, and ",
        ),
    ]

    print_log("Loading Target Model (GPT-2 Baseline Full Attention, 12 layers, 12 heads, 768 dim)...")
    target_model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()


    def build_draft_model(n_layers=4):
        """Build student draft model with SVD linear attention."""
        base = GPT2LMHeadModel.from_pretrained(model_path)
        cfg = GPT2Config(
            n_layer=n_layers, n_head=base.config.n_head, n_embd=base.config.n_embd, vocab_size=base.config.vocab_size
        )
        draft = GPT2LMHeadModel(cfg)
        draft.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
        draft.transformer.wpe.load_state_dict(base.transformer.wpe.state_dict())
        draft.lm_head.load_state_dict(base.lm_head.state_dict())
        draft.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())
        for i in range(n_layers):
            draft.transformer.h[i].load_state_dict(base.transformer.h[i].state_dict())
        del base
        replace_attention(draft, attention_key="svd_linear_attention")
        return draft.to(device).eval()


    print_log("Building SVD Linear Attention Draft Models (Compact 4-Layer and Ultra-Compact 2-Layer)...")
    draft_4l = build_draft_model(n_layers=4)
    draft_2l = build_draft_model(n_layers=2)


    def standard_autoregressive_greedy(model, input_ids, max_new_tokens=32):
        """Runs baseline target autoregressive greedy generation and measures latency."""
        curr_ids = input_ids.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(curr_ids).logits[:, -1, :]
                next_t = torch.argmax(logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_t], dim=-1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_s = max_new_tokens / elapsed
        return curr_ids, elapsed, tok_s


    K_VALUES = list(range(1, 9))
    MAX_NEW_TOKENS = 32

    benchmark_matrix = []

    print_log("=== Executing Full Matrix Sweep across K in [1..8] ===")

    for p_name, prompt_text in prompts:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        print_log(f"\nEvaluating Prompt: [{p_name}] (len={input_ids.shape[1]})")

        target_tokens, target_time, target_tok_s = standard_autoregressive_greedy(
            target_model, input_ids, max_new_tokens=MAX_NEW_TOKENS
        )
        print_log(f"Baseline Target: {target_tok_s:.2f} tok/s ({target_time * 1000:.2f} ms total)")

        for k in K_VALUES:
            for d_name, d_model in [("SVD_Linear_4L", draft_4l), ("SVD_Linear_2L", draft_2l)]:
                decoder = SelfSpeculativeDecoder(
                    target_model=target_model, draft_model=d_model, max_draft_k=k, adaptive_k=False
                )
                torch.cuda.synchronize()
                t_spec_start = time.perf_counter()
                spec_tokens, stats = decoder.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
                torch.cuda.synchronize()
                t_spec_elapsed = time.perf_counter() - t_spec_start

                exact_match = torch.equal(spec_tokens, target_tokens)

                wall_clock_speedup = target_time / t_spec_elapsed

                record = {
                    "prompt": p_name,
                    "k": k,
                    "draft_model": d_name,
                    "baseline_tok_s": round(target_tok_s, 2),
                    "baseline_latency_s": round(target_time, 4),
                    "spec_tok_s": stats.tokens_per_sec,
                    "spec_latency_s": round(t_spec_elapsed, 4),
                    "acceptance_rate": round(stats.acceptance_rate, 4),
                    "target_forward_passes": stats.total_target_forward_passes,
                    "tokens_per_pass": stats.speedup_over_standard,
                    "wall_clock_speedup": round(wall_clock_speedup, 3),
                    "exact_bit_match": exact_match,
                }
                benchmark_matrix.append(record)

                print_log(
                    f"K={k} [{d_name}]: Acc={stats.acceptance_rate * 100:.1f}%, TargetPasses={stats.total_target_forward_passes}, "
                    f"Tok/Pass={stats.speedup_over_standard}x, Speed={stats.tokens_per_sec:.1f} tok/s, "
                    f"Speedup={wall_clock_speedup:.2f}x, ExactMatch={exact_match}"
                )

    out_file = os.path.join(os.environ.get("NEURALOPS_OUTPUT_DIR", "."), "speculative_benchmark_k1_k8_results.json")
    with open(out_file, "w") as f:
        json.dump(benchmark_matrix, f, indent=2)

    print_log(f"All benchmarks successfully finished! Results dumped to {out_file}")



if __name__ == "__main__":
    main()
