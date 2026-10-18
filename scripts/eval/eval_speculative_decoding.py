#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval speculative decoding module.

Author: Taha Majlesi
"""

# Author: Taha Majlesi
import json
import time

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder
from neuralops.utils.replace_attention import replace_attention


def print_log(msg):
    """Execute print log routine."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_log(f"Speculative Decoding Benchmark running on: {torch.cuda.get_device_name(0)}")

model_path = "/mnt/d/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Prompts spanning mathematical reasoning, deductive logic, and natural text
prompts = [
    "Problem: In a sequence of reasoning steps, let x_0 = 3. At each step t >= 1, x_t = 2 * x_{t-1} + 1. Compute x_1, x_2, x_3, and explain: ",
    "The primary advantages of speculative decoding in large language models are ",
    "Step-by-step mathematical proof that the sum of first n positive integers is n(n+1)/2: First, consider ",
    "Deep neural networks achieve efficient inference through model compression, singular value decomposition, and ",
]

# Baseline Target Model (Full 12-layer MHA GPT-2)
print_log("Loading full Target Model (GPT-2 Baseline MHA)...")
target_model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()


def build_svd_linear_draft(n_layers=4):
    """Build compact early-exit student draft model with SVD linear attention."""
    base = GPT2LMHeadModel.from_pretrained(model_path)
    draft_cfg = GPT2Config(
        n_layer=n_layers, n_head=base.config.n_head, n_embd=base.config.n_embd, vocab_size=base.config.vocab_size
    )
    draft = GPT2LMHeadModel(draft_cfg)
    draft.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    draft.transformer.wpe.load_state_dict(base.transformer.wpe.state_dict())
    draft.lm_head.load_state_dict(base.lm_head.state_dict())
    draft.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())
    for i in range(n_layers):
        draft.transformer.h[i].load_state_dict(base.transformer.h[i].state_dict())
    del base
    replace_attention(draft, attention_key="svd_linear_attention")
    return draft.to(device).eval()


def build_full_svd_draft():
    """Full 12-layer model surgically converted to SVD linear attention."""
    draft = GPT2LMHeadModel.from_pretrained(model_path)
    replace_attention(draft, attention_key="svd_linear_attention")
    return draft.to(device).eval()


print_log("Building draft models...")
draft_compact_svd = build_svd_linear_draft(n_layers=4)
draft_full_svd = build_full_svd_draft()


def benchmark_standard_autoregressive(target_mod, input_ids, max_new_tokens=40):
    """Measures standard baseline autoregressive greedy decoding."""
    curr_ids = input_ids.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = target_mod(curr_ids).logits[:, -1, :]
            next_t = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_t], dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s = max_new_tokens / elapsed
    return curr_ids, elapsed, tok_s


results_list = []

# Benchmark K sweep: K in [1, 2, 3, 4, 5]
k_values = [1, 2, 3, 4, 5]

print_log("=== Running Draft-and-Verify Speculative Decoding Benchmarks ===")

for p_idx, prompt in enumerate(prompts):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print_log(f"\n--- Prompt {p_idx + 1}: '{prompt[:45]}...' (len={input_ids.shape[1]}) ---")

    # 1. Baseline Target standard generation
    std_tokens, std_time, std_tok_s = benchmark_standard_autoregressive(target_model, input_ids, max_new_tokens=40)
    print_log(f"Baseline Autoregressive Target: {std_tok_s:.2f} tok/s ({std_time:.4f}s)")

    for k_val in k_values:
        # 2. Speculative Decoding with Compact 4-layer SVD Linear Draft
        decoder_compact = SelfSpeculativeDecoder(
            target_model=target_model, draft_model=draft_compact_svd, max_draft_k=k_val, adaptive_k=False
        )
        torch.cuda.synchronize()
        spec_tokens_c, stats_c = decoder_compact.generate(input_ids, max_new_tokens=40, temperature=0.0)
        torch.cuda.synchronize()

        # Verify exact token match with target greedy output
        exact_match_c = torch.equal(spec_tokens_c, std_tokens)

        # 3. Speculative Decoding with Full 12-layer SVD Linear Draft
        decoder_full = SelfSpeculativeDecoder(
            target_model=target_model, draft_model=draft_full_svd, max_draft_k=k_val, adaptive_k=False
        )
        torch.cuda.synchronize()
        spec_tokens_f, stats_f = decoder_full.generate(input_ids, max_new_tokens=40, temperature=0.0)
        torch.cuda.synchronize()
        exact_match_f = torch.equal(spec_tokens_f, std_tokens)

        speedup_real_c = stats_c.tokens_per_sec / std_tok_s
        speedup_real_f = stats_f.tokens_per_sec / std_tok_s

        print_log(
            f"K={k_val} | Compact SVD Draft: Acc={stats_c.acceptance_rate * 100:.1f}%, TargetPasses={stats_c.total_target_forward_passes}, "
            f"TokensPerPass={stats_c.speedup_over_standard}x, Tok/s={stats_c.tokens_per_sec}, ExactMatch={exact_match_c}"
        )
        print_log(
            f"K={k_val} | Full 12L SVD Draft: Acc={stats_f.acceptance_rate * 100:.1f}%, TargetPasses={stats_f.total_target_forward_passes}, "
            f"TokensPerPass={stats_f.speedup_over_standard}x, Tok/s={stats_f.tokens_per_sec}, ExactMatch={exact_match_f}"
        )

        results_list.append(
            {
                "prompt_idx": p_idx,
                "k": k_val,
                "baseline_tok_s": round(std_tok_s, 2),
                "baseline_time_s": round(std_time, 4),
                "compact_svd": {
                    "acceptance_rate": stats_c.acceptance_rate,
                    "target_passes": stats_c.total_target_forward_passes,
                    "tokens_per_pass": stats_c.speedup_over_standard,
                    "tokens_per_sec": stats_c.tokens_per_sec,
                    "speedup_ratio": round(speedup_real_c, 3),
                    "exact_match": exact_match_c,
                },
                "full_svd": {
                    "acceptance_rate": stats_f.acceptance_rate,
                    "target_passes": stats_f.total_target_forward_passes,
                    "tokens_per_pass": stats_f.speedup_over_standard,
                    "tokens_per_sec": stats_f.tokens_per_sec,
                    "speedup_ratio": round(speedup_real_f, 3),
                    "exact_match": exact_match_f,
                },
            }
        )

output_path = "/mnt/d/SVD_linear_Attention/speculative_decoding_results.json"
with open(output_path, "w") as f:
    json.dump(results_list, f, indent=2)

print_log(f"All speculative decoding benchmarks completed successfully! Written to {output_path}")
