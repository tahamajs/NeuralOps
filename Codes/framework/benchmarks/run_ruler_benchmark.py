#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) Context Retrieval Benchmark (RULER-style) for GPT-2.

Evaluates long-context retrieval capabilities of attention and compression modules
across multiple context lengths (e.g. 512, 1024, 2048, 4096) and depths (0.1, 0.25, 0.5, 0.75, 0.9).

Features:
  1. Diverse fixed needle triples (code, city, date, element, password, etc.).
  2. Multi-depth placement across synthetic neutral haystack tokens.
  3. Automatic context clamping to model positional limits (default 1024 for standard GPT-2).
  4. Resilient OOM and exception handling (skips/records failed context lengths cleanly).
  5. JSON and Markdown summary table output.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT), str(ROOT / "Codes")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Lazy ML imports (imported in main to allow --help without heavy dependencies)
# ---------------------------------------------------------------------------
np = None
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
ATTENTION_REGISTRY = None
COMPRESSOR_REGISTRY = None
apply_module_to_model = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("RulerNIAH")

# ---------------------------------------------------------------------------
# Needle & Haystack Bank
# ---------------------------------------------------------------------------
NEEDLE_BANK: list[tuple[str, str, str]] = [
    ("The secret passcode is 7402.", "What is the secret passcode?", "7402"),
    ("The hidden city is Zephyria.", "What is the hidden city?", "Zephyria"),
    ("The magic number is 9183.", "What is the magic number?", "9183"),
    ("The important date is March 15, 2042.", "What is the important date?", "March 15, 2042"),
    ("The rare element is Vibranium.", "What is the rare element?", "Vibranium"),
    ("The key is hidden under the oak tree.", "Where is the key hidden?", "oak tree"),
    ("The password for the vault is 'tangerine'.", "What is the vault password?", "tangerine"),
    ("The scientist's name is Dr. Aris Thorne.", "What is the scientist's name?", "Aris Thorne"),
]

FILLER_SENTENCES: list[str] = [
    "The grass is green. The sky is blue. The sun is yellow. ",
    "Here we go. There and back again. ",
    "A journey of a thousand miles begins with a single step. ",
    "The quick brown fox jumps over the lazy dog. ",
    "To be or not to be, that is the ultimate question of existence. ",
    "All that glitters is not necessarily pure gold. ",
    "Actions speak far louder than any spoken words. ",
    "Practice makes perfect in every single scientific craft. ",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Needle-in-a-Haystack (NIAH) context retrieval benchmark on GPT-2 with an injected module."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/gpt2_wikitext",
        help="Path or HF repo id of pretrained model (default: models/gpt2_wikitext)",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="full",
        help="Name of registered attention or compressor module (or 'full' for baseline)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Rank for low-rank approximation (default: 64)",
    )
    parser.add_argument(
        "--context_lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Context lengths to evaluate (default: 512 1024 2048 4096)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of evaluation samples per context length (default: 20)",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
        help="Needle placement depths as ratios in [0.0, 1.0] (default: 0.1 0.25 0.5 0.75 0.9)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/07_ruler_benchmark",
        help="Directory to save evaluation results (default: results/07_ruler_benchmark)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_position",
        type=int,
        default=1024,
        help="Maximum position embedding limit for the architecture (default: 1024 for standard GPT-2)",
    )
    parser.add_argument(
        "--no_inject",
        action="store_true",
        help="If set, do not inject any module (evaluate raw baseline model)",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="If set, run smoke test with num_samples=4 and context_lengths=[512]",
    )
    return parser.parse_args(argv)


def construct_niah_prompt(
    tokenizer: Any,
    needle: str,
    question: str,
    depth: float,
    effective_length: int,
) -> list[int]:
    """
    Construct token IDs for needle inserted into haystack filler at the requested depth ratio.
    The final tokens will end with question and answer prompt.
    """
    suffix = f"\n\nQuestion: {question}\nAnswer:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    needle_ids = tokenizer.encode(f" {needle} ", add_special_tokens=False)

    budget = effective_length - len(suffix_ids) - len(needle_ids)
    if budget <= 0:
        # Prompt length too short for filler; just return needle + suffix
        prompt_ids = needle_ids + suffix_ids
        return prompt_ids[-effective_length:]

    pos = int(budget * depth)
    after_budget = max(0, budget - pos)

    # Generate filler token sequence
    filler_str = " ".join(FILLER_SENTENCES) + " "
    filler_ids = tokenizer.encode(filler_str, add_special_tokens=False)
    repeats = (budget // max(1, len(filler_ids))) + 3
    full_filler = filler_ids * repeats

    before_ids = full_filler[:pos]
    after_ids = full_filler[:after_budget]

    prompt_ids = before_ids + needle_ids + after_ids + suffix_ids
    if len(prompt_ids) > effective_length:
        prompt_ids = prompt_ids[-effective_length:]
    return prompt_ids


def evaluate_context_length(
    model: torch.nn.Module,
    tokenizer: Any,
    context_len: int,
    depths: list[float],
    num_samples: int,
    max_position: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate retrieval accuracy across num_samples for a given context length.
    Clamps effective prompt length to min(context_len, max_position).
    """
    effective_len = min(context_len, max_position)
    LOG.info("Evaluating context_len=%d (clamped effective_len=%d, samples=%d)...",
             context_len, effective_len, num_samples)

    depth_stats: dict[str, dict[str, int]] = {
        str(round(d, 4)): {"correct": 0, "total": 0} for d in depths
    }
    total_correct = 0

    for i in range(num_samples):
        needle, question, answer = NEEDLE_BANK[i % len(NEEDLE_BANK)]
        depth = depths[i % len(depths)]
        depth_key = str(round(depth, 4))

        prompt_ids = construct_niah_prompt(
            tokenizer=tokenizer,
            needle=needle,
            question=question,
            depth=depth,
            effective_length=effective_len,
        )

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = output_ids[0][input_tensor.shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            is_correct = answer.lower() in generated_text.lower()
        except torch.cuda.OutOfMemoryError as oom_err:
            LOG.warning("CUDA OOM at ctx_len=%d, sample=%d: %s", context_len, i, oom_err)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise oom_err
        except Exception as gen_err:
            LOG.warning("Generation error at ctx_len=%d, sample=%d: %s", context_len, i, gen_err)
            is_correct = False

        depth_stats[depth_key]["total"] += 1
        if is_correct:
            depth_stats[depth_key]["correct"] += 1
            total_correct += 1

        LOG.info("ctx_len=%d depth=%.2f correct=%s (expected: '%s')",
                 context_len, depth, is_correct, answer)

    accuracy = round(total_correct / max(1, num_samples), 4)
    by_depth: dict[str, float] = {}
    for d_str, counts in depth_stats.items():
        by_depth[d_str] = round(counts["correct"] / max(1, counts["total"]), 4)

    return {
        "accuracy": accuracy,
        "by_depth": by_depth,
        "num_samples": num_samples,
        "effective_context_len": effective_len,
    }


def print_summary_table(results: dict[str, Any], depths: list[float], module_name: str) -> None:
    """Print summary table formatted like: ctx_len | accuracy | depth_0.1 | depth_0.5 | depth_0.9"""
    print("\n" + "=" * 75)
    print(f" RULER NIAH RETRIEVAL BENCHMARK SUMMARY: {module_name}")
    print("=" * 75)

    rep_depths = [d for d in [0.1, 0.5, 0.9] if any(abs(d - cur) < 1e-4 for cur in depths)]
    if not rep_depths:
        rep_depths = depths[:3]

    depth_headers = [f"depth_{d}" for d in rep_depths]
    header = f"{'ctx_len':<10} | {'accuracy':<10} | " + " | ".join(f"{dh:<10}" for dh in depth_headers)
    print(header)
    print("-" * len(header))

    for ctx_len_str, res in results.items():
        acc_str = f"{res.get('accuracy', 0.0):<10.4f}" if "accuracy" in res else f"{'ERR':<10}"
        by_d = res.get("by_depth", {})
        d_vals = []
        for rd in rep_depths:
            val = by_d.get(str(round(rd, 4)), by_d.get(str(rd), 0.0))
            d_vals.append(f"{float(val):<10.4f}")
        print(f"{ctx_len_str:<10} | {acc_str} | " + " | ".join(d_vals))
    print("=" * 75 + "\n")


def main(argv: list[str] | None = None) -> int:
    global np, torch, AutoModelForCausalLM, AutoTokenizer, ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, apply_module_to_model
    args = parse_args(argv)

    try:
        import numpy as _np
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer
        from framework.core.registry import ATTENTION_REGISTRY as _ATTENTION_REGISTRY, COMPRESSOR_REGISTRY as _COMPRESSOR_REGISTRY
        from framework.eval_all_models_datasets import apply_module_to_model as _apply_module_to_model

        np = _np
        torch = _torch
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        ATTENTION_REGISTRY = _ATTENTION_REGISTRY
        COMPRESSOR_REGISTRY = _COMPRESSOR_REGISTRY
        apply_module_to_model = _apply_module_to_model
    except Exception as imp_err:
        LOG.error("Failed to import PyTorch or framework dependencies: %s", imp_err)
        return 1

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.smoke_test:
        LOG.info("Smoke test active: context_lengths=[512], num_samples=4")
        args.context_lengths = [512]
        args.num_samples = 4

    # Device verification
    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA requested but not available; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    # Load model and tokenizer
    tok_src = args.model_path if os.path.exists(args.model_path) else "gpt2"
    LOG.info("Loading tokenizer and model from '%s'...", tok_src)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(tok_src)

    # Inject module if requested
    injection_stats: dict[str, Any] = {}
    if not args.no_inject and args.module != "full":
        typ = "attention" if args.module in ATTENTION_REGISTRY else "compressor"
        LOG.info("Injecting %s module '%s' (rank=%d)...", typ, args.module, args.rank)
        try:
            model, stats = apply_module_to_model(
                model,
                args.module,
                module_type=typ,
                rank=args.rank,
                device=device,
                tokenizer=tokenizer,
            )
            if isinstance(stats, dict):
                injection_stats = stats
            elif isinstance(stats, (int, float)):
                injection_stats = {"compression_rate": float(stats)}
            else:
                injection_stats = {"raw_stats": str(stats)}
        except Exception as e:
            LOG.error("Failed to inject module '%s': %s", args.module, e, exc_info=True)
            return 2
    else:
        LOG.info("Running baseline without injection (module='%s')", args.module)

    model.to(device)
    model.eval()

    all_results: dict[str, Any] = {}

    for ctx_len in args.context_lengths:
        try:
            res = evaluate_context_length(
                model=model,
                tokenizer=tokenizer,
                context_len=ctx_len,
                depths=args.depths,
                num_samples=args.num_samples,
                max_position=args.max_position,
                device=device,
            )
            all_results[str(ctx_len)] = res
        except torch.cuda.OutOfMemoryError:
            LOG.error("OOM encountered at context length %d; skipping length.", ctx_len)
            all_results[str(ctx_len)] = {
                "accuracy": 0.0,
                "error": "CUDA OutOfMemoryError",
                "by_depth": {str(round(d, 4)): 0.0 for d in args.depths},
                "num_samples": 0,
            }
        except Exception as ex:
            LOG.error("Error at context length %d: %s", ctx_len, ex, exc_info=True)
            all_results[str(ctx_len)] = {
                "accuracy": 0.0,
                "error": str(ex),
                "by_depth": {str(round(d, 4)): 0.0 for d in args.depths},
                "num_samples": 0,
            }

    print_summary_table(all_results, args.depths, args.module)

    # Save output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.module}_niah.json"

    payload: dict[str, Any] = {
        "module": args.module,
        "model_path": args.model_path,
        "rank": args.rank,
        "context_lengths": args.context_lengths,
        "depths": args.depths,
        "num_samples": args.num_samples,
        "results": all_results,
        "config": {
            "max_position": args.max_position,
            "device": device_str,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "no_inject": args.no_inject,
            "injection_stats": injection_stats,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Saved RULER NIAH benchmark results to %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
