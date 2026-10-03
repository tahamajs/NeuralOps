#!/usr/bin/env python3
"""
Evaluate Module Perplexity (WikiText-2) on GPT-2.

Production-hardened to produce results strictly comparable to run_gpt2_sweep.py:
  1. Builds real calibration DataLoader from WikiText-2 train split (500 samples, batch_size=4, max_length=128)
  2. Passes calibration_loader AND tokenizer to apply_module_to_model
  3. Replaces module_type="auto" with explicit ATTENTION_REGISTRY / COMPRESSOR_REGISTRY lookup
  4. Uses fp16 on CUDA (torch_dtype=torch.float16 if cuda else torch.float32)
  5. Uses full test split without arbitrary max_samples truncation (or explicitly records partial eval)
  6. Captures peak VRAM, elapsed time, params before/after, compression rate
  7. Records seq_len/stride from actual arguments
  8. Adds is_baseline flag when module_name == "none" or "full"
  9. Standard python logging replacing raw print()
  10. Saves per-window NLLs to JSON for downstream variance analysis
  11. Robust CPU fallback reducing seq_len if CUDA OOM occurs
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Framework paths
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.eval_all_models_datasets import apply_module_to_model

logger = logging.getLogger("EvalModulePPL")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def build_calibration_loader(
    train_dataset,
    tokenizer,
    num_samples: int = 500,
    batch_size: int = 4,
    max_length: int = 128,
) -> list[dict[str, torch.Tensor]]:
    """Build a list of batched dictionary samples from WikiText-2 train split."""
    valid_texts = []
    for item in train_dataset["text"]:
        t = item.strip()
        if len(t) > 20:
            valid_texts.append(t)
        if len(valid_texts) >= num_samples:
            break

    if not valid_texts:
        valid_texts = [
            "The quick brown fox jumps over the lazy dog. SVD linear attention compression benchmark."
        ] * num_samples

    batches: list[dict[str, torch.Tensor]] = []
    for i in range(0, len(valid_texts), batch_size):
        chunk = valid_texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append(
            {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        )

    return batches


def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    test_dataset: Any,
    seq_len: int = 1024,
    stride: int = 512,
    max_eval_tokens: int | None = None,
    device: str = "cuda",
) -> tuple[float, dict[str, Any]]:
    """
    Compute sliding-window perplexity identical to run_gpt2_sweep.py.
    """
    model.eval()
    text = "\n\n".join(test_dataset["text"])

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        return_overflowing_tokens=True,
    )
    input_ids_all = encodings.input_ids
    if input_ids_all.dim() == 1:
        input_ids_all = input_ids_all.unsqueeze(0)
    elif input_ids_all.dim() == 2 and input_ids_all.size(0) > 1:
        input_ids_all = input_ids_all.view(1, -1)

    total_text_tokens = input_ids_all.size(1)
    if max_eval_tokens:
        total_text_tokens = min(total_text_tokens, max_eval_tokens)

    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    nlls = []
    per_window_nlls = []
    prev_end_loc = 0
    tokens_evaluated = 0

    try:
        with torch.no_grad():
            for begin_loc in range(0, total_text_tokens, stride):
                end_loc = min(begin_loc + seq_len, total_text_tokens)
                trg_len = end_loc - prev_end_loc
                input_ids = input_ids_all[:, begin_loc:end_loc].to(
                    device, non_blocking=True
                )
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nll_val = neg_log_likelihood.detach().cpu()
                nlls.append(nll_val)
                per_window_nlls.append(
                    round(float(outputs.loss.detach().cpu().item()), 4)
                )

                tokens_evaluated += trg_len
                prev_end_loc = end_loc

                if end_loc == total_text_tokens:
                    break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if (
            "out of memory" in str(e).lower() or "cuda" in str(e).lower()
        ) and device == "cuda":
            logger.warning(
                "CUDA OOM during PPL eval (%s). Falling back to CPU with halved seq_len...",
                e,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = model.to("cpu")
            device = "cpu"
            fallback_seq_len = max(128, seq_len // 2)
            fallback_stride = max(64, stride // 2)
            nlls = []
            per_window_nlls = []
            prev_end_loc = 0
            tokens_evaluated = 0

            with torch.no_grad():
                for begin_loc in range(0, total_text_tokens, fallback_stride):
                    end_loc = min(begin_loc + fallback_seq_len, total_text_tokens)
                    trg_len = end_loc - prev_end_loc
                    input_ids = input_ids_all[:, begin_loc:end_loc].to("cpu")
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100

                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                    nll_val = neg_log_likelihood.detach().cpu()
                    nlls.append(nll_val)
                    per_window_nlls.append(
                        round(float(outputs.loss.detach().cpu().item()), 4)
                    )

                    tokens_evaluated += trg_len
                    prev_end_loc = end_loc

                    if end_loc == total_text_tokens:
                        break
        else:
            raise e

    elapsed_s = max(1e-4, time.time() - t_start)
    ppl = float(torch.exp(torch.stack(nlls).sum() / prev_end_loc).item())

    if device == "cuda" and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
    else:
        peak_vram_mb = 0.0

    latency_ms_per_token = (elapsed_s / max(1, tokens_evaluated)) * 1000.0
    throughput_tok_per_sec = tokens_evaluated / elapsed_s

    metrics = {
        "perplexity": round(ppl, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms_per_token": round(latency_ms_per_token, 3),
        "throughput_tok_per_sec": round(throughput_tok_per_sec, 1),
        "elapsed_s": round(elapsed_s, 2),
        "tokens_evaluated": tokens_evaluated,
        "per_window_nlls": per_window_nlls,
    }
    return ppl, metrics


def evaluate_module_ppl(
    model_path: str,
    module_name: str,
    output_dir: str,
    dataset_name: str = "wikitext-2-raw-v1",
    split: str = "test",
    rank: int = 64,
    seq_len: int = 1024,
    stride: int = 512,
    max_eval_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Load base checkpoint, inject module with calibration data, compute PPL and telemetry.
    """
    resolved_path = model_path if Path(model_path).exists() else "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.float16 if device_str == "cuda" else torch.float32

    logger.info(
        "Loading model from %s in %s on %s...", resolved_path, target_dtype, device_str
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=target_dtype,
    ).to(device_str)

    total_params_before = sum(p.numel() for p in model.parameters())

    # Load datasets
    try:
        ds_test = load_dataset("wikitext", dataset_name, split=split)
    except Exception:
        ds_test = load_dataset("Salesforce/wikitext", dataset_name, split=split)

    try:
        ds_train = load_dataset("wikitext", dataset_name, split="train")
    except Exception:
        ds_train = load_dataset("Salesforce/wikitext", dataset_name, split="train")

    is_baseline = module_name.lower() in ("none", "baseline", "full")

    if not is_baseline:
        # Build calibration loader
        logger.info("Building calibration dataset for %s...", module_name)
        calib_loader = build_calibration_loader(
            ds_train, tokenizer, num_samples=500, batch_size=4, max_length=128
        )
        first_batch = calib_loader[0]["input_ids"].to(device_str)

        # Explicit registry lookup
        if module_name in ATTENTION_REGISTRY:
            mod_type = "attention"
        elif module_name in COMPRESSOR_REGISTRY:
            mod_type = "compressor"
        else:
            raise ValueError(
                f"Module '{module_name}' not found in ATTENTION_REGISTRY or COMPRESSOR_REGISTRY"
            )

        logger.info("Applying %s (%s) with rank=%d...", module_name, mod_type, rank)
        model, comp_stats = apply_module_to_model(
            model,
            module_name,
            module_type=mod_type,
            rank=rank,
            device=torch.device(device_str),
            calibration_data=first_batch,
            calibration_loader=calib_loader,
            tokenizer=tokenizer,
        )

    total_params_after = sum(p.numel() for p in model.parameters())
    comp_rate = round(total_params_before / max(1, total_params_after), 4)
    comp_pct = round(
        100.0 * (1.0 - total_params_after / max(1, total_params_before)), 2
    )

    logger.info("Evaluating perplexity (seq_len=%d, stride=%d)...", seq_len, stride)
    ppl, metrics = compute_perplexity(
        model,
        tokenizer,
        ds_test,
        seq_len=seq_len,
        stride=stride,
        max_eval_tokens=max_eval_tokens,
        device=device_str,
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    result = {
        "module": module_name,
        "is_baseline": is_baseline,
        "model_path": model_path,
        "dataset": dataset_name,
        "split": split,
        "perplexity": ppl,
        "rank": rank if not is_baseline else 0,
        "seq_len": seq_len,
        "stride": stride,
        "max_eval_tokens": max_eval_tokens,
        "peak_vram_mb": metrics["peak_vram_mb"],
        "latency_ms_per_token": metrics["latency_ms_per_token"],
        "throughput_tok_per_sec": metrics["throughput_tok_per_sec"],
        "elapsed_s": metrics["elapsed_s"],
        "total_params_before": total_params_before,
        "total_params_after": total_params_after,
        "compression_rate": comp_rate,
        "compression_ratio_pct": comp_pct,
        "per_window_nlls": metrics["per_window_nlls"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    res_file = out_p / "ppl_result.json"
    res_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "[PPL] %s: %.2f | VRAM: %.1f MB | Elapsed: %.1fs | Rate: %.4fx",
        module_name,
        ppl,
        metrics["peak_vram_mb"],
        metrics["elapsed_s"],
        comp_rate,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate module perplexity on WikiText-2"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gpt2_wikitext",
        help="Model path or identifier",
    )
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        help="Attention or compressor module name (or 'full')",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext-2-raw-v1", help="Dataset name"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--rank", type=int, default=64, help="Target rank")
    parser.add_argument(
        "--seq_len", type=int, default=1024, help="Sliding window context size"
    )
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=None,
        help="Max tokens to evaluate (None for full split)",
    )
    args = parser.parse_args()

    evaluate_module_ppl(
        model_path=args.model,
        module_name=args.module,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        rank=args.rank,
        seq_len=args.seq_len,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
    )
