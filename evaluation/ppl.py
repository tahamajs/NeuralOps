#!/usr/bin/env python3
"""
High-Performance Positional and Block Perplexity Evaluator.
Modeled after FlashLinearAttention / Long-Context LM evaluation protocols.
Supports evaluating fixed-length contexts (block_size, bucket_size, batch_size),
tracking position-wise NLL buckets across the context window, and streaming
evaluations on PG19, Wikitext, or custom datasets.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Framework paths
HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for p in [str(CODES), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.evaluation.eval_patcher import apply_module_to_model

logger = logging.getLogger("neuralops.evaluation.ppl")


class PerplexityEvaluator:
    """
    Perplexity and Positional NLL Evaluator.
    Computes overall perplexity and position-wise negative log-likelihood (NLL)
    bucketed over context windows of length `block_size`.
    """

    def __init__(
        self,
        model: Union[str, nn.Module] = "gpt2",
        module_name: Optional[str] = None,
        module_type: str = "attention",
        block_size: int = 1024,
        bucket_size: int = 64,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "float16",
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.device = torch.device(device if (torch.cuda.is_available() and device != "cpu") else "cpu")
        self.block_size = int(block_size)
        self.bucket_size = int(bucket_size)
        self.batch_size = int(batch_size)
        self.num_buckets = max(1, self.block_size // self.bucket_size)

        if isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            self.dtype = dtype_map.get(dtype.lower(), torch.float16 if self.device.type == "cuda" else torch.float32)
        else:
            self.dtype = dtype

        # Model Loading
        if isinstance(model, str):
            self.model_name = model
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
        else:
            self.model = model
            self.model_name = getattr(model, "name_or_path", "neuralops_model")

        self.module_name = module_name
        self.module_type = module_type
        if module_name and module_name.lower() not in ["none", "null", "baseline"]:
            logger.info(f"Applying NeuralOps {module_type}: {module_name}")
            self.model, _ = apply_module_to_model(
                self.model,
                module_name=module_name,
                module_type=module_type,
            )

        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        if tokenizer is None:
            tok_path = model if isinstance(model, str) else "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"

    def load_dataset_tokens(
        self,
        dataset_name_or_path: str = "wikitext",
        dataset_config_name: Optional[str] = "wikitext-2-raw-v1",
        split: str = "test",
        max_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Loads raw text dataset and encodes into a continuous 1D token tensor.
        """
        from datasets import load_dataset

        logger.info(f"Loading dataset: {dataset_name_or_path} (config={dataset_config_name}, split={split})")
        if dataset_config_name:
            ds = load_dataset(dataset_name_or_path, dataset_config_name, split=split)
        else:
            ds = load_dataset(dataset_name_or_path, split=split)

        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        text_data = "\n\n".join([t for t in ds[text_col] if t and t.strip()])
        
        encoded = self.tokenizer.encode(text_data, add_special_tokens=False)
        if max_tokens and len(encoded) > max_tokens:
            encoded = encoded[:max_tokens]

        tokens_tensor = torch.tensor(encoded, dtype=torch.long)
        logger.info(f"Loaded {len(tokens_tensor):,} tokens for evaluation.")
        return tokens_tensor

    def evaluate_tokens(
        self,
        tokens: torch.Tensor,
        stride: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates tokens over fixed blocks of length block_size.
        Tracks position-wise cross-entropy losses into bucket_size bins.
        """
        if stride is None:
            stride = self.block_size

        total_tokens = len(tokens)
        if total_tokens < self.block_size + 1:
            raise ValueError(f"Tokens length ({total_tokens}) must be >= block_size + 1 ({self.block_size + 1})")

        # Position-wise loss accumulator
        # Each position i in [0, block_size - 1] predicts token i + 1
        pos_losses = torch.zeros(self.block_size, dtype=torch.float64, device="cpu")
        pos_counts = torch.zeros(self.block_size, dtype=torch.int64, device="cpu")

        block_nlls: List[float] = []
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        batch_inputs: List[torch.Tensor] = []
        batch_targets: List[torch.Tensor] = []

        for idx in range(0, total_tokens - self.block_size, stride):
            chunk = tokens[idx : idx + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                break
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])

            if len(batch_inputs) == self.batch_size:
                inp_b = torch.stack(batch_inputs, dim=0).to(self.device)
                tgt_b = torch.stack(batch_targets, dim=0).to(self.device)

                with torch.no_grad():
                    logits = self.model(inp_b).logits  # [B, L, V]
                    loss = loss_fn(logits.transpose(1, 2), tgt_b)  # [B, L]
                    
                    # Accumulate positional loss
                    loss_sum = loss.sum(dim=0).cpu().to(torch.float64)  # [L]
                    b_size = inp_b.size(0)
                    pos_losses[:loss_sum.size(0)] += loss_sum
                    pos_counts[:loss_sum.size(0)] += b_size
                    
                    # Batch block nlls
                    for b_i in range(b_size):
                        block_nlls.append(float(loss[b_i].mean().item()))

                batch_inputs.clear()
                batch_targets.clear()

        # Remainder batch
        if batch_inputs:
            inp_b = torch.stack(batch_inputs, dim=0).to(self.device)
            tgt_b = torch.stack(batch_targets, dim=0).to(self.device)
            with torch.no_grad():
                logits = self.model(inp_b).logits
                loss = loss_fn(logits.transpose(1, 2), tgt_b)
                loss_sum = loss.sum(dim=0).cpu().to(torch.float64)
                b_size = inp_b.size(0)
                pos_losses[:loss_sum.size(0)] += loss_sum
                pos_counts[:loss_sum.size(0)] += b_size
                for b_i in range(b_size):
                    block_nlls.append(float(loss[b_i].mean().item()))
            batch_inputs.clear()
            batch_targets.clear()

        total_elapsed = time.perf_counter() - start_time
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0.0
        )

        valid_mask = pos_counts > 0
        total_nll = float(pos_losses[valid_mask].sum().item())
        total_eval_tokens = int(pos_counts[valid_mask].sum().item())
        avg_nll = total_nll / max(1, total_eval_tokens)
        ppl = math.exp(min(avg_nll, 100.0))

        # Position-wise bucket aggregation
        positional_nll = (pos_losses / pos_counts.clamp(min=1)).tolist()
        bucket_nlls: List[float] = []
        for b_idx in range(self.num_buckets):
            b_start = b_idx * self.bucket_size
            b_end = min((b_idx + 1) * self.bucket_size, self.block_size)
            if b_end > b_start:
                b_slice = pos_losses[b_start:b_end].sum().item() / max(1, pos_counts[b_start:b_end].sum().item())
                bucket_nlls.append(float(b_slice))

        tokens_per_sec = total_eval_tokens / max(total_elapsed, 1e-6)

        results = {
            "mean_perplexity": ppl,
            "positional_ppl_buckets": bucket_nlls,
            "model": self.model_name,
            "module_name": self.module_name or "baseline",
            "module_type": self.module_type,
            "perplexity": ppl,
            "mean_nll": avg_nll,
            "total_nll": total_nll,
            "total_tokens": total_eval_tokens,
            "block_size": self.block_size,
            "bucket_size": self.bucket_size,
            "num_blocks": len(block_nlls),
            "positional_nll": positional_nll,
            "bucket_nlls": bucket_nlls,
            "bucket_perplexities": [math.exp(min(b, 100.0)) for b in bucket_nlls],
            "throughput_tok_per_sec": tokens_per_sec,
            "elapsed_sec": total_elapsed,
            "peak_vram_mb": peak_vram_mb,
        }
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FLA-grade Positional & Block Perplexity Evaluator")
    parser.add_argument("--path", type=str, default="gpt2", help="Model path or name")
    parser.add_argument("--module-name", type=str, default=None, help="NeuralOps attention/compressor name")
    parser.add_argument("--module-type", type=str, default="attention", choices=["attention", "compressor"])
    parser.add_argument("--data", type=str, default="wikitext", help="Dataset name (wikitext or pg19)")
    parser.add_argument("--data-config", type=str, default=None, help="Dataset config name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--block_size", type=int, default=1024, help="Block context length")
    parser.add_argument("--bucket_size", type=int, default=64, help="Positional bucket size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_tokens", type=int, default=10000, help="Limit total tokens for fast eval")
    parser.add_argument("--dtype", type=str, default="float16", help="Precision")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="JSON output destination")
    args = parser.parse_args()

    cfg_name = args.data_config
    if cfg_name is None:
        if args.data == "wikitext":
            cfg_name = "wikitext-2-raw-v1"

    evaluator = PerplexityEvaluator(
        model=args.path,
        module_name=args.module_name,
        module_type=args.module_type,
        block_size=args.block_size,
        bucket_size=args.bucket_size,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )

    tokens = evaluator.load_dataset_tokens(
        dataset_name_or_path=args.data,
        dataset_config_name=cfg_name,
        split=args.split,
        max_tokens=args.max_tokens,
    )

    results = evaluator.evaluate_tokens(tokens)
    print(f"\n=== Perplexity Evaluation Results: {results['model']} ({results['module_name']}) ===")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print(f"Mean NLL:   {results['mean_nll']:.4f}")
    print(f"Throughput: {results['throughput_tok_per_sec']:.2f} tok/s")
    print(f"Peak VRAM:  {results['peak_vram_mb']:.2f} MB")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
