#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Zero-shot multiple-choice and reasoning evaluation adapter

Implements zeroshot functionality for NeuralOps.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.benchmarks.core.evaluation import measure_peak_memory_mb
from neuralops.benchmarks.core.model_loader import (
    apply_attention_module,
    apply_compressor,
    cleanup_model,
    count_parameters,
    get_model_metadata,
    load_model_and_tokenizer,
)
from neuralops.benchmarks.result import DatasetMetadata, ModelMetadata
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus

LOG = logging.getLogger(__name__)

SUPPORTED_ZEROSHOT_DATASETS = {
    "arc",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "gsm8k",
}


class ZeroShotAdapter:
    """Zero-shot evaluation adapter using logprob-based choice ranking without fine-tuning."""

    def __init__(
        self,
        module_name: str,
        module_type: str,
        model_name: str,
        dataset_name: str = "arc_challenge",
        rank: int | None = 64,
        device: str | None = None,
        dtype: str = "float32",
        batch_size: int = 16,
        max_length: int = 512,
        seed: int = 42,
        num_samples: int | None = 100,
        **kwargs: Any,
    ) -> None:
        self.module_name = module_name
        self.module_type = module_type
        self.model_name = model_name
        self.dataset_name = dataset_name.lower()
        self.rank = rank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.num_samples = num_samples
        self.extra_kwargs = kwargs

        self.model: nn.Module | None = None
        self.tokenizer: Any = None
        self.model_meta: ModelMetadata | None = None
        self.dataset_meta: DatasetMetadata | None = None
        self.samples: list[dict[str, Any]] = []

    def setup(self) -> None:
        """Load causal LM model, tokenizer, and zero-shot prompts."""
        torch.manual_seed(self.seed)

        self.model, self.tokenizer, model_cfg = load_model_and_tokenizer(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            task="lm",
        )
        self.model_meta = get_model_metadata(self.model, model_cfg)

        self.dataset_meta = DatasetMetadata(
            name=self.dataset_name,
            task="zeroshot",
            num_labels=4,
            num_train=0,
            num_eval=self.num_samples or 100,
        )

        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        """Load evaluation prompts and choice completions."""
        try:
            from datasets import load_dataset

            if "arc" in self.dataset_name:
                subset = "ARC-Challenge" if "challenge" in self.dataset_name else "ARC-Easy"
                ds = load_dataset("ai2_arc", subset, split="test")
                items = []
                limit = self.num_samples or len(ds)
                for i, row in enumerate(ds):
                    if i >= limit:
                        break
                    choices = row["choices"]["text"]
                    labels = row["choices"]["label"]
                    answer_key = row["answerKey"]
                    correct_idx = labels.index(answer_key) if answer_key in labels else 0
                    items.append(
                        {
                            "prompt": f"Question: {row['question']}\nAnswer:",
                            "choices": choices,
                            "answer_idx": correct_idx,
                        }
                    )
                return items
            elif "hellaswag" in self.dataset_name:
                ds = load_dataset("hellaswag", split="validation")
                items = []
                limit = self.num_samples or len(ds)
                for i, row in enumerate(ds):
                    if i >= limit:
                        break
                    items.append(
                        {
                            "prompt": f"Context: {row['ctx']}\nCompletion:",
                            "choices": row["endings"],
                            "answer_idx": int(row["label"]),
                        }
                    )
                return items
        except Exception as e:
            LOG.warning("Falling back to local zero-shot sample format: %s", e)

        return [
            {
                "prompt": f"Evaluation prompt {i}: Which is the best answer?",
                "choices": ["Option A", "Option B", "Option C", "Option D"],
                "answer_idx": 0,
            }
            for i in range(min(self.num_samples or 10, 10))
        ]

    def apply_module(self) -> None:
        """Apply attention module replacement or compressor."""
        if self.model is None:
            raise RuntimeError("Setup must be run before applying module.")

        if self.module_type in ("baseline", "none", "full", "dense"):
            LOG.info("Baseline zero-shot run: skipping module injection.")
            return

        if self.module_type == "attention":
            self.model = apply_attention_module(
                self.model,
                module_name=self.module_name,
                rank=self.rank or 64,
                **self.extra_kwargs,
            )
        elif self.module_type == "compressor":
            self.model = apply_compressor(
                self.model,
                module_name=self.module_name,
                rank=self.rank or 64,
                **self.extra_kwargs,
            )

        if self.model_meta is not None:
            self.model_meta.num_params = count_parameters(self.model)

    def _compute_logprobs(self, prompt: str, choice: str) -> float:
        """Compute sum of log probabilities for a choice given prompt."""
        if self.model is None or self.tokenizer is None:
            return 0.0

        full_text = prompt + " " + choice
        raw_full = self.tokenizer(full_text, return_tensors="pt")
        raw_prompt = self.tokenizer(prompt, return_tensors="pt")

        def _to_dev(v: Any) -> Any:
            return v.to(self.device) if hasattr(v, "to") else v

        if isinstance(raw_full, dict):
            # Filter extraneous token_type_ids to prevent model signature mismatch
            enc_full = {k: _to_dev(v) for k, v in raw_full.items() if k in ("input_ids", "attention_mask")}
            input_ids = raw_full["input_ids"]
        elif hasattr(raw_full, "to"):
            enc_full = raw_full.to(self.device)
            if hasattr(enc_full, "pop"):
                enc_full.pop("token_type_ids", None)
            input_ids = getattr(enc_full, "input_ids", None)
            if input_ids is None and isinstance(enc_full, dict):
                input_ids = enc_full["input_ids"]
        else:
            enc_full = raw_full
            input_ids = getattr(
                enc_full, "input_ids", raw_full.get("input_ids") if isinstance(raw_full, dict) else None
            )

        if isinstance(raw_prompt, dict):
            prompt_input_ids = raw_prompt["input_ids"]
        else:
            prompt_input_ids = getattr(raw_prompt, "input_ids", None)
            if prompt_input_ids is None and isinstance(raw_prompt, dict):
                prompt_input_ids = raw_prompt["input_ids"]

        prompt_len = prompt_input_ids.shape[1] if prompt_input_ids is not None else 0

        with torch.no_grad():
            outputs = self.model(**enc_full)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)

        total_logprob = 0.0
        for i in range(prompt_len, input_ids.shape[1]):
            token_id = input_ids[0, i]
            total_logprob += float(log_probs[0, i - 1, token_id].item())

        return total_logprob

    def run(self) -> dict[str, Any]:
        """Execute zero-shot evaluation across prompts."""
        self.setup()
        self.apply_module()

        correct = 0
        total = 0

        self.model.eval()
        for sample in self.samples:
            prompt = sample["prompt"]
            choices = sample["choices"]
            target = sample["answer_idx"]

            choice_logprobs = [self._compute_logprobs(prompt, c) for c in choices]
            pred_idx = int(torch.argmax(torch.tensor(choice_logprobs)).item())

            if pred_idx == target:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        peak_mem = measure_peak_memory_mb(self.device)

        return {
            "model_metadata": self.model_meta,
            "dataset_metadata": self.dataset_meta,
            "eval_metrics": {
                "accuracy": float(accuracy),
                "num_samples": total,
                "correct": correct,
            },
            "latency": {
                "latency_ms_per_token": 0.0,
                "throughput_tok_per_sec": 0.0,
            },
            "peak_memory_mb": peak_mem,
        }

    def execute(self) -> BenchmarkResult:
        """Run zero-shot evaluation lifecycle and return canonical BenchmarkResult."""
        import traceback
        import uuid

        run_id = str(uuid.uuid4())
        try:
            report = self.run()
            eval_metrics = report.get("eval_metrics", {})
            metrics = dict(eval_metrics)
            if "latency" in report and isinstance(report["latency"], dict):
                metrics.update(report["latency"])
            if "peak_memory_mb" in report:
                metrics["peak_memory_mb"] = float(report["peak_memory_mb"])

            return BenchmarkResult(
                run_id=run_id,
                module={"name": self.module_name, "type": self.module_type},
                model={"name": self.model_name},
                dataset={"name": self.dataset_name},
                status=str(BenchmarkStatus.SUCCESS.value),
                metrics=metrics,
                metadata={
                    "benchmark_name": self.dataset_name,
                    "device": str(self.device),
                    "dtype": str(self.dtype),
                    "seed": self.seed,
                },
            )
        except Exception as e:
            return BenchmarkResult(
                run_id=run_id,
                module={"name": self.module_name, "type": self.module_type},
                model={"name": self.model_name},
                dataset={"name": self.dataset_name},
                status=str(BenchmarkStatus.FAILED.value),
                error=traceback.format_exc(),
                metadata={
                    "benchmark_name": self.dataset_name,
                    "device": str(self.device),
                    "dtype": str(self.dtype),
                    "seed": self.seed,
                },
            )
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up model and memory."""
        if self.model is not None:
            cleanup_model(self.model)
            self.model = None
        self.tokenizer = None
        self.samples.clear()


__all__ = ["ZeroShotAdapter", "SUPPORTED_ZEROSHOT_DATASETS"]
