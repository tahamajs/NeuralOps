#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""RULER: Retrieval Under Long-context Evaluation for Real-world language models.

Evaluates long-context effective capacity using real synthetic tasks:
single passkey retrieval, multi-key retrieval, variable tracking, and aggregation
across variable context lengths and depths with real models and tokenizers.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("RulerEvaluator")

RULER_HAYSTACK_CORPUS = [
    "The city of Geneva is situated on the shores of Lake Geneva at the foot of the Alps.",
    "Quantum computing uses the principles of superposition and entanglement to perform complex operations.",
    "Linear attention mechanisms replace quadratic softmax computation with kernel feature maps.",
    "Singular Value Decomposition decomposes any real matrix into orthogonal bases and singular values.",
    "Fast Walsh-Hadamard transform computes discrete orthogonal rotations in O(N log N) arithmetic operations.",
    "Speculative decoding accelerates autoregressive token generation by drafting candidates and verifying in parallel.",
    "Modern deep neural networks benefit substantially from low-rank weight factorization and FP8 tensor operations.",
    "The Nile River flows northward through eastern Africa and is historically among the longest rivers on Earth.",
    "Photosynthesis is the biochemical process by which green plants transform light energy into chemical energy.",
    "Distributed data parallel training synchronizes gradients across all compute nodes using all-reduce primitives.",
    "Rotary Position Embeddings incorporate relative positional information directly into query and key representations.",
    "FlashAttention computes exact softmax attention with tiling to reduce memory reads and writes to high bandwidth memory.",
]

SUPPORTED_RULER_TASKS = [
    "single_needle",
    "multi_needle",
    "variable_tracking",
]


class RulerEvaluator:
    """Standardized RULER Long-Context Evaluator.

    Evaluates effective context capacity across configurable tasks and context lengths
    using real models, real tokenizers, and real structured prompt synthesis.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        context_lengths: list[int] | None = None,
        tasks: list[str] | None = None,
        num_samples_per_task: int = 2,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        """Initialize the RULER evaluator.

        Args:
            model: Target PyTorch causal language model.
            tokenizer: HuggingFace or compatible AutoTokenizer instance.
            context_lengths: Sequence context lengths to evaluate.
            tasks: List of task names ('single_needle', 'multi_needle', 'variable_tracking').
            num_samples_per_task: Number of trials per context length and task.
            device: Target execution device.
            seed: Seed for random number generation to ensure reproducibility.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_lengths = context_lengths or [256, 512, 1024]
        self.tasks = tasks or list(SUPPORTED_RULER_TASKS)
        self.num_samples_per_task = max(1, num_samples_per_task)
        self.seed = seed

        if isinstance(device, str):
            if device == "cuda" and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self._cached_tokens: list[int] | None = None

    def _seed_rng(self) -> None:
        """Seed all RNGs for deterministic evaluation."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _get_haystack_tokens(self) -> list[int]:
        """Retrieve pre-tokenized haystack corpus to minimize string formatting overhead."""
        if self._cached_tokens is None:
            pooled: list[int] = []
            for paragraph in RULER_HAYSTACK_CORPUS:
                tokens = self.tokenizer.encode(paragraph + " ", add_special_tokens=False)
                pooled.extend(tokens)
            self._cached_tokens = pooled or [1]
        return self._cached_tokens

    def _generate_single_needle_task(self, target_len: int, depth: float = 0.5) -> tuple[list[int], str]:
        """Generate a single needle retrieval prompt and expected answer."""
        passkey = str(random.randint(10000, 99999))
        needle_str = f" Crucial notice: The secret security code is {passkey}. Memorize this number. "
        query_str = "\nQuestion: What is the secret security code? Answer: The secret security code is"

        needle_toks = self.tokenizer.encode(needle_str, add_special_tokens=False)
        query_toks = self.tokenizer.encode(query_str, add_special_tokens=False)

        budget = max(32, target_len - len(needle_toks) - len(query_toks))
        pool = self._get_haystack_tokens()
        repeats = (budget // len(pool)) + 1
        haystack = (pool * repeats)[:budget]

        insert_idx = int(len(haystack) * max(0.0, min(1.0, depth)))
        tokens = haystack[:insert_idx] + needle_toks + haystack[insert_idx:] + query_toks
        return tokens, passkey

    def _generate_multi_needle_task(self, target_len: int, depth: float = 0.5) -> tuple[list[int], str]:
        """Generate multiple needles prompt and query a specific one."""
        key1 = str(random.randint(10000, 49999))
        key2 = str(random.randint(50000, 99999))
        needle1_str = f" Note Alpha: The primary key is {key1}. "
        needle2_str = f" Note Beta: The secondary key is {key2}. "
        query_str = "\nQuestion: What is the secondary key? Answer: The secondary key is"

        n1_toks = self.tokenizer.encode(needle1_str, add_special_tokens=False)
        n2_toks = self.tokenizer.encode(needle2_str, add_special_tokens=False)
        q_toks = self.tokenizer.encode(query_str, add_special_tokens=False)

        budget = max(32, target_len - len(n1_toks) - len(n2_toks) - len(q_toks))
        pool = self._get_haystack_tokens()
        repeats = (budget // len(pool)) + 1
        haystack = (pool * repeats)[:budget]

        idx1 = int(len(haystack) * 0.25)
        idx2 = int(len(haystack) * 0.75)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        tokens = haystack[:idx1] + n1_toks + haystack[idx1:idx2] + n2_toks + haystack[idx2:] + q_toks
        return tokens, key2

    def _generate_variable_tracking_task(self, target_len: int, depth: float = 0.5) -> tuple[list[int], str]:
        """Generate variable tracking assignment chain."""
        val = str(random.randint(100, 999))
        step1 = f" Variable assignment: Var_A = {val}. "
        step2 = " Transfer: Var_B = Var_A. "
        step3 = " Output assignment: Var_C = Var_B. "
        query_str = "\nQuestion: What is the final value of Var_C? Answer: Var_C is"

        s1_toks = self.tokenizer.encode(step1, add_special_tokens=False)
        s2_toks = self.tokenizer.encode(step2, add_special_tokens=False)
        s3_toks = self.tokenizer.encode(step3, add_special_tokens=False)
        q_toks = self.tokenizer.encode(query_str, add_special_tokens=False)

        total_inject = len(s1_toks) + len(s2_toks) + len(s3_toks) + len(q_toks)
        budget = max(32, target_len - total_inject)
        pool = self._get_haystack_tokens()
        repeats = (budget // len(pool)) + 1
        haystack = (pool * repeats)[:budget]

        part_len = len(haystack) // 3
        tokens = (
            haystack[:part_len]
            + s1_toks
            + haystack[part_len : 2 * part_len]
            + s2_toks
            + haystack[2 * part_len :]
            + s3_toks
            + q_toks
        )
        return tokens, val

    def _generate_sample(self, task: str, target_len: int, depth: float = 0.5) -> tuple[list[int], str]:
        """Dispatch task generation according to specified task name."""
        if task == "multi_needle":
            return self._generate_multi_needle_task(target_len, depth)
        if task == "variable_tracking":
            return self._generate_variable_tracking_task(target_len, depth)
        return self._generate_single_needle_task(target_len, depth)

    def _run_forward_generation(self, input_ids: torch.Tensor, max_tokens: int = 6) -> str:
        """Autoregressively decode next tokens to retrieve generated answer."""
        out = input_ids
        for _ in range(max_tokens):
            logits = self.model(out)
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=-1)

        gen_tokens = out[0, input_ids.shape[1] :]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def evaluate(self) -> dict[str, Any]:
        """Execute full RULER long-context benchmark matrix.

        Returns:
            Dictionary containing accuracy_grid, task_accuracy, mean_accuracy, and records.
        """
        self._seed_rng()
        self.model.eval()
        self.model.to(self.device)

        records: list[dict[str, Any]] = []
        grid = np.zeros((len(self.context_lengths), len(self.tasks)))
        max_pos = getattr(getattr(self.model, "config", None), "max_position_embeddings", 4096) or 4096

        for i, ctx_len in enumerate(self.context_lengths):
            effective_len = min(ctx_len, max_pos - 10)
            for j, task in enumerate(self.tasks):
                successes = 0
                for trial in range(self.num_samples_per_task):
                    depth = 0.2 + (0.6 * (trial / max(1, self.num_samples_per_task)))
                    tokens, expected = self._generate_sample(task, effective_len, depth=depth)

                    if len(tokens) > max_pos - 8:
                        q_len = 16
                        tokens = tokens[: max_pos - 8 - q_len] + tokens[-q_len:]

                    input_ids = torch.tensor([tokens], device=self.device)

                    with torch.no_grad():
                        prediction = self._run_forward_generation(input_ids)

                    is_correct = expected in prediction
                    if is_correct:
                        successes += 1

                    records.append(
                        {
                            "context_length": ctx_len,
                            "task": task,
                            "trial": trial,
                            "expected": expected,
                            "prediction": prediction.strip(),
                            "correct": is_correct,
                        }
                    )

                acc = successes / self.num_samples_per_task
                grid[i, j] = acc
                logger.info("Context %d | Task %s: Accuracy=%.2f%%", ctx_len, task, acc * 100)

        task_accuracy = {task: float(np.mean(grid[:, j])) for j, task in enumerate(self.tasks)}
        context_accuracy = {str(ctx): float(np.mean(grid[i, :])) for i, ctx in enumerate(self.context_lengths)}

        return {
            "context_lengths": self.context_lengths,
            "tasks": self.tasks,
            "accuracy_grid": grid.tolist(),
            "task_accuracy": task_accuracy,
            "context_accuracy": context_accuracy,
            "mean_accuracy": float(np.mean(grid)),
            "seed": self.seed,
            "records": records,
        }


def evaluate_ruler(
    model: nn.Module,
    tokenizer: Any,
    context_lengths: list[int] | None = None,
    tasks: list[str] | None = None,
    num_samples_per_task: int = 2,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Convenience helper to instantiate and execute RulerEvaluator.

    Args:
        model: Target PyTorch causal language model.
        tokenizer: HuggingFace or compatible AutoTokenizer.
        context_lengths: Sequence lengths to test.
        tasks: Specific subset of RULER tasks.
        num_samples_per_task: Samples per task cell.
        device: Target execution device.
        seed: Random seed.

    Returns:
        Structured dictionary of evaluation results.
    """
    evaluator = RulerEvaluator(
        model=model,
        tokenizer=tokenizer,
        context_lengths=context_lengths,
        tasks=tasks,
        num_samples_per_task=num_samples_per_task,
        device=device,
        seed=seed,
    )
    return evaluator.evaluate()


__all__ = [
    "RulerEvaluator",
    "evaluate_ruler",
    "SUPPORTED_RULER_TASKS",
]
