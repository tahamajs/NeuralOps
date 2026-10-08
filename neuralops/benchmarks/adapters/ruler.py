"""RULER / Needle-in-a-Haystack long-context benchmark adapter."""
from __future__ import annotations

import logging
import random
import time
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

LOG = logging.getLogger(__name__)

SUPPORTED_CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]


class RulerAdapter:
    """Evaluates long-context effective capacity using Needle-in-a-Haystack synthetic tasks."""

    def __init__(
        self,
        module_name: str,
        module_type: str,
        model_name: str,
        dataset_name: str = "ruler",
        rank: int | None = 64,
        device: str | None = None,
        dtype: str = "float32",
        context_length: int = 4096,
        num_samples: int = 10,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.module_name = module_name
        self.module_type = module_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rank = rank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.context_length = context_length
        self.num_samples = num_samples
        self.seed = seed
        self.extra_kwargs = kwargs

        self.model: nn.Module | None = None
        self.tokenizer: Any = None
        self.model_meta: ModelMetadata | None = None
        self.dataset_meta: DatasetMetadata | None = None

    def setup(self) -> None:
        """Load target model and tokenizer with extended position context."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.model, self.tokenizer, model_cfg = load_model_and_tokenizer(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            task="lm",
        )
        self.model_meta = get_model_metadata(self.model, model_cfg)

        self.dataset_meta = DatasetMetadata(
            name=f"ruler_{self.context_length}",
            task="ruler",
            num_labels=1,
            num_train=0,
            num_eval=self.num_samples,
        )

    def apply_module(self) -> None:
        """Inject candidate attention replacement or compression algorithm."""
        if self.model is None:
            raise RuntimeError("Setup must be called before applying module.")

        if self.module_type in ("baseline", "none", "full", "dense"):
            LOG.info("Baseline RULER run: skipping module injection.")
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

    def _generate_synthetic_niah(self, depth: float) -> tuple[str, str, str]:
        """Generate needle, haystack, and query question at a relative context depth."""
        secret_number = str(random.randint(100000, 999999))
        needle = f" The special magic passcode is {secret_number}. Remember this passcode."
        haystack_filler = " The grass is green. The sky is blue. Numbers and facts surround us every day."

        # Estimate repetitions to reach desired context token length
        tokens_per_filler = max(len(self.tokenizer(haystack_filler).input_ids), 1) if self.tokenizer else 15
        needed_fillers = max(self.context_length // tokens_per_filler, 20)

        insertion_index = int(needed_fillers * depth)
        parts = [haystack_filler] * needed_fillers
        parts.insert(insertion_index, needle)

        prompt = "".join(parts) + "\nQuestion: What is the special magic passcode?\nAnswer: The special magic passcode is"
        return prompt, secret_number, secret_number

    def run(self) -> dict[str, Any]:
        """Execute Needle-in-a-Haystack sweeps across various depths."""
        self.setup()
        self.apply_module()

        depths = [0.1, 0.25, 0.5, 0.75, 0.9]
        found_count = 0
        total_tests = 0
        total_latency_s = 0.0

        self.model.eval()
        for i in range(self.num_samples):
            depth = depths[i % len(depths)]
            prompt, expected, _ = self._generate_synthetic_niah(depth)

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.context_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            start_t = time.perf_counter()
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                ) if hasattr(self.model, "generate") else None

            duration = time.perf_counter() - start_t
            total_latency_s += duration

            if gen_out is not None:
                decoded = self.tokenizer.decode(gen_out[0], skip_special_tokens=True)
                if expected in decoded:
                    found_count += 1
            total_tests += 1

        accuracy = found_count / max(total_tests, 1)
        peak_mem = measure_peak_memory_mb(self.device)
        avg_latency_ms = (total_latency_s / max(total_tests, 1)) * 1000.0

        return {
            "model_metadata": self.model_meta,
            "dataset_metadata": self.dataset_meta,
            "eval_metrics": {
                "accuracy": float(accuracy),
                "needle_found_rate": float(accuracy),
                "context_length": self.context_length,
                "num_samples": total_tests,
            },
            "latency": {
                "latency_ms_per_pass": float(avg_latency_ms),
                "throughput_tok_per_sec": float((self.context_length * total_tests) / max(total_latency_s, 1e-6)),
            },
            "peak_memory_mb": peak_mem,
        }

    def cleanup(self) -> None:
        """Clean up model and memory."""
        if self.model is not None:
            cleanup_model(self.model)
            self.model = None
        self.tokenizer = None


__all__ = ["RulerAdapter", "SUPPORTED_CONTEXT_LENGTHS"]
