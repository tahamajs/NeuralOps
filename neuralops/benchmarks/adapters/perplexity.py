"""Perplexity / Language Modeling task adapter for neuralops benchmark suite."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from neuralops.benchmarks.core.data_loader import (
    get_dataset_metadata,
    load_dataset_by_name,
    make_dataloader,
)
from neuralops.benchmarks.core.evaluation import (
    evaluate_perplexity,
    measure_latency_and_throughput,
    measure_peak_memory_mb,
)
from neuralops.benchmarks.core.model_loader import (
    apply_attention_module,
    apply_compressor,
    cleanup_model,
    count_parameters,
    get_model_metadata,
    load_model_and_tokenizer,
)
from neuralops.benchmarks.result import DatasetMetadata, ModelMetadata


logger = logging.getLogger(__name__)


class PerplexityAdapter:
    """Task adapter for language modeling / perplexity benchmarks (e.g. WikiText-2, PTB)."""

    def __init__(
        self,
        module_name: str,
        module_type: str,
        model_name: str,
        dataset_name: str,
        rank: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 512,
        stride: int = 512,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.module_name = module_name
        self.module_type = module_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rank = rank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or "float32"
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.seed = seed
        self.extra_kwargs = kwargs

        self.model = None
        self.tokenizer = None
        self.eval_loader = None
        self.model_meta: Optional[ModelMetadata] = None
        self.dataset_meta: Optional[DatasetMetadata] = None

    def setup(self) -> None:
        """Load causal LM / masked LM model, tokenizer, and tokenized dataset."""
        torch.manual_seed(self.seed)

        self.model, self.tokenizer, model_cfg = load_model_and_tokenizer(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            task="lm",
        )
        self.model_meta = get_model_metadata(self.model, model_cfg)

        # Load validation / test split
        eval_split = "validation"
        try:
            tokenized_eval = load_dataset_by_name(
                self.dataset_name,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="validation",
            )
        except Exception:
            eval_split = "test"
            tokenized_eval = load_dataset_by_name(
                self.dataset_name,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="test",
            )

        self.dataset_meta = get_dataset_metadata(
            tokenized_eval,
            name=self.dataset_name,
            split=eval_split,
        )

        self.eval_loader = make_dataloader(tokenized_eval, batch_size=self.batch_size, shuffle=False)

    def apply_module(self) -> None:
        """Apply attention module replacement or compression algorithm."""
        if self.model is None:
            raise RuntimeError("Setup must be run before applying module.")

        if self.module_type == "attention":
            self.model = apply_attention_module(
                self.model,
                module_name=self.module_name,
                rank=self.rank,
                **self.extra_kwargs,
            )
        elif self.module_type == "compressor":
            self.model = apply_compressor(
                self.model,
                module_name=self.module_name,
                rank=self.rank,
                calibration_loader=self.eval_loader,
                **self.extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown module type: {self.module_type}")

        if self.model_meta is not None:
            self.model_meta.num_params = count_parameters(self.model)

    def run(self) -> Dict[str, Any]:
        """Execute setup, module injection, perplexity evaluation, and latency benchmarking."""
        self.setup()
        self.apply_module()

        # Evaluate perplexity
        eval_metrics = evaluate_perplexity(
            model=self.model,
            eval_loader=self.eval_loader,
            device=self.device,
            stride=self.stride,
        )

        # Measure latency and throughput
        latency_stats = measure_latency_and_throughput(
            model=self.model,
            dataloader=self.eval_loader,
            device=self.device,
        )

        # Memory measurement
        peak_mem = measure_peak_memory_mb(self.device)

        return {
            "model_metadata": self.model_meta,
            "dataset_metadata": self.dataset_meta,
            "train_metrics": {},
            "eval_metrics": eval_metrics,
            "latency": latency_stats,
            "peak_memory_mb": peak_mem,
        }

    def cleanup(self) -> None:
        """Clean up memory and resources."""
        if self.model is not None:
            cleanup_model(self.model)
            self.model = None
        self.eval_loader = None
        self.tokenizer = None
