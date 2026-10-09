"""Classification task adapter for neuralops benchmark suite."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from neuralops.benchmarks.core.data_loader import (
    get_dataset_metadata,
    get_num_labels,
    load_dataset_by_name,
    make_dataloader,
)
from neuralops.benchmarks.core.evaluation import (
    evaluate_classification,
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
from neuralops.benchmarks.core.training import finetune_classification
from neuralops.benchmarks.result import DatasetMetadata, ModelMetadata


logger = logging.getLogger(__name__)


class ClassificationAdapter:
    """Task adapter for sequence classification benchmarks (e.g. GLUE tasks)."""

    def __init__(
        self,
        module_name: str,
        module_type: str,
        model_name: str,
        dataset_name: str,
        rank: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
        epochs: int = 3,
        lr: float = 2e-5,
        seed: int = 42,
        eval_only: bool = False,
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
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.eval_only = eval_only
        self.extra_kwargs = kwargs

        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.eval_loader = None
        self.model_meta: Optional[ModelMetadata] = None
        self.dataset_meta: Optional[DatasetMetadata] = None

    def setup(self) -> None:
        """Load datasets, tokenizers, and base model with classification head."""
        torch.manual_seed(self.seed)

        # 1. Determine number of labels for dataset
        num_labels = get_num_labels(self.dataset_name)

        # 2. Load model and tokenizer with sequence classification head
        self.model, self.tokenizer, model_cfg = load_model_and_tokenizer(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            task="classification",
            num_labels=num_labels,
        )
        self.model_meta = get_model_metadata(self.model, model_cfg)

        # 3. Tokenize datasets
        tokenized_train = load_dataset_by_name(
            self.dataset_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split="train",
        )
        # Determine eval split
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
            tokenized_train,
            name=self.dataset_name,
            split=f"train+{eval_split}",
            num_train=len(tokenized_train) if hasattr(tokenized_train, "__len__") else None,
            num_eval=len(tokenized_eval) if hasattr(tokenized_eval, "__len__") else None,
        )

        self.train_loader = make_dataloader(tokenized_train, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = make_dataloader(tokenized_eval, batch_size=self.batch_size, shuffle=False)

    def apply_module(self) -> None:
        """Apply attention module replacement or compression algorithm."""
        if self.model is None:
            raise RuntimeError("Setup must be run before applying module.")

        params_before = count_parameters(self.model)
        if self.model_meta is not None:
            self.model_meta.num_params = params_before

        if self.module_type in ("baseline", "none", "full", "dense"):
            logger.info("Baseline run: skipping module injection.")
            return

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
                calibration_loader=self.train_loader,
                **self.extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown module type: {self.module_type}")

        # Update params in metadata
        params_after = count_parameters(self.model)
        if self.module_type == "compressor" and params_after >= params_before:
            logger.warning(
                "Suspicious compression: params_after (%d) >= params_before (%d) for compressor '%s'",
                params_after,
                params_before,
                self.module_name,
            )
        if self.model_meta is not None:
            self.model_meta.num_params = params_before
            self.model_meta.num_params_after = params_after

    def run(self) -> Dict[str, Any]:
        """Execute setup, module injection, training (if applicable), and evaluation."""
        self.setup()
        self.apply_module()

        train_metrics = {}
        train_duration_s = 0.0
        if not self.eval_only and self.epochs > 0:
            train_t_start = time.perf_counter()
            train_metrics = finetune_classification(
                model=self.model,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
                epochs=self.epochs,
                lr=self.lr,
                device=self.device,
            )
            train_duration_s = round(time.perf_counter() - train_t_start, 2)

        # Baseline evaluation
        eval_metrics = evaluate_classification(
            model=self.model,
            eval_loader=self.eval_loader,
            device=self.device,
        )

        # Latency and throughput
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
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "latency": latency_stats,
            "peak_memory_mb": peak_mem,
            "train_duration_s": train_duration_s,
        }

    def cleanup(self) -> None:
        """Clean up memory and resources."""
        if self.model is not None:
            cleanup_model(self.model)
            self.model = None
        self.train_loader = None
        self.eval_loader = None
        self.tokenizer = None
