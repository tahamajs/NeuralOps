"""Universal Task Adapters supporting all 14 Hugging Face task types."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Type
import torch
import torch.nn as nn
from tqdm import tqdm

from neuralops.benchmarks.core.evaluation import (
    measure_latency_and_throughput,
    measure_peak_memory_mb,
)
from neuralops.benchmarks.core.model_loader import (
    apply_attention_module,
    apply_compressor,
    cleanup_model,
)
from neuralops.benchmarks.data.loader import (
    load_universal_dataset,
    make_universal_dataloader,
)
from neuralops.benchmarks.metrics.evaluator import compute_task_metrics
from neuralops.benchmarks.models.loader import (
    inspect_model_metadata,
    load_universal_model,
)

LOG = logging.getLogger(__name__)

TASK_ADAPTER_REGISTRY: Dict[str, Type["TaskAdapter"]] = {}


def register_task(task_name: str) -> Callable[[Type["TaskAdapter"]], Type["TaskAdapter"]]:
    """Decorator to register a task adapter."""
    def decorator(cls: Type["TaskAdapter"]) -> Type["TaskAdapter"]:
        TASK_ADAPTER_REGISTRY[task_name.lower()] = cls
        return cls
    return decorator


class TaskAdapter:
    """Base class for universal task execution adapters."""

    def __init__(
        self,
        module_name: str,
        module_type: str,
        model_name: str,
        dataset_name: str,
        rank: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32",
        batch_size: int = 32,
        max_length: int = 128,
        epochs: int = 3,
        lr: float = 2e-5,
        seed: int = 42,
        eval_only: bool = False,
        task: str = "classification",
        **kwargs: Any,
    ) -> None:
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
        self.task = task
        self.extra_kwargs = kwargs

        self.model: Optional[nn.Module] = None
        self.tokenizer: Any = None
        self.config: Any = None
        self.train_loader: Any = None
        self.eval_loader: Any = None
        self.model_meta: Any = None
        self.dataset_meta: Any = None

    def setup(self) -> None:
        """Load model, tokenizer, and dataset."""
        torch.manual_seed(self.seed)

        self.model, self.tokenizer, self.config = load_universal_model(
            model_name=self.model_name,
            task=self.task,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.extra_kwargs.get("trust_remote_code", True),
            hf_token=self.extra_kwargs.get("hf_token"),
            tokenizer_name=self.extra_kwargs.get("tokenizer"),
        )
        self.model_meta = inspect_model_metadata(self.model, self.config, self.model_name)

        # Ingest datasets
        train_ds, _ = load_universal_dataset(
            dataset_name_or_path=self.dataset_name,
            tokenizer=self.tokenizer,
            task=self.task,
            split="train",
            max_length=self.max_length,
            max_samples=self.extra_kwargs.get("max_samples"),
            text_column=self.extra_kwargs.get("text_column"),
            label_column=self.extra_kwargs.get("label_column"),
            language=self.extra_kwargs.get("language"),
        )
        val_ds, ds_meta = load_universal_dataset(
            dataset_name_or_path=self.dataset_name,
            tokenizer=self.tokenizer,
            task=self.task,
            split="validation" if "validation" in ("validation", "test") else "test",
            max_length=self.max_length,
            max_samples=min(self.extra_kwargs.get("max_samples") or 1000, 1000),
            text_column=self.extra_kwargs.get("text_column"),
            label_column=self.extra_kwargs.get("label_column"),
            language=self.extra_kwargs.get("language"),
        )
        self.dataset_meta = ds_meta
        self.train_loader = make_universal_dataloader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = make_universal_dataloader(val_ds, batch_size=self.batch_size, shuffle=False)

    def apply_module(self) -> None:
        """Inject candidate attention replacement or compressor."""
        if self.model is None:
            raise RuntimeError("Setup must be run before applying module.")

        if self.module_type in ("baseline", "none", "full", "dense"):
            LOG.info("Baseline run: skipping module injection.")
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
                calibration_loader=self.train_loader,
                **self.extra_kwargs,
            )

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Run single fine-tuning epoch."""
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in self.train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device) if len(batch) > 2 else input_ids

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if steps >= 200:  # Bound steps for fast sweeps
                break
        return total_loss / max(steps, 1)

    def evaluate(self) -> Dict[str, float]:
        """Compute task evaluation metrics on validation set."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device) if len(batch) > 2 else input_ids

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                total_loss += loss.item()
                steps += 1

                logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.detach().cpu().numpy().tolist())

        metrics = compute_task_metrics(self.task, all_preds, all_targets)
        metrics["loss"] = total_loss / max(steps, 1)
        return metrics

    def run(self) -> Dict[str, Any]:
        """Execute full benchmark pipeline."""
        self.setup()
        self.apply_module()

        train_duration_s = 0.0
        train_metrics = {}
        training_type = self.extra_kwargs.get("training_type", "eval_only" if self.eval_only else "finetune")

        if not self.eval_only and self.epochs > 0 and training_type != "none":
            t_start = time.perf_counter()
            from neuralops.benchmarks.core.training import train_with_training_type
            teacher_model = None
            if training_type in ("kd", "distillation"):
                teacher_name = self.extra_kwargs.get("kd_teacher", self.model_name)
                from neuralops.benchmarks.models.loader import load_universal_model
                teacher_model, _, _ = load_universal_model(
                    model_name=teacher_name,
                    task=self.task,
                    device=self.device,
                    dtype=self.dtype,
                )
                teacher_model.eval()

            train_metrics = train_with_training_type(
                model=self.model,
                train_loader=self.train_loader,
                training_type=training_type,
                epochs=self.epochs,
                lr=self.lr,
                device=self.device,
                teacher=teacher_model,
                kd_temperature=self.extra_kwargs.get("kd_temperature", 2.0),
                kd_alpha=self.extra_kwargs.get("kd_alpha", 0.5),
                recovery_steps=self.extra_kwargs.get("recovery_steps", 50),
            )
            train_duration_s = round(time.perf_counter() - t_start, 2)
            if teacher_model is not None:
                del teacher_model

        eval_metrics = self.evaluate()
        latency_info = measure_latency_and_throughput(self.model, self.eval_loader, self.device)
        peak_mem = measure_peak_memory_mb(self.device)

        return {
            "model_metadata": self.model_meta,
            "dataset_metadata": self.dataset_meta,
            "train_metrics": {},
            "eval_metrics": eval_metrics,
            "latency": latency_info,
            "peak_memory_mb": peak_mem,
            "train_duration_s": train_duration_s,
        }

    def cleanup(self) -> None:
        """Clean up memory and allocations."""
        if self.model is not None:
            cleanup_model(self.model)
            self.model = None
        self.train_loader = None
        self.eval_loader = None
        self.tokenizer = None


@register_task("classification")
class UniversalClassificationAdapter(TaskAdapter):
    pass


@register_task("question_answering")
@register_task("qa")
class QuestionAnsweringAdapter(TaskAdapter):
    pass


@register_task("summarization")
class SummarizationAdapter(TaskAdapter):
    pass


@register_task("translation")
class TranslationAdapter(TaskAdapter):
    pass


@register_task("instruction_tuning")
@register_task("instruction-tuning")
class InstructionTuningAdapter(TaskAdapter):
    pass


@register_task("language_modeling")
@register_task("lm")
class LanguageModelingAdapter(TaskAdapter):
    pass


@register_task("regression")
class RegressionAdapter(TaskAdapter):
    pass


@register_task("multiple_choice")
class MultipleChoiceAdapter(TaskAdapter):
    pass


def select_universal_task_adapter(task_name: str) -> Type[TaskAdapter]:
    """Resolve matching TaskAdapter from registry."""
    task_clean = task_name.lower().replace("-", "_")
    for key, adapter_cls in TASK_ADAPTER_REGISTRY.items():
        if key in task_clean:
            return adapter_cls
    return UniversalClassificationAdapter


__all__ = [
    "TaskAdapter",
    "register_task",
    "UniversalClassificationAdapter",
    "QuestionAnsweringAdapter",
    "SummarizationAdapter",
    "TranslationAdapter",
    "InstructionTuningAdapter",
    "LanguageModelingAdapter",
    "RegressionAdapter",
    "MultipleChoiceAdapter",
    "select_universal_task_adapter",
    "TASK_ADAPTER_REGISTRY",
]
