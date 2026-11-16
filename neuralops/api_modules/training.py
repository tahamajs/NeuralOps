# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transplantation, healing, training, distillation, GRPO, and dataset preparation:
surgical_transplant_and_heal_pipeline, heal_distributed, train, distill,
evaluate, load_and_prepare_dataset, train_grpo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.api_modules.attention_surgery import replace_attention

logger = logging.getLogger("NeuralOps.API")


def surgical_transplant_and_heal_pipeline(
    teacher_model: nn.Module,
    student_base_model: Optional[nn.Module] = None,
    target_attention: str = "gated_delta_net",
    calibration_inputs: Any = None,
    heal_steps: int = 25,
    heal_lr: float = 2e-3,
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """End-to-end surgical transplant, closed-form Ridge alignment, and micro-healing pipeline.

    Executes:
      1. replace_attention on student model (raw replacement baseline)
      2. calibrate_attention_subspace (closed-form analytical Ridge projection alignment)
      3. heal_replaced_model (layerwise local gradient micro-healing with best-state rollback)
      4. Optional save_surgical_checkpoint serialization
    """
    import copy

    from neuralops.core._attention_surgery import heal_replaced_model

    if student_base_model is None:
        student = copy.deepcopy(teacher_model)
    else:
        student = student_base_model

    call_kwargs = dict(kwargs)
    if (
        "embed_dim" not in call_kwargs
        and "d_model" not in call_kwargs
        and "hidden_size" not in call_kwargs
        and "dim" not in call_kwargs
    ):
        t_dim = getattr(
            teacher_model, "dim", getattr(teacher_model, "hidden_size", getattr(teacher_model, "d_model", None))
        )
        if t_dim is None and hasattr(teacher_model, "config"):
            t_dim = getattr(teacher_model.config, "hidden_size", getattr(teacher_model.config, "d_model", None))
        if t_dim is None:
            if (
                calibration_inputs is not None
                and isinstance(calibration_inputs, torch.Tensor)
                and calibration_inputs.ndim >= 2
            ):
                t_dim = calibration_inputs.shape[-1]
            elif (
                isinstance(calibration_inputs, (list, tuple))
                and len(calibration_inputs) > 0
                and isinstance(calibration_inputs[0], torch.Tensor)
            ):
                t_dim = calibration_inputs[0].shape[-1]
        if t_dim is not None:
            call_kwargs["embed_dim"] = t_dim

    student = replace_attention(student, attention=target_attention, **call_kwargs)

    initial_metrics: Dict[str, float] = {}
    if calibration_inputs is not None:
        try:
            student.eval()
            teacher_model.eval()
            with torch.no_grad():
                inp = calibration_inputs if not isinstance(calibration_inputs, (list, tuple)) else calibration_inputs[0]
                y_t = teacher_model(inp) if not isinstance(inp, dict) else teacher_model(**inp)
                y_s = student(inp) if not isinstance(inp, dict) else student(**inp)
                if isinstance(y_t, tuple):
                    y_t = y_t[0]
                if isinstance(y_s, tuple):
                    y_s = y_s[0]
                mse_raw = F.mse_loss(y_s.float(), y_t.float()).item()
                cos_raw = F.cosine_similarity(y_s.float().flatten(), y_t.float().flatten(), dim=0).item()
                initial_metrics = {"mse_raw": mse_raw, "cos_sim_raw": cos_raw}
        except (RuntimeError, ValueError, TypeError, AttributeError) as e_metric:
            logger.debug("Raw metric check skipped: %s", e_metric)

    heal_stats: Dict[str, Any] = {}
    if calibration_inputs is not None:
        heal_stats = heal_replaced_model(
            teacher_model=teacher_model,
            student_model=student,
            calibration_inputs=calibration_inputs,
            steps=heal_steps,
            lr=heal_lr,
            also_ridge_init=True,
        )

    saved_path = None
    if save_dir is not None:
        from neuralops.models.surgical_io import save_surgical_checkpoint

        saved_path = save_surgical_checkpoint(
            model=student,
            save_dir=save_dir,
            metadata={"target_attention": target_attention, "heal_stats": heal_stats},
        )

    return {
        "model": student,
        "initial_metrics": initial_metrics,
        "heal_stats": heal_stats,
        "saved_path": str(saved_path) if saved_path is not None else None,
    }


def heal_distributed(
    student: nn.Module,
    teacher: nn.Module,
    calibration_loader: Any,
    steps: int = 50,
    lr: float = 5e-5,
    parallel_healing: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Knowledge distillation healing for post-transformation recovery.

    If parallel_healing is True and a distributed context is active with >1 GPU,
    uses FSDP2Distillation to shard student and teacher across devices.
    Otherwise falls back to single-device heal_replaced_model.
    """
    if parallel_healing:
        try:
            from neuralops.distributed.context import DistributedContext
            from neuralops.distributed.fsdp2_distill import FSDP2Distillation

            ctx = DistributedContext()
            if ctx.is_distributed():
                distill_impl = FSDP2Distillation(context=ctx)
                return distill_impl.heal_replaced_model_distributed(
                    student=student,
                    teacher=teacher,
                    calibration_loader=calibration_loader,
                    steps=steps,
                    lr=lr,
                    **kwargs,
                )
        except Exception as exc:
            logger.error(
                "Parallel healing explicitly requested but failed: %s",
                exc,
            )
            raise RuntimeError(f"Parallel healing failed: {exc}") from exc

    from neuralops.core._attention_surgery import heal_replaced_model

    return heal_replaced_model(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=calibration_loader,
        steps=steps,
        lr=lr,
        **kwargs,
    )


def train(
    model: nn.Module,
    dataset: Any,
    epochs: int = 3,
    lr: float = 2e-5,
    eval_dataset: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train or fine-tune a model on a dataset.

    Args:
        model: PyTorch model to train.
        dataset: Training dataset or DataLoader.
        epochs: Number of training epochs (default: 3).
        lr: Learning rate (default: 2e-5).
        eval_dataset: Optional evaluation dataset or DataLoader.
        device: Target execution device.
        seed: Random seed for reproducible training (default: 42).
        **kwargs: Additional training arguments.

    Returns:
        Dictionary containing training telemetry and metrics.
    """
    from torch.utils.data import DataLoader

    from neuralops.benchmarks.core.training import finetune_classification
    from neuralops.core.utils.general import set_seed

    resolved_seed = 42 if seed is None else seed
    set_seed(resolved_seed)
    logger.info("Starting training run (epochs=%d, lr=%s, seed=%s)", epochs, lr, resolved_seed)

    train_loader = (
        dataset
        if isinstance(dataset, DataLoader)
        else DataLoader(dataset, batch_size=kwargs.pop("batch_size", 32), shuffle=True)
    )
    eval_loader = (
        eval_dataset
        if (eval_dataset is None or isinstance(eval_dataset, DataLoader))
        else DataLoader(eval_dataset, batch_size=kwargs.pop("eval_batch_size", 32))
    )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return finetune_classification(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=epochs,
        lr=lr,
        device=dev,
        seed=resolved_seed,
        **kwargs,
    )


def distill(
    student: nn.Module,
    teacher: nn.Module,
    dataset: Any,
    epochs: int = 3,
    lr: float = 2e-5,
    device: Optional[Union[str, torch.device]] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    seed: Optional[int] = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute knowledge distillation from teacher to student.

    Args:
        student: Student model to train.
        teacher: Teacher model providing soft target distributions.
        dataset: Training dataset or DataLoader.
        epochs: Number of distillation epochs (default: 3).
        lr: Learning rate (default: 2e-5).
        device: Target device.
        temperature: Softmax temperature for KD loss (default: 2.0).
        alpha: Weight for KD loss vs task loss (default: 0.5).
        seed: Random seed for reproducible distillation (default: 42).
        **kwargs: Additional training arguments.

    Returns:
        Dictionary containing distillation metrics.
    """
    from torch.utils.data import DataLoader

    from neuralops.core.utils.general import set_seed
    from neuralops.trainer.distillation.multi_level_kd import distill as _canonical_distill

    resolved_seed = 42 if seed is None else seed
    set_seed(resolved_seed)
    logger.info(
        "Starting knowledge distillation run (epochs=%d, lr=%s, alpha=%s, seed=%s)", epochs, lr, alpha, resolved_seed
    )

    train_loader = (
        dataset
        if isinstance(dataset, DataLoader)
        else DataLoader(dataset, batch_size=kwargs.pop("batch_size", 32), shuffle=True)
    )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return _canonical_distill(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        device=dev,
        temperature=temperature,
        alpha=alpha,
        seed=resolved_seed,
        **kwargs,
    )


def evaluate(
    model: nn.Module,
    dataset: Any = "sst2",
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 16,
    seq_len: int = 128,
    metric_fn: Optional[Any] = None,
    seed: Optional[int] = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    # Rule: no function in api.py or core/compression_pipeline.py mutates an input dict or batch;
    # if it needs to change one, it copies first or reads non-destructively.
    """Evaluate a named benchmark dataset or an arbitrary PyTorch loader.

    Args:
        model: Model to evaluate.
        dataset: Dataset identifier or a ``DataLoader``/iterable of batches.
        device: Execution device.
        batch_size: Evaluation batch size.
        seq_len: Sequence length.
        metric_fn: Optional callable receiving ``(predictions, labels)``.
        **kwargs: Extra evaluation parameters for the named benchmark path.

    Returns:
        Dictionary containing evaluation metrics (accuracy, latency, memory, etc.).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from neuralops.core.utils.general import set_seed

    resolved_seed = 42 if seed is None else seed
    set_seed(resolved_seed)
    logger.info("Starting evaluation on device %s (batch_size=%d, seq_len=%d)", dev, batch_size, seq_len)
    if not isinstance(dataset, str):
        from torch.utils.data import DataLoader

        loader = dataset if isinstance(dataset, DataLoader) else DataLoader(dataset, batch_size=batch_size)
        model = model.to(dev)
        model.eval()
        totals: Dict[str, float] = {}
        samples = 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    labels = batch.get("labels", batch.get("label", None))
                    inputs = {
                        key: value.to(dev) if hasattr(value, "to") else value
                        for key, value in batch.items()
                        if key not in ("labels", "label")
                    }
                elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, labels = {"input_ids": batch[0].to(dev)}, batch[-1].to(dev)
                else:
                    raise TypeError("Evaluation batches must be mappings or (inputs, labels) tuples.")
                outputs = model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0]
                predictions = logits.argmax(dim=-1) if logits.ndim > 1 else logits
                count = int(labels.shape[0]) if labels is not None and hasattr(labels, "shape") else 0
                samples += count
                if metric_fn is not None:
                    metrics = metric_fn(predictions, labels)
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            totals[key] = totals.get(key, 0.0) + float(value) * max(count, 1)
                    else:
                        totals["metric"] = totals.get("metric", 0.0) + float(metrics) * max(count, 1)
                elif labels is not None:
                    totals["accuracy"] = totals.get("accuracy", 0.0) + float((predictions == labels).sum())
        divisor = max(samples, 1)
        if metric_fn is not None:
            totals = {key: value / divisor for key, value in totals.items()}
        elif samples:
            totals["accuracy"] /= samples
        return {"metrics": totals, "num_samples": samples, "device": str(dev)}

    from neuralops.evaluation.eval_all_models_datasets import evaluate_single_combination_opt

    result = evaluate_single_combination_opt(
        track=kwargs.pop("track", "track1_svd"),
        module_name=kwargs.pop("module_name", "standard"),
        module_type=kwargs.pop("module_type", "attention"),
        model_family=kwargs.pop("model_family", "distilbert"),
        dataset=dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        device=torch.device(dev),
        **kwargs,
    )
    return result.to_dict() if hasattr(result, "to_dict") else dict(result.__dict__)


def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer: Any,
    max_length: int = 128,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Load a supported HF dataset and return train/evaluation DataLoaders.

    ``dataset_name`` accepts GLUE aliases such as ``sst2`` and Wikitext
    aliases such as ``wikitext2``. Dataset downloading remains lazy and only
    occurs when this helper is called.
    """
    from neuralops.benchmarks.core.data_loader import load_dataset_by_name, make_dataloader

    train_dataset = load_dataset_by_name(
        dataset_name, tokenizer, max_length=max_length, split="train", max_samples=max_samples
    )
    eval_split = (
        "validation"
        if dataset_name.lower().replace("glue_", "")
        in {"sst2", "mrpc", "mnli", "qqp", "qnli", "rte", "cola", "stsb", "wnli"}
        else "test"
    )
    eval_dataset = load_dataset_by_name(
        dataset_name, tokenizer, max_length=max_length, split=eval_split, max_samples=max_samples
    )
    return (
        make_dataloader(train_dataset, batch_size=batch_size, shuffle=True),
        make_dataloader(eval_dataset, batch_size=eval_batch_size or batch_size, shuffle=False),
    )


def train_grpo(
    model: nn.Module,
    reward_funcs: Any,
    dataset: Any,
    training_args: Any = None,
    **kwargs: Any,
) -> Any:
    """Train with TRL's GRPO trainer when the optional dependency is present.

    The import is intentionally lazy so the core NeuralOps import does not
    require TRL. ``training_args`` may be a ``GRPOConfig``; otherwise keyword
    arguments are passed to a newly constructed config.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("train_grpo requires the optional 'trl' package.") from exc

    return_trainer = bool(kwargs.pop("return_trainer", False))
    args = training_args
    if args is None:
        config_kwargs = kwargs.pop("grpo_config", {})
        args = GRPOConfig(output_dir=kwargs.pop("output_dir", "./neuralops-grpo"), **config_kwargs)
    trainer_kwargs = dict(kwargs)
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=dataset,
            **trainer_kwargs,
        )
    except TypeError as exc:
        if "reward_funcs" not in str(exc):
            raise
        trainer = GRPOTrainer(
            model=model,
            reward_func=reward_funcs,
            args=args,
            train_dataset=dataset,
            **trainer_kwargs,
        )
    if return_trainer:
        return trainer
    return trainer.train()
