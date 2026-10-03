# framework/experiment/manager.py
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from framework.callbacks import CALLBACK_REGISTRY

# Core registry and utilities
from framework.core.registry import TRAINER_REGISTRY
from framework.core.utils import get_device, set_seed
from framework.data.augmentation import AUGMENTATION_REGISTRY
from framework.distributed import (
    DistributedConfig,
    cleanup_distributed,
    wrap_model_with_distributed,
)

# Local experiment modules
from framework.experiment_utils.config import ExperimentConfig
from framework.experiment_utils.data import load_and_preprocess_data
from framework.experiment_utils.model import apply_attention_to_model

# High‑level services
from framework.hyperopt import HyperOptimizer
from framework.model_registry import ModelRegistry
from framework.tracking import ExperimentTracker
from framework.trainer.losses import LOSS_REGISTRY

# Extended registries
from framework.trainer.methods.registry import METHOD_REGISTRY
from framework.trainer.optimizers import OPTIMIZER_REGISTRY
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)

from Codes.framework.callbacks.django import DjangoCallback

logger = logging.getLogger(__name__)


@dataclass
class FullExperimentConfig(ExperimentConfig):
    """Extended configuration with all optional sub‑configs."""

    hpo_config: dict[str, Any] = field(default_factory=dict)
    distributed_config: dict[str, Any] = field(default_factory=dict)
    tracking_config: dict[str, Any] = field(default_factory=dict)
    deployment_config: dict[str, Any] = field(default_factory=dict)


class ExperimentManager:
    """
    High‑level experiment orchestration.

    Usage from Django:
        config = FullExperimentConfig(**request.data)
        manager = ExperimentManager(config)
        thread = threading.Thread(target=manager.run, args=(stop_event,))
        thread.start()
    """

    def __init__(self, config: FullExperimentConfig) -> None:
        self.config = config
        # Initialize tracker early – logs everything from the start
        self.tracker = ExperimentTracker(config.tracking_config)
        # Thread pool for parallel HPO trials (when applicable)
        self._executor = None
        self.stop_event = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, stop_event: Any | None = None) -> dict[str, Any]:
        """
        Execute the full experiment lifecycle.
        Returns a dictionary with final metrics and output paths.
        """
        self.stop_event = stop_event or self._dummy_event()
        try:
            # 1. Hyperparameter optimization (optional)
            if self.config.hpo_config.get("enabled"):
                self.config = self._run_hpo()

            # 2. Single training run (with optional distributed training)
            result = self._run_single()

            # 3. Track final results
            self.tracker.finalize(result)

            # 4. Register the best model (optional)
            if self.config.deployment_config.get("register_model"):
                ModelRegistry.register(
                    artifact_path=result["output_dir"],
                    metrics=result["metrics"],
                    tags=getattr(self.config, "tags", []),
                )

            return result
        except Exception as e:
            logger.exception("Experiment failed: %s", e)
            raise
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Hyperparameter optimization
    # ------------------------------------------------------------------
    def _run_hpo(self) -> FullExperimentConfig:
        """
        Run HPO using the configured backend.
        The objective function is a single training run with the given params.
        """
        optimizer = HyperOptimizer(self.config.hpo_config)

        def objective(trial_params: dict[str, Any]) -> float:
            """Single trial – merge HPO‑suggested params into config."""
            trial_cfg = FullExperimentConfig(**{**self.config.__dict__, **trial_params})
            trial_cfg.hpo_config["enabled"] = False  # prevent recursion
            manager = ExperimentManager(trial_cfg)
            result = manager._run_single()
            # Return the metric to maximize (e.g., eval_accuracy)
            return result["metrics"].get(
                self.config.hpo_config.get("metric", "eval_accuracy"), 0.0
            )

        best_params = optimizer.optimize(objective)
        # Merge best params into the original config
        for k, v in best_params.items():
            setattr(self.config, k, v)
        logger.info("HPO finished. Best params: %s", best_params)
        return self.config

    # ------------------------------------------------------------------
    # Single training run (the core)
    # ------------------------------------------------------------------
    def _run_single(self) -> dict[str, Any]:
        """Build and execute one training run from current config."""
        cfg = self.config
        set_seed(cfg.seed)
        device = get_device()

        # ---- Data loading ----
        tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)

        # Data augmentation (if any)
        for aug_name in cfg.augmentation:
            aug = AUGMENTATION_REGISTRY.get(aug_name)
            if aug:
                dataset = aug(dataset)

        # ---- Model preparation ----
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=num_labels
        )
        # Apply selected attention
        if cfg.attn_type and cfg.attn_type != "full":
            model = apply_attention_to_model(model, cfg.attn_type)
        model.to(device)

        # ---- Distributed training (optional) ----
        dist_config = (
            DistributedConfig(**cfg.distributed_config)
            if cfg.distributed_config
            else None
        )
        if dist_config:
            model = wrap_model_with_distributed(model, dist_config)

        # ---- Apply training method (LoRA, R‑Drop, etc.) ----
        if cfg.method and cfg.method != "baseline":
            method_cls = METHOD_REGISTRY.get(cfg.method)
            if method_cls:
                model = method_cls.apply(model, cfg)

        # ---- Build callbacks ----
        callbacks = self._build_callbacks()

        # ---- Build training arguments ----
        training_args = self._build_training_args(cfg, output_dir=cfg.output_dir)

        # ---- Create trainer via registry ----
        trainer_cls = TRAINER_REGISTRY.get(cfg.trainer_type)
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=callbacks,
            # Pass custom loss / optimizer if the trainer supports them
            loss_fn=self._get_loss_fn(cfg),
            optimizer_cls=self._get_optimizer_cls(cfg),
        )

        # ---- Attach callbacks and train ----
        for cb in callbacks:
            trainer.add_callback(cb)

        logger.info("Starting single run training...")
        trainer.train()
        eval_result = trainer.evaluate()
        logger.info("Training completed.")

        # ---- Save model ----
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name or "run")
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # ---- Post‑training compression (optional) ----
        if cfg.compressor:
            self._apply_post_training_compression(output_dir)

        return {
            "output_dir": output_dir,
            "metrics": eval_result,
            "history": trainer.state.log_history,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _build_callbacks(self) -> Any:
        """Instantiate callbacks based on config."""
        cbs = []
        callback_names = getattr(self.config, "selected_callbacks", [])
        for name in callback_names:
            cb_cls = CALLBACK_REGISTRY.get(name)
            if cb_cls:
                cbs.append(cb_cls(**getattr(self.config, f"{name}_kwargs", {})))
        # Always include DjangoCallback if progress/log/final are provided
        if hasattr(self, "progress_cb"):
            cbs.append(
                DjangoCallback(
                    progress_cb=self.progress_cb,
                    log_cb=self.log_cb,
                    final_cb=self.final_cb,
                    stop_event=self.stop_event,
                )
            )
        return cbs

    def _build_training_args(self, cfg, output_dir) -> Any:
        """Create HuggingFace TrainingArguments from config."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size * 2,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.scheduler_type,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=getattr(cfg, "metric_for_best", "eval_accuracy"),
            fp16=cfg.mixed_precision == "fp16",
            bf16=cfg.mixed_precision == "bf16",
            remove_unused_columns=False,
            report_to=cfg.tracking_config.get("report_to", "none"),
            max_grad_norm=cfg.max_grad_norm,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

    def _get_loss_fn(self, cfg) -> Any:
        """get loss fn."""
        loss_name = getattr(cfg, "loss_type", "cross_entropy")
        loss_cls = LOSS_REGISTRY.get(loss_name)
        if loss_cls:
            return loss_cls()
        return None

    def _get_optimizer_cls(self, cfg) -> Any:
        """get optimizer cls."""
        opt_name = getattr(cfg, "optimizer_type", "adamw")
        return OPTIMIZER_REGISTRY.get(opt_name)

    def _apply_post_training_compression(self, output_dir) -> None:
        """Run compression after training."""
        from framework.experiment_utils.main import run_compression

        run_compression(self.config, output_dir)

    def _cleanup(self) -> None:
        """Release resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
        if self.config.distributed_config:
            cleanup_distributed()

    @staticmethod
    def _dummy_event() -> Any:
        """dummy event."""
        import threading

        return threading.Event()
