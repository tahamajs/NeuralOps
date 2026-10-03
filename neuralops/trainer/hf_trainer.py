from typing import Any

from .base_trainer import BaseTrainer
from .registry import TrainerRegistry
from .trainer import create_trainer


@TrainerRegistry.register("hf")
class HuggingFaceTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        config: dict[str, Any],
        results_manager,
        compute_metrics=None,
        **kwargs,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.results_manager = results_manager
        self.compute_metrics = compute_metrics
        self.kwargs = kwargs
        self._trainer = None
        self._callbacks: list[Any] = []

    def add_callback(self, callback: Any) -> None:
        """Add a callback to the trainer."""
        if callback is not None:
            self._callbacks.append(callback)
            if self._trainer is not None:
                self._trainer.add_callback(callback)

    @property
    def args(self) -> Any:
        """Return training arguments."""
        if self._trainer is None:
            self._build()
        return self._trainer.args

    @property
    def state(self) -> Any:
        """Return trainer state."""
        if self._trainer is not None:
            return self._trainer.state
        return None

    def _build(self) -> Any:
        """Build underlying HuggingFace Trainer."""
        output_dir = self.results_manager.get_checkpoint_dir()
        training_args_dict = {
            "output_dir": output_dir,
            "learning_rate": self.config.get("learning_rate", 2e-5),
            "per_device_train_batch_size": self.config.get("batch_size", 16),
            "per_device_eval_batch_size": self.config.get("batch_size", 16),
            "num_train_epochs": self.config.get("epochs", 3),
            "warmup_ratio": self.config.get("warmup_ratio", 0.1),
            "max_grad_norm": self.config.get("max_grad_norm", 1.0),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "eval_strategy": self.config.get("eval_strategy", "epoch"),
            "save_strategy": self.config.get("save_strategy", "epoch"),
            "save_total_limit": self.config.get("save_total_limit", 2),
            "load_best_model_at_end": (
                self.config.get("load_best_model_at_end", True)
                if self.eval_dataset is not None
                else False
            ),
            "metric_for_best_model": self.config.get(
                "metric_for_best_model", "eval_loss"
            ),
            "fp16": self.config.get("fp16", False),
            "logging_dir": str(self.results_manager.log_dir),
            "logging_steps": self.config.get("logging_steps", 10),
            "report_to": "none",
            # Disable safetensors to avoid RuntimeError from shared-memory tensors in custom attention modules
            "save_safetensors": False,
            # GPU throughput
            "dataloader_pin_memory": True,
            "dataloader_num_workers": self.config.get("num_workers", 0),
            "remove_unused_columns": False,
        }
        self._trainer = create_trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            **training_args_dict,
            **self.kwargs,
        )
        self._trainer = self.results_manager.wrap_trainer(self._trainer)
        for cb in self._callbacks:
            self._trainer.add_callback(cb)

        original_train = self._trainer.train

        def train_and_save(*args, **train_kwargs) -> Any:
            """Train, persist artifacts, and generate structured summary reports."""
            result = original_train(*args, **train_kwargs)
            self.results_manager.save_final_model(self.model)
            self.results_manager.save_plots()
            last_metrics = (
                getattr(self._trainer.state, "log_history", [])[-1]
                if getattr(self._trainer.state, "log_history", [])
                else {}
            )
            self.results_manager.generate_summary_report(final_metrics=last_metrics)
            return result

        self._trainer.train = train_and_save

    def train(self) -> Any:
        """Fit the model on the provided dataset."""
        if self._trainer is None:
            self._build()
        return self._trainer.train()

    def evaluate(self) -> Any:
        """Run evaluate and return results."""
        if self._trainer is None:
            self._build()
        return self._trainer.evaluate()
