"""
SVDAwareTrainer – custom HuggingFace Trainer with SVD compression callbacks.
"""

import gc
from typing import Any
from collections.abc import Callable

import torch
from transformers import Trainer, TrainingArguments


class SVDAwareTrainer(Trainer):
    """Trainer that supports pre/post training callbacks and attention map logging."""

    def __init__(
        self,
        visualizer: Any | None = None,
        pre_train_callback: Callable | None = None,
        post_train_callback: Callable | None = None,
        log_attention_maps: bool = False,
        attention_layers: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.visualizer = visualizer
        self.pre_train_callback = pre_train_callback
        self.post_train_callback = post_train_callback
        self.log_attention_maps = log_attention_maps
        self.attention_layers = attention_layers or [0, 5, 11]
        self._attention_hooks = []
        self._attention_maps = []

    def train(self, *args, **kwargs) -> Any:
        """Fit the model on the provided dataset."""
        if self.pre_train_callback:
            self.pre_train_callback(self.model, self.train_dataset)
        if self.log_attention_maps:
            self._register_attention_hooks()
        result = super().train(*args, **kwargs)
        if self.post_train_callback:
            self.post_train_callback(self.model, self.train_dataset)
        self._remove_attention_hooks()
        self._cleanup_memory()
        return result

    def evaluate(self, eval_dataset=None, **kwargs) -> Any:
        """Run evaluate and return results."""
        if self.log_attention_maps:
            self._register_attention_hooks()
        result = super().evaluate(eval_dataset, **kwargs)
        if self.log_attention_maps and self.visualizer:
            self._save_attention_maps()
        self._remove_attention_hooks()
        return result

    def _register_attention_hooks(self) -> None:
        """ register attention hooks."""
        self._attention_maps = []

        def hook_fn(module, input, output, layer_idx) -> None:
            """Hook fn."""
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attention_maps.append(
                    {"layer": layer_idx, "attention": output[1].detach().cpu()}
                )

        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                layer_idx = None
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layer" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
                if layer_idx in self.attention_layers:
                    hook = module.register_forward_hook(
                        lambda m, i, o, idx=layer_idx: hook_fn(m, i, o, idx)
                    )
                    self._attention_hooks.append(hook)

    def _remove_attention_hooks(self) -> None:
        """ remove attention hooks."""
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []

    def _save_attention_maps(self) -> None:
        """ save attention maps."""
        if self.visualizer and self._attention_maps:
            for attn_map in self._attention_maps:
                self.visualizer.plot_attention_heatmap(
                    attention_matrix=attn_map["attention"][0, 0].numpy(),
                    tokens=None,
                    layer_idx=attn_map["layer"],
                    head_idx=0,
                    rank=getattr(self.args, "rank", 64),
                    model_name=getattr(self.model.config, "_name_or_path", "model"),
                    task_name=getattr(self.args, "task_name", "unknown"),
                )

    def _determine_best_metric(self, metrics, trial=None):
        """Safely determine best metric without throwing KeyError if key mismatch occurs."""
        target = getattr(self.args, "metric_for_best_model", None)
        if target and target not in metrics:
            candidates = [
                target,
                f"eval_{target}",
                target.replace("eval_", ""),
                "eval_accuracy",
                "accuracy",
                "eval_loss",
                "loss",
            ]
            found = False
            for cand in candidates:
                if cand in metrics:
                    self.args.metric_for_best_model = cand
                    found = True
                    break
            if not found:
                return False
        return super()._determine_best_metric(metrics, trial=trial)

    def _cleanup_memory(self) -> None:
        """Free GPU/MPS/CPU memory after each training run."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass


def create_trainer(
    model,
    train_dataset,
    eval_dataset,
    output_dir: str = "./output",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    compute_metrics: Callable | None = None,
    **kwargs,
) -> SVDAwareTrainer:
    """Build and return trainer."""
    if eval_dataset is None:
        eval_strategy = "no"
        load_best_model_at_end = False

    trainer_kwarg_names = {
        "visualizer",
        "pre_train_callback",
        "post_train_callback",
        "log_attention_maps",
        "attention_layers",
        "callbacks",
        "optimizers",
    }
    trainer_kwargs = {k: v for k, v in kwargs.items() if k in trainer_kwarg_names}
    training_args_kwargs = {k: v for k, v in kwargs.items() if k not in trainer_kwarg_names}

    per_device_train_batch_size = training_args_kwargs.pop("per_device_train_batch_size", batch_size)
    per_device_eval_batch_size = training_args_kwargs.pop("per_device_eval_batch_size", batch_size)
    learning_rate = training_args_kwargs.pop("learning_rate", learning_rate)
    num_train_epochs = training_args_kwargs.pop("num_train_epochs", epochs)
    warmup_ratio = training_args_kwargs.pop("warmup_ratio", warmup_ratio)
    max_grad_norm = training_args_kwargs.pop("max_grad_norm", max_grad_norm)
    eval_strategy = training_args_kwargs.pop("eval_strategy", eval_strategy)
    save_strategy = training_args_kwargs.pop("save_strategy", save_strategy)
    save_total_limit = training_args_kwargs.pop("save_total_limit", save_total_limit)
    load_best_model_at_end = training_args_kwargs.pop("load_best_model_at_end", load_best_model_at_end)
    metric_for_best_model = training_args_kwargs.pop("metric_for_best_model", metric_for_best_model)
    if compute_metrics is None:
        metric_for_best_model = "eval_loss"
    greater_is_better = training_args_kwargs.pop("greater_is_better", None)
    if greater_is_better is None and eval_dataset is not None:
        greater_is_better = False if "loss" in metric_for_best_model else True

    # Pop GPU-specific optional args that may not exist in all transformers versions
    save_safetensors = training_args_kwargs.pop("save_safetensors", False)
    dataloader_pin_memory = training_args_kwargs.pop("dataloader_pin_memory", True)
    dataloader_num_workers = training_args_kwargs.pop("dataloader_num_workers", 0)
    _ = training_args_kwargs.pop("remove_unused_columns", None)  # handled below

    # Precision & Linear-Time Scalability Features
    fp16 = training_args_kwargs.pop("fp16", False)
    bf16 = training_args_kwargs.pop("bf16", False)
    gradient_checkpointing = training_args_kwargs.pop("gradient_checkpointing", False)
    optim = training_args_kwargs.pop("optim", "adamw_hf")
    lr_scheduler_type = training_args_kwargs.pop("lr_scheduler_type", "linear")

    # Build only the kwargs that TrainingArguments actually supports
    import inspect as _inspect
    _ta_params = set(_inspect.signature(TrainingArguments.__init__).parameters)

    optional_ta_kwargs = {}
    if "save_safetensors" in _ta_params:
        optional_ta_kwargs["save_safetensors"] = save_safetensors
    if "dataloader_pin_memory" in _ta_params:
        optional_ta_kwargs["dataloader_pin_memory"] = dataloader_pin_memory
    if "dataloader_num_workers" in _ta_params:
        optional_ta_kwargs["dataloader_num_workers"] = dataloader_num_workers

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model if eval_dataset is not None else None,
        greater_is_better=greater_is_better if eval_dataset is not None else None,
        logging_steps=training_args_kwargs.pop("logging_steps", 10),
        disable_tqdm=False,
        report_to=training_args_kwargs.pop("report_to", "none"),
        remove_unused_columns=False,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        **optional_ta_kwargs,
        **training_args_kwargs,
    )
    return SVDAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )
