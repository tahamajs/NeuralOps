"""
Factory to select the appropriate trainer or training method based on config.
Supports standard HuggingFace Trainer, custom loops, Unsloth, and RL (via TRL).
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import Trainer, TrainingArguments
from .trainer import SVDAwareTrainer, create_trainer
from .utils import train_model as custom_train_loop
from .results_manager import ResultsManager

def get_trainer(
    model,
    train_dataset,
    eval_dataset,
    results_manager: Optional[ResultsManager] = None,
    config: Optional[Dict[str, Any]] = None,
    use_unsloth: bool = False,
    use_rl: bool = False,
    **kwargs
) -> Tuple[Trainer, Optional[ResultsManager]]:
    """
    Return a trainer (or run training) based on configuration.

    Args:
        model: the model to train.
        train_dataset: training dataset.
        eval_dataset: evaluation dataset.
        results_manager: optional ResultsManager to handle logging and saving.
        config: dictionary with training settings.
        use_unsloth: whether to use Unsloth's optimized training.
        use_rl: whether to use RL training (requires TRL).
        **kwargs: additional arguments passed to the trainer.

    Returns:
        Tuple of (trainer, results_manager) – after training the results_manager will contain
        the run directory and saved artefacts.
    """
    if results_manager is None:
        results_manager = ResultsManager(
            experiment_name=config.get("experiment_name", "experiment"),
            config=config
        )

    # Common training args
    output_dir = results_manager.get_checkpoint_dir()
    training_args_dict = {
        "output_dir": output_dir,
        "learning_rate": config.get("learning_rate", 2e-5),
        "per_device_train_batch_size": config.get("batch_size", 16),
        "per_device_eval_batch_size": config.get("batch_size", 16),
        "num_train_epochs": config.get("epochs", 3),
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "weight_decay": config.get("weight_decay", 0.01),
        "eval_strategy": config.get("eval_strategy", "epoch"),
        "save_strategy": config.get("save_strategy", "epoch"),
        "save_total_limit": config.get("save_total_limit", 2),
        "load_best_model_at_end": config.get("load_best_model_at_end", True),
        "metric_for_best_model": config.get("metric_for_best_model", "eval_loss"),
        "fp16": config.get("fp16", torch.cuda.is_available()),
        "logging_dir": str(results_manager.log_dir),
        "logging_steps": config.get("logging_steps", 50),
        "report_to": "none",
    }

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            # Unsloth models use their own Trainer
            trainer = Trainer(
                model=model,
                args=TrainingArguments(**training_args_dict),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                **kwargs
            )
            # Wrap with results_manager callbacks
            return results_manager.wrap_trainer(trainer), results_manager
        except ImportError:
            raise ImportError("Unsloth not installed. Run: pip install unsloth")

    if use_rl:
        try:
            from trl import PPOTrainer, PPOConfig
            ppo_config = PPOConfig(
                batch_size=training_args_dict["per_device_train_batch_size"],
                learning_rate=training_args_dict["learning_rate"],
                log_with=None,
            )
            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=model,
                tokenizer=kwargs.get("tokenizer"),
                dataset=train_dataset,
            )
            # RL training loop (simplified)
            for epoch, batch in enumerate(train_dataset):
                stats = ppo_trainer.step(batch["input_ids"], batch["attention_mask"], batch["labels"])
                results_manager.log_metrics(stats, step=epoch)
            results_manager.save_final_model(model)
            results_manager.save_plots()
            return ppo_trainer, results_manager
        except ImportError:
            raise ImportError("TRL not installed. Run: pip install trl")

    # Standard HuggingFace Trainer (or SVDAwareTrainer)
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **training_args_dict,
        **kwargs
    )
    # Wrap with results_manager if provided
    if results_manager:
        trainer = results_manager.wrap_trainer(trainer)

    # Attach a post-training hook to save final model and plots
    original_train = trainer.train
    def train_and_save(*args, **train_kwargs):
        result = original_train(*args, **train_kwargs)
        if results_manager:
            results_manager.save_final_model(model)
            results_manager.save_plots()
        return result
    trainer.train = train_and_save

    return trainer, results_manager