# framework/experiment/train.py
import os
import logging
from transformers import TrainingArguments, DataCollatorWithPadding
from framework.trainer.trainer import CompressionTrainer, get_compute_metrics
from framework.trainer.methods.rdrop_r3f import MethodTrainer
from experiment_logging import MetricsTracker, LoggingCallback

logger = logging.getLogger(__name__)

def setup_trainer(model, tokenizer, dataset, config, tracker, logger):
    """Create and configure trainer."""
    metric_map = {
        "cola": "eval_matthews_correlation",
        "stsb": "eval_pearson",
        "mrpc": "eval_f1",
        "qqp": "eval_f1"
    }
    metric_for_best = metric_map.get(config.task, "eval_accuracy")
    is_regression = (config.task == "stsb")

    output_dir = os.path.join(config.output_dir, config.exp_name or f"{config.attn_type}_{config.method}_{config.task}_{config.seed}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Choose trainer class
    if config.method in ("rdrop", "r3f"):
        trainer_cls = MethodTrainer
        extra = {
            "method": config.method,
            "rdrop_alpha": config.rdrop_alpha,
            "r3f_epsilon": config.r3f_epsilon,
            "is_regression": is_regression,
        }
    else:
        trainer_cls = CompressionTrainer
        extra = {"ortho_weight": 0.0}

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(config.task),
        **extra
    )

    trainer.add_callback(LoggingCallback(tracker, logger, plots_dir=output_dir))

    return trainer