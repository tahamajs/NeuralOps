from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw"
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.001
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 2
    save_only_model: bool = True
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    bf16: bool = False
    report_to: str = "none"
