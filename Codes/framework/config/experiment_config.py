from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    path: str
    task_type: str  # "mc" or "single"
    num_choices: int = 0
    num_labels: int = 2


@dataclass
class ExperimentConfig:
    base_models: dict[str, str] = field(
        default_factory=lambda: {"bert": "bert-base-uncased"}
    )
    svd_ranks: list[int] = field(default_factory=lambda: [8, 16, 32])
    datasets: list[DatasetConfig] = field(default_factory=list)
    min_samples: int = 10
    max_samples: int = 200
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    val_split: float = 0.2
    seed: int = 42
    gradient_accumulation_steps: int = 1

    # Advanced Training & Linear-Time Scalability Options
    optimizer_type: str = "adamw"           # Options: adamw, sgd, adafactor
    lr_scheduler_type: str = "linear"       # Options: linear, cosine, cosine_with_restarts
    warmup_steps: int = 0
    weight_decay: float = 0.01
    mixed_precision: str = "no"             # Options: no, fp16, bf16
    gradient_checkpointing: bool = False    # Useful for extremely long sequences / linear time tracking
    max_grad_norm: float = 1.0              # Gradient clipping threshold

    save_checkpoint_interval: int = 10
    output_dir: str = "./results"
    plots_dir: str = "./plots/comprehensive"
    log_dir: str = "./logs"
    selected_models: list[str] = field(default_factory=list)
    selected_datasets: list[str] = field(default_factory=list)
    selected_ranks: list[int] = field(default_factory=list)
