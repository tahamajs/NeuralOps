from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DatasetConfig:
    name: str
    path: str
    task_type: str           # "mc" or "single"
    num_choices: int = 0
    num_labels: int = 2

@dataclass
class ExperimentConfig:
    base_models: Dict[str, str] = field(default_factory=lambda: {"bert": "bert-base-uncased"})
    svd_ranks: List[int] = field(default_factory=lambda: [8, 16, 32])
    datasets: List[DatasetConfig] = field(default_factory=list)
    min_samples: int = 10
    max_samples: int = 200
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    val_split: float = 0.2
    seed: int = 42
    gradient_accumulation_steps: int = 1
    save_checkpoint_interval: int = 10
    output_dir: str = "./results"
    plots_dir: str = "./plots/comprehensive"
    log_dir: str = "./logs"
    selected_models: List[str] = field(default_factory=list)
    selected_datasets: List[str] = field(default_factory=list)
    selected_ranks: List[int] = field(default_factory=list)