import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

BASE_DIR = Path.home() / "Documents" / "huggingface_secure_cache"
HF_CACHE_DIR = BASE_DIR / "models"
DATASET_CACHE_DIR = BASE_DIR / "datasets"
EXPERIMENTS_DIR = BASE_DIR / "experiments_logs"

for p in [BASE_DIR, HF_CACHE_DIR, DATASET_CACHE_DIR, EXPERIMENTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
os.environ["HF_HOME"] = str(BASE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(DATASET_CACHE_DIR)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision('high')
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@dataclass
class HardwareConfig:
    device: str = "auto"
    force_cpu_for_svd: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = False

@dataclass
class DataConfig:
    local_glue_path: str = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    max_seq_length: int = 128
    batch_size: int = 8
    eval_batch_size: int = 16
    num_workers: int = 4
    limit_samples_per_task: int = 1000

@dataclass
class ModelConfig:
    name: str = "bert-base-uncased"
    attention_type: str = "svd"
    rank: int = 64
    num_labels: int = 2
    cache_dir: str = str(HF_CACHE_DIR)

@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    early_stopping_patience: Optional[int] = None

@dataclass
class SVDConfig:
    energy_threshold: float = 0.95
    min_rank: int = 4
    max_rank: int = 128
    compression_method: str = "standard_svd"
    adaptive_ranks: bool = True

@dataclass
class ExperimentConfig:
    tasks: List[str] = field(default_factory=lambda: ["cola", "sst2", "mrpc"])
    seed: int = 42
    num_seeds: int = 1
    output_dir: str = "./results"
    log_dir: str = "./logs"
    visualize: bool = True
    generate_report: bool = True

@dataclass
class MainConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def get_device(self) -> torch.device:
        dev = self.hardware.device
        if dev == "auto":
            return get_device()
        return torch.device(dev)

config = MainConfig()
DEVICE = config.get_device()