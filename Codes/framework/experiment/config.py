# framework/experiment/config.py
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ExperimentConfig:
    # Model & data
    model_name: str = "distilbert-base-uncased"
    attn_type: str = "full"
    method: str = "baseline"
    compressor: Optional[str] = None
    task: str = "sst2"
    max_len: int = 128
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    grad_accum: int = 1
    seed: int = 42
    output_dir: str = "./experiments"
    log_dir: str = "./logs"
    exp_name: Optional[str] = None

    # Method‑specific
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    rdrop_alpha: float = 5.0
    r3f_epsilon: float = 1e-5
    mixout_p: float = 0.9
    llrd_decay: float = 0.95
    dislora_beta: float = 0.5
    dislora_capacity_factor: float = 1.0
    fisher_num_batches: int = 20
    fisher_sample_size: int = 200
    gelora_margin: float = 1.0
    gelora_min_rank: int = 2
    gelora_max_rank: int = 32
    fisher_scale_max: float = 2.0
    use_lion: bool = False
    gradient_checkpointing: bool = False
    early_stopping_patience: Optional[int] = None
    max_grad_norm: float = 1.0
    num_seeds: int = 1

    # Attention‑specific
    num_landmarks: int = 16
    feature_dim: int = 128
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    use_gate: bool = True
    progressive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_config_from_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Efficient Attention GLUE Runner")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--attn_type", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--compressor", type=str, default=None)
    parser.add_argument("--task", "-t", type=str, default="sst2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default=None)
    # Method params
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--rdrop_alpha", type=float, default=5.0)
    parser.add_argument("--r3f_epsilon", type=float, default=1e-5)
    parser.add_argument("--mixout_p", type=float, default=0.9)
    parser.add_argument("--llrd_decay", type=float, default=0.95)
    parser.add_argument("--dislora_beta", type=float, default=0.5)
    parser.add_argument("--dislora_capacity_factor", type=float, default=1.0)
    parser.add_argument("--fisher_num_batches", type=int, default=20)
    parser.add_argument("--fisher_sample_size", type=int, default=200)
    parser.add_argument("--gelora_margin", type=float, default=1.0)
    parser.add_argument("--gelora_min_rank", type=int, default=2)
    parser.add_argument("--gelora_max_rank", type=int, default=32)
    parser.add_argument("--fisher_scale_max", type=float, default=2.0)
    parser.add_argument("--use_lion", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_seeds", type=int, default=1)
    # Attention params
    parser.add_argument("--num_landmarks", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_gate", action="store_true")

    args = parser.parse_args()
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        # Override with command line arguments (CLI takes precedence)
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                yaml_cfg[key] = value
        cfg_dict = yaml_cfg
    else:
        cfg_dict = vars(args)

    # Convert to ExperimentConfig dataclass
    cfg_dict['use_gate'] = not cfg_dict.get('no_gate', False)  # fix flag
    return ExperimentConfig(**cfg_dict)