"""
YAML‑based configuration loader with precedence:
  defaults → config file(s) → environment variables → CLI arguments

Inspired by Hydra (Facebook AI), torchtune, and click-configfile.
"""

import os
import copy
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # ── Model ──
    model_name: str = "distilbert-base-uncased"
    attention_type: str = "full"
    compression_rank: int = 64
    use_unsloth: bool = False
    cache_dir: Optional[str] = None

    # ── Data ──
    dataset_name: str = "mrpc"
    dataset_path: Optional[str] = None
    max_seq_length: int = 128
    split_ratio: float = 0.1
    glue_base_path: str = "./data/glue"
    num_labels: Optional[int] = None

    # ── Training ──
    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42

    # ── Device ──
    device: str = "auto"
    fp16: bool = True
    bf16: bool = False

    # ── Logging ──
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2

    # ── Compression (for `compress` command) ──
    compression_method: str = "low_rank"
    target_sparsity: float = 0.5
    target_rank: int = 64

    # ── Synthetic Data ──
    synthetic_domain: str = "movies"
    synthetic_samples: int = 5000
    synthetic_output: str = "./data/synthetic.csv"

    # meta
    config_name: str = "default"
    config_description: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        # filter out keys not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: Path):
        """Save config to a YAML file."""
        d = asdict(self)
        with open(path, "w") as f:
            yaml.safe_dump(d, f, default_flow_style=False, sort_keys=False)

    def merge(self, other: Dict[str, Any]) -> "TrainingConfig":
        """Merge a dictionary on top of current config (other overrides self)."""
        current = asdict(self)
        # only override non‑None values
        for k, v in other.items():
            if v is not None and k in current:
                current[k] = v
        return TrainingConfig(**current)

    def merge_file(self, path: Path) -> "TrainingConfig":
        """Merge a YAML config file on top of current config."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return self.merge(data)


def load_config(
    config_paths: Optional[List[Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "ETK_"
) -> TrainingConfig:
    """
    Load and merge configuration from multiple sources.

    Precedence (highest wins):
      1. CLI arguments (cli_overrides)
      2. Environment variables (ETK_*)
      3. Config files (config_paths), merged left to right
      4. defaults.yaml
    """
    # Start with built‑in defaults
    cfg = TrainingConfig()

    # Layer 1: default.yaml (always loaded if it exists)
    default_path = CONFIGS_DIR / "default.yaml"
    if default_path.exists():
        cfg = cfg.merge_file(default_path)

    # Layer 2: additional config files
    if config_paths:
        for p in config_paths:
            if Path(p).exists():
                cfg = cfg.merge_file(Path(p))

    # Layer 3: environment variables
    cfg = _apply_env_overrides(cfg, env_prefix)

    # Layer 4: CLI overrides
    if cli_overrides:
        cfg = cfg.merge(cli_overrides)

    # Auto‑detect device
    if cfg.device == "auto":
        import torch
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def _apply_env_overrides(cfg: TrainingConfig, prefix: str) -> TrainingConfig:
    """Apply environment variables like ETK_MODEL_NAME, ETK_BATCH_SIZE, etc."""
    overrides = {}
    for field in TrainingConfig.__dataclass_fields__:
        env_var = f"{prefix}{field.upper()}"
        val = os.environ.get(env_var)
        if val is not None:
            # try to cast to the correct type
            ftype = TrainingConfig.__dataclass_fields__[field].type
            if ftype == bool:
                overrides[field] = val.lower() in ("1", "true", "yes")
            elif ftype == int:
                overrides[field] = int(val)
            elif ftype == float:
                overrides[field] = float(val)
            elif ftype == Optional[str]:
                overrides[field] = val if val else None
            else:
                overrides[field] = val
    return cfg.merge(overrides) if overrides else cfg


def list_available_configs() -> Dict[str, List[str]]:
    """Return all available config profiles grouped by category."""
    result = {}
    for category in ["tasks", "models", "attentions", "presets"]:
        cat_dir = CONFIGS_DIR / category
        if cat_dir.exists():
            result[category] = [p.stem for p in sorted(cat_dir.glob("*.yaml"))]
    return result


def resolve_config_sources(
    task: Optional[str] = None,
    model: Optional[str] = None,
    attention: Optional[str] = None,
    preset: Optional[str] = None,
    custom_configs: Optional[List[Path]] = None,
) -> List[Path]:
    """
    Build a list of config file paths from named presets / categories.
    Order: preset → task → model → attention → custom
    """
    paths = []

    if preset:
        p = CONFIGS_DIR / "presets" / f"{preset}.yaml"
        if p.exists():
            paths.append(p)

    if task:
        p = CONFIGS_DIR / "tasks" / f"{task}.yaml"
        if p.exists():
            paths.append(p)

    if model:
        p = CONFIGS_DIR / "models" / f"{model}.yaml"
        if p.exists():
            paths.append(p)

    if attention:
        p = CONFIGS_DIR / "attentions" / f"{attention}.yaml"
        if p.exists():
            paths.append(p)

    if custom_configs:
        paths.extend(custom_configs)

    return paths