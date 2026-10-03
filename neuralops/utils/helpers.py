# framework/utils/helpers.py
import json
import os
from pathlib import Path
from typing import Any

import torch

from .seed import set_all_seeds


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set the seed."""
    set_all_seeds(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def format_number(num: int) -> str:
    """Format number."""
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if abs(num) >= 1_000:
        return f"{num/1_000:.0f}K"
    return str(num)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str | Path) -> Path:
    """Ensure dir."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Persist json to disk."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load json from disk or config."""
    with open(path) as f:
        return json.load(f)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten dict."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
