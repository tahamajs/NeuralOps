# framework/utils/helpers.py
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Union
from .seed import set_all_seeds

def set_seed(seed: int, deterministic: bool = False):
    set_all_seeds(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return ' '.join(parts)

def format_number(num: int) -> str:
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if abs(num) >= 1_000:
        return f"{num/1_000:.0f}K"
    return str(num)

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(path: Union[str, Path]) -> Any:
    with open(path, 'r') as f:
        return json.load(f)

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
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