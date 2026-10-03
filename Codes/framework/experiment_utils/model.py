from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from ..utils.layer_norm_fix import fix_layernorm_naming


class _MockObjectsManager:
    """Mock manager for non-Django or standalone usage."""

    def filter(self, *args, **kwargs) -> Any:
        return self

    def order_by(self, *args, **kwargs) -> list["TrainingRun"]:
        return []


@dataclass
class TrainingRun:
    """Dataclass representing a training run for comparison and logging."""

    id: Any = None
    name: str = ""
    status: str = "completed"
    created_at: datetime = field(default_factory=datetime.now)
    config: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    training_history: dict[str, Any] = field(default_factory=dict)
    actual_duration: float = 0.0

    objects: Any = _MockObjectsManager()


def load_model(cfg: Any, num_labels: int, device: str = "auto") -> Any:
    """Load model using config and num_labels."""
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, config=config
    )
    fix_layernorm_naming(model)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model
