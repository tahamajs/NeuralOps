from typing import Any

from neuralops.core.registry import Registry

from .hf_trainer import HuggingFaceTrainer

TRAINER_FACTORY_REGISTRY = Registry("trainer_factories")


def get_trainer(key: str = "hf") -> Any:
    """Return a trainer class (not instance)."""
    trainer_cls = TRAINER_FACTORY_REGISTRY.get(key)
    if trainer_cls is None:
        return HuggingFaceTrainer  # default fallback
    return trainer_cls
