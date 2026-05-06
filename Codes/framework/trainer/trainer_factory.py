"""Trainer factory that returns the correct trainer class based on a key."""

from framework.core.registry import Registry
from framework.trainer.base_trainer import BaseTrainer

TRAINER_FACTORY_REGISTRY = Registry("trainer_factories")

def get_trainer(key: str = "hf"):
    """Return a trainer class (not instance)."""
    trainer_cls = TRAINER_FACTORY_REGISTRY.get(key)
    if trainer_cls is None:
        # Fallback to HuggingFace trainer
        from framework.trainer.hf_trainer import HuggingFaceTrainer
        return HuggingFaceTrainer
    return trainer_cls