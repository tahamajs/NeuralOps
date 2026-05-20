import logging
from typing import Dict, Type
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class TrainerRegistry:
    """Registry for trainer classes."""
    _registry: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: Type[BaseTrainer]) -> Type[BaseTrainer]:
            if not issubclass(klass, BaseTrainer):
                raise TypeError(f"{klass.__name__} must inherit from BaseTrainer")
            cls._registry[name] = klass
            logger.info(f"Registered trainer: {name} -> {klass.__name__}")
            return klass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseTrainer]:
        if name not in cls._registry:
            raise ValueError(f"Unknown trainer '{name}'. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list:
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTrainer:
        return cls.get(name)(**kwargs)