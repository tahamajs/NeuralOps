from typing import Any
import logging

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TrainerRegistry:
    """Registry for trainer classes."""

    _registry: dict[str, type[BaseTrainer]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Register."""
        def decorator(klass: type[BaseTrainer]) -> type[BaseTrainer]:
            """Decorator."""
            if not issubclass(klass, BaseTrainer):
                raise TypeError(f"{klass.__name__} must inherit from BaseTrainer")
            cls._registry[name] = klass
            logger.info(f"Registered trainer: {name} -> {klass.__name__}")
            return klass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseTrainer]:
        """Get."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown trainer '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list:
        """List."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTrainer:
        """Create."""
        return cls.get(name)(**kwargs)
