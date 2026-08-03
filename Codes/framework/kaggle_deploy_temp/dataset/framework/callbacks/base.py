"""
Base callback classes following the HuggingFace TrainerCallback pattern.
"""

from typing import List, Optional

from transformers import TrainerCallback


class BaseCallback(TrainerCallback):
    """Extended base callback with extra event hooks."""

    def on_init_end(self, args, state, control, **kwargs) -> None:
        """On init end."""
        pass

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        """On epoch begin."""
        pass

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        """On epoch end."""
        pass

    def on_training_end(self, args, state, control, **kwargs) -> None:
        """On training end."""
        pass


class CallbackList:
    """Container that calls each callback in order."""

    def __init__(self, callbacks: Optional[List[TrainerCallback]] = None) -> None:
        self._callbacks = callbacks or []

    def add(self, callback: TrainerCallback) -> None:
        """Add."""
        self._callbacks.append(callback)

    def remove(self, callback: TrainerCallback) -> None:
        """Remove."""
        self._callbacks.remove(callback)

    def __iter__(self):
        return iter(self._callbacks)

    def __len__(self):
        return len(self._callbacks)

    def on_train_begin(self, *args, **kwargs) -> None:
        """On train begin."""
        for cb in self._callbacks:
            cb.on_train_begin(*args, **kwargs)

    def on_log(self, *args, **kwargs) -> None:
        """On log."""
        for cb in self._callbacks:
            cb.on_log(*args, **kwargs)

    def on_step_end(self, *args, **kwargs) -> None:
        """On step end."""
        for cb in self._callbacks:
            cb.on_step_end(*args, **kwargs)

    def on_evaluate(self, *args, **kwargs) -> None:
        """On evaluate."""
        for cb in self._callbacks:
            cb.on_evaluate(*args, **kwargs)

    def on_train_end(self, *args, **kwargs) -> None:
        """On train end."""
        for cb in self._callbacks:
            cb.on_train_end(*args, **kwargs)
