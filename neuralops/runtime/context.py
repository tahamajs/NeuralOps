#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Thread-safe ContextVar-based Runtime Context and Ergonomic Scoping.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional
import torch

from neuralops.runtime.config import RuntimeConfig
from neuralops.runtime.events import EventSink

_DEFAULT_CONFIG = RuntimeConfig()
_DEFAULT_EVENT_SINK = EventSink()

_CURRENT_CONFIG: contextvars.ContextVar[Optional[RuntimeConfig]] = contextvars.ContextVar(
    "current_runtime_config", default=None
)
_CURRENT_EVENT_SINK: contextvars.ContextVar[Optional[EventSink]] = contextvars.ContextVar(
    "current_event_sink", default=None
)


def get_runtime_config() -> RuntimeConfig:
    """Returns the currently active RuntimeConfig."""
    cfg = _CURRENT_CONFIG.get()
    return cfg if cfg is not None else _DEFAULT_CONFIG


def set_runtime_config(config: RuntimeConfig) -> None:
    """Sets the active RuntimeConfig globally for the current context."""
    _CURRENT_CONFIG.set(config)


def get_event_sink() -> EventSink:
    """Returns the currently active EventSink."""
    sink = _CURRENT_EVENT_SINK.get()
    return sink if sink is not None else _DEFAULT_EVENT_SINK


def set_event_sink(sink: EventSink) -> None:
    """Sets the active EventSink for the current context."""
    _CURRENT_EVENT_SINK.set(sink)


def init(
    device: str = "auto",
    dtype: str = "auto",
    seed: int = 42,
    deterministic: bool = False,
    log_level: str = "INFO",
    max_memory_fraction: float = 0.90,
) -> RuntimeConfig:
    """Initializes the NeuralOps global runtime environment with user preferences.

    Parameters:
        device: "auto", "cuda", "cpu", or "mps".
        dtype: "auto", "float16", "bfloat16", or "float32".
        seed: Random seed to initialize torch and random state.
        deterministic: Whether to set deterministic flags on PyTorch accelerators.
        log_level: Logging severity level.
        max_memory_fraction: VRAM fraction threshold.

    Returns:
        The configured RuntimeConfig instance.
    """
    config = RuntimeConfig(
        device=device,
        dtype=dtype,
        seed=seed,
        deterministic=deterministic,
        log_level=log_level,  # type: ignore[arg-type]
        max_memory_fraction=max_memory_fraction,
    )
    set_runtime_config(config)

    # Apply seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return config


class use_runtime:
    """Ergonomic context manager for temporarily overriding runtime settings.

    Example:
        >>> with neuralops.use_runtime(dtype="float32", device="cpu"):
        ...     # Runs in float32 on CPU within this block
        ...     output = model(inputs)
    """

    def __init__(self, **overrides: Any) -> None:
        current = get_runtime_config()
        updated_dict = dict(current.__dict__)
        updated_dict.update(overrides)
        self.new_config = RuntimeConfig(**updated_dict)
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> RuntimeConfig:
        self.token = _CURRENT_CONFIG.set(self.new_config)
        return self.new_config

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.token is not None:
            _CURRENT_CONFIG.reset(self.token)
