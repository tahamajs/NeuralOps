#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Lazy public exports for experiment logging."""

from importlib import import_module

_EXPORTS = {
    "LoggingConfig": ("config", "LoggingConfig"),
    "setup_logging": ("logger", "setup_logging"),
    "shutdown_logging": ("logger", "shutdown_logging"),
    "get_logger": ("logger", "get_logger"),
    "get_rank": ("logger", "get_rank"),
    "MetricsTracker": ("tracker", "MetricsTracker"),
    "LoggingCallback": ("callbacks", "LoggingCallback"),
    "auto_setup": ("integration", "auto_setup"),
    "ResultsManager": ("neuralops.trainer.results_manager", "ResultsManager"),
}
_MODULES = {"callbacks", "config", "integration", "logger", "tracker"}

__all__ = [*_EXPORTS, *_MODULES]


def __getattr__(name: str):
    """Load optional tracking dependencies only when their APIs are requested."""
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        target = module_name if module_name.startswith("neuralops.") else f"{__name__}.{module_name}"
        value = getattr(import_module(target), attribute)
    elif name in _MODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
