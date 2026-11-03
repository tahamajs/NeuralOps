#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Runtime Context, Configuration, and Event Subsystem.
"""

from __future__ import annotations

from neuralops.runtime.config import RuntimeConfig
from neuralops.runtime.context import (
    get_event_sink,
    get_runtime_config,
    init,
    set_event_sink,
    set_runtime_config,
    use_runtime,
)
from neuralops.runtime.events import EventListener, EventSink, RunEvent

__all__ = [
    "RuntimeConfig",
    "RunEvent",
    "EventListener",
    "EventSink",
    "get_runtime_config",
    "set_runtime_config",
    "get_event_sink",
    "set_event_sink",
    "init",
    "use_runtime",
]
