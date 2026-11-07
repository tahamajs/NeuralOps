#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Event Bus and Lifecycle Notifications for Decoupled Telemetry.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("NeuralOps.Runtime.Events")


@dataclass(frozen=True)
class RunEvent:
    """Represents an event emitted during model surgery, compression, or execution.

    Attributes:
        topic: Categorical topic (e.g., 'stage.start', 'stage.end', 'metric.loss', 'error').
        payload: Structured dictionary containing event attributes and telemetry.
        timestamp: ISO 8601 UTC timestamp of creation.
    """

    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


EventListener = Callable[[RunEvent], None]


class EventSink:
    """Thread-safe publish-subscribe event sink for dispatching runtime telemetry."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventListener]] = {}
        self._global_listeners: List[EventListener] = []

    def subscribe(self, topic: str, listener: EventListener) -> None:
        """Subscribes a listener callback to a specific topic."""
        if topic not in self._listeners:
            self._listeners[topic] = []
        self._listeners[topic].append(listener)

    def subscribe_all(self, listener: EventListener) -> None:
        """Subscribes a listener callback to all topics."""
        self._global_listeners.append(listener)

    def emit(self, topic: str, **payload: Any) -> RunEvent:
        """Emits an event to all subscribed listeners."""
        event = RunEvent(topic=topic, payload=payload)
        for listener in self._global_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.debug(f"Global listener failed on event {topic}: {e}")

        for listener in self._listeners.get(topic, []):
            try:
                listener(event)
            except Exception as e:
                logger.debug(f"Listener failed on event {topic}: {e}")

        return event
