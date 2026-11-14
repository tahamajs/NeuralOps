#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for logging warnings on event listener failures."""

from __future__ import annotations

import logging

from neuralops.runtime.events import EventSink


def test_listener_failure_logs_warning_with_exc_info(caplog):
    """Verify EventSink.emit logs a warning when a listener fails."""
    sink = EventSink()

    def bad_listener(event):
        raise ValueError("Listener exploded")

    sink.subscribe("test.topic", bad_listener)

    with caplog.at_level(logging.WARNING, logger="NeuralOps.Runtime.Events"):
        event = sink.emit("test.topic", key="val")

    assert event.topic == "test.topic"
    assert "Listener failed on event test.topic: ValueError: Listener exploded" in caplog.text


def test_global_listener_failure_logs_warning_with_exc_info(caplog):
    """Verify EventSink.emit logs a warning when a global listener fails."""
    sink = EventSink()

    def bad_global_listener(event):
        raise RuntimeError("Global listener exploded")

    sink.subscribe_all(bad_global_listener)

    with caplog.at_level(logging.WARNING, logger="NeuralOps.Runtime.Events"):
        event = sink.emit("any.topic")

    assert event.topic == "any.topic"
    assert "Listener failed on event any.topic: RuntimeError: Global listener exploded" in caplog.text
