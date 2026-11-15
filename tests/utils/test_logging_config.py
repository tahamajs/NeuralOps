#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for centralized logging_config and deduplication filter."""

from __future__ import annotations

import logging
import io

from neuralops.utils.logging_config import DedupFilter, configure_logging, warn_once


def test_dedup_filter():
    """Verify that DedupFilter suppresses duplicate records from the same call site."""
    dedup = DedupFilter()
    logger = logging.getLogger("test_dedup_logger")
    record1 = logger.makeRecord("test_dedup_logger", logging.INFO, "module_a.py", 42, "msg1", (), None)
    record2 = logger.makeRecord("test_dedup_logger", logging.INFO, "module_a.py", 42, "msg2", (), None)
    record3 = logger.makeRecord("test_dedup_logger", logging.INFO, "module_a.py", 43, "msg3", (), None)

    # First record from line 42 should pass
    assert dedup.filter(record1) is True
    # Second record from same line 42 and level should be suppressed
    assert dedup.filter(record2) is False
    # Record from line 43 should pass
    assert dedup.filter(record3) is True


def test_warn_once():
    """Verify warn_once only logs once for a given key."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    test_logger = logging.getLogger("test_warn_once_logger")
    test_logger.setLevel(logging.WARNING)
    test_logger.addHandler(handler)

    unique_key = "unique_deprecation_key_123"
    warn_once(unique_key, "This is a deprecation warning", logger=test_logger)
    warn_once(unique_key, "This is a deprecation warning", logger=test_logger)

    output = stream.getvalue().strip()
    # Should only appear once
    assert output.count("This is a deprecation warning") == 1


def test_configure_logging_dedup_and_idempotence():
    """Verify configure_logging does not duplicate handlers and dedup filter works."""
    root = logging.getLogger()
    # Save original state
    orig_handlers = list(root.handlers)
    orig_level = root.level

    try:
        # Reset handlers for isolated test
        root.handlers = []

        # First call
        configure_logging(level="DEBUG", dedup=True)
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG
        assert any(isinstance(f, DedupFilter) for f in root.handlers[0].filters)

        # Second call with different level
        configure_logging(level="WARNING", dedup=True)
        # Should not add another handler (idempotent / no duplicates)
        assert len(root.handlers) == 1
        assert root.level == logging.WARNING

        # Dedup filter should still be present exactly once
        dedup_filters = [f for f in root.handlers[0].filters if isinstance(f, DedupFilter)]
        assert len(dedup_filters) == 1
    finally:
        root.handlers = orig_handlers
        root.setLevel(orig_level)
