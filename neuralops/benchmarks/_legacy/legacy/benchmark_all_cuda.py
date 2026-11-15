#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Deprecated entry point. Use `python3 -m neuralops.benchmarks.cli run --category cuda` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "Deprecated: This script has been quarantined in _legacy and is scheduled for removal. "
    "Please use canonical NeuralOps CLI v2 ('neuralops benchmark') instead.",
    DeprecationWarning,
    stacklevel=2,
)
import sys

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(
        "[NOTICE] This script was removed in October 2026. "
        "The unified benchmark runner is: python3 -m neuralops.benchmarks.cli run --category cuda"
    )
    sys.exit(0)
