#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deprecated core configuration module.

DEPRECATION NOTICE:
This module is deprecated in favor of `neuralops.config` and will be removed in version 2.2.0.
Please update your imports:
    from neuralops.config import Config, DummyConfig, load_config
"""

from __future__ import annotations

import warnings

from neuralops.config.config import Config, DummyConfig
from neuralops.config.loader import load_config

warnings.warn(
    "neuralops.core.config is deprecated. Use neuralops.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Config", "DummyConfig", "load_config"]
