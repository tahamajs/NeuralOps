#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility shim forwarding neuralops.benchmarks.logging to run_logging."""

from __future__ import annotations

import importlib
import sys

if __name__ == "logging":
    # If accidentally placed on sys.path and imported as stdlib 'logging', forward to real stdlib
    del sys.modules["logging"]
    _stdlib_logging = importlib.import_module("logging")
    sys.modules["logging"] = _stdlib_logging
    for _k, _v in _stdlib_logging.__dict__.items():
        globals()[_k] = _v
else:
    from neuralops.benchmarks.run_logging import JSONLHandler, setup_run_logger

    __all__ = ["JSONLHandler", "setup_run_logger"]
