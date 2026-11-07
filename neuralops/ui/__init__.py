#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Web UI Package.

Minimal, publication-grade web dashboard and interactive exploration interface.
"""

from __future__ import annotations

__all__ = ["build_ui_parser", "cmd_ui", "create_app"]

from neuralops.ui.cli import build_ui_parser, cmd_ui
from neuralops.ui.server import create_app
