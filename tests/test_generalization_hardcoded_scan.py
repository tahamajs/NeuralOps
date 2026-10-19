# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verification: Hardcoded String / Shape Scan."""
from pathlib import Path

import pytest


def test_no_hardcoded_roberta_in_core():
    root = Path(__file__).resolve().parent.parent / "core"
    for py_file in root.glob("*.py"):
        text = py_file.read_text()
        if py_file.name == "architecture.py":
            assert 'family == "roberta"' not in text
            assert 'family == "gpt2"' not in text
