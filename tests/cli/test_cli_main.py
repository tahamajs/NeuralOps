#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for NeuralOps CLI Main Router and Fail-Fast Subcommand Registration (F-58).
"""

from __future__ import annotations

import sys
import pytest

from neuralops.cli.main import build_main_parser, main


def test_all_subcommands_registered() -> None:
    """Verify all 10 core subcommands are registered in the main parser."""
    parser = build_main_parser()

    # Extract registered subcommand names from subparsers action
    subparser_actions = [
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(subparser_actions) == 1, "Expected exactly one subparser action"

    registered_choices = set(subparser_actions[0].choices.keys())

    expected_subcommands = {
        "train",
        "compress",
        "eval",
        "serve",
        "test",
        "doctor",
        "info",
        "surgery",
        "transplant",
        "ui",
    }

    missing = expected_subcommands - registered_choices
    assert not missing, f"Expected subcommands missing from CLI: {missing}"


def test_broken_subcommand_import_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a broken subcommand import fails fast with an exception instead of silently dropping."""
    # Monkeypatch sys.modules to simulate broken import for neuralops.cli.compress_cmd
    monkeypatch.setitem(sys.modules, "neuralops.cli.compress_cmd", None)

    with pytest.raises((ImportError, ModuleNotFoundError)):
        build_main_parser()
