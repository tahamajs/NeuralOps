#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Standalone Evaluation CLI Delegation and Flag Parity (F-63).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuralops.cli.eval_cmd import build_eval_parser
from neuralops.evaluation.standalone.cli import _build_parser, main


def test_standalone_cli_delegates_to_eval_cmd() -> None:
    """Verify neuralops.evaluation.standalone.cli.main delegates directly to cmd_eval."""
    mock_cmd_eval = MagicMock(return_value=0)

    with patch("neuralops.evaluation.standalone.cli.cmd_eval", mock_cmd_eval):
        exit_code = main(["--model", "gpt2", "--dataset", "wikitext"])
        assert exit_code == 0
        assert mock_cmd_eval.call_count == 1

        called_args = mock_cmd_eval.call_args[0][0]
        assert called_args.model == "gpt2"
        assert called_args.dataset == "wikitext"


def test_eval_flags_match() -> None:
    """Verify that options between neuralops-eval and neuralops eval match."""
    standalone_parser = _build_parser()
    eval_cmd_parser = build_eval_parser()

    standalone_options = {
        opt
        for action in standalone_parser._actions
        for opt in action.option_strings
    }
    canonical_options = {
        opt
        for action in eval_cmd_parser._actions
        for opt in action.option_strings
    }

    # All canonical options must be present in standalone parser
    missing_in_standalone = canonical_options - standalone_options
    assert not missing_in_standalone, (
        f"Standalone eval CLI is missing options present in canonical eval: {missing_in_standalone}"
    )

    # Standalone parser options should match canonical options
    difference = standalone_options.symmetric_difference(canonical_options)
    assert not difference, f"Discrepancy in eval options: {difference}"
