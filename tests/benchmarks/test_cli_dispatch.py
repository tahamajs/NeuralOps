#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for CLI benchmark dispatch and argument resolution.
Tests argument parsing, defaults, validation, and error propagation
for 'eval' and legacy benchmark subcommands.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from neuralops.cli.bench_cmd import add_eval_subparser, build_parser, cmd_bench, handle_eval


class TestCLIDispatch(unittest.TestCase):
    """Test suite verifying CLI argument parser and command dispatch contracts."""

    def setUp(self) -> None:
        self.parser = build_parser()

    def test_eval_subcommand_arguments_parsing(self) -> None:
        """Verify parsing of all flags for 'eval' subcommand."""
        cmd_line = [
            "eval",
            "--model", "meta-llama/Llama-2-7b-hf",
            "--sweep-attentions", "fms_scan,gla_attention",
            "--sweep-compressors", "svd,hnc",
            "--benchmarks", "perplexity,flash_attn_tflops",
            "--pre-train-steps", "100",
            "--pre-train-engine", "multi_level_kd",
            "--pre-train-optimizer", "muon",
            "--pre-train-scheduler", "wsd",
            "--auto-calibrate",
            "--resume",
            "--output-dir", "results/test_eval",
            "--device", "cpu",
        ]
        args = self.parser.parse_args(cmd_line)
        self.assertEqual(args.subcommand, "eval")
        self.assertEqual(args.model, "meta-llama/Llama-2-7b-hf")
        self.assertEqual(args.sweep_attentions, "fms_scan,gla_attention")
        self.assertEqual(args.sweep_compressors, "svd,hnc")
        self.assertEqual(args.benchmarks, "perplexity,flash_attn_tflops")
        self.assertEqual(args.pre_train_steps, 100)
        self.assertEqual(args.pre_train_engine, "multi_level_kd")
        self.assertEqual(args.pre_train_optimizer, "muon")
        self.assertEqual(args.pre_train_scheduler, "wsd")
        self.assertTrue(args.auto_calibrate)
        self.assertTrue(args.resume)
        self.assertEqual(args.output_dir, "results/test_eval")
        self.assertEqual(args.device, "cpu")

    def test_eval_subcommand_defaults(self) -> None:
        """Verify default argument values for 'eval' subcommand."""
        args = self.parser.parse_args(["eval"])
        self.assertEqual(args.model, "gpt2")
        self.assertIsNone(args.sweep_attentions)
        self.assertIsNone(args.sweep_compressors)
        self.assertEqual(args.benchmarks, "perplexity")
        self.assertEqual(args.pre_train_steps, 0)
        self.assertEqual(args.pre_train_engine, "multi_level_kd")
        self.assertEqual(args.pre_train_optimizer, "muon")
        self.assertEqual(args.pre_train_scheduler, "wsd")
        self.assertFalse(args.auto_calibrate)
        self.assertFalse(args.resume)
        self.assertEqual(args.output_dir, "results/eval_matrix")
        self.assertEqual(args.device, "cpu")

    def test_legacy_benchmark_arguments_compatibility(self) -> None:
        """Verify backward compatibility for legacy 'neuralops bench' arguments."""
        cmd_line = [
            "--module", "fms_scan",
            "--mode", "profile",
            "--model", "gpt2",
            "--dataset", "sst2",
            "--rank", "16",
            "--device", "cpu",
        ]
        args = self.parser.parse_args(cmd_line)
        self.assertIsNone(args.subcommand)
        self.assertEqual(args.module, "fms_scan")
        self.assertEqual(args.mode, "profile")
        self.assertEqual(args.model, "gpt2")
        self.assertEqual(args.dataset, "sst2")
        self.assertEqual(args.rank, 16)
        self.assertEqual(args.device, "cpu")

    def test_cmd_bench_dispatches_to_handle_eval(self) -> None:
        """Verify cmd_bench routes 'eval' subcommand invocation to handle_eval."""
        args = self.parser.parse_args(["eval", "--model", "gpt2", "--device", "cpu"])
        with patch("neuralops.cli.bench_cmd.handle_eval", return_value=0) as mock_handle:
            code = cmd_bench(args)
            self.assertEqual(code, 0)
            mock_handle.assert_called_once_with(args)

    def test_add_eval_subparser_helper(self) -> None:
        """Verify add_eval_subparser attaches to external subparsers correctly."""
        base_parser = argparse.ArgumentParser()
        subparsers = base_parser.add_subparsers(dest="action")
        subparser = add_eval_subparser(subparsers)
        self.assertIsNotNone(subparser)

        parsed = base_parser.parse_args(["eval", "--pre-train-steps", "50"])
        self.assertEqual(parsed.action, "eval")
        self.assertEqual(parsed.pre_train_steps, 50)


if __name__ == "__main__":
    unittest.main()
