#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive unit tests for all neuralops.cli commands, parsers, and handlers."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

import neuralops.cli as cli
from neuralops.cli.compress_cmd import build_compress_parser, cmd_compress
from neuralops.cli.doctor import cmd_doctor
from neuralops.cli.eval_cmd import build_eval_parser
from neuralops.cli.info import cmd_info
from neuralops.cli.list_cmd import build_list_parser, cmd_list
from neuralops.cli.serve_cmd import build_serve_parser
from neuralops.cli.surgery_cmd import add_surgery_subparser, cmd_surgery, handle_surgery_command
from neuralops.cli.test_cmd import build_test_parser
from neuralops.cli.train_cmd import build_train_parser


def test_cli_submodules_importable_via_getattr():
    """Verify that all owned CLI modules are accessible as attributes on neuralops.cli."""
    expected_modules = [
        "list_cmd",
        "info",
        "compress_cmd",
        "train_cmd",
        "eval_cmd",
        "serve_cmd",
        "test_cmd",
        "doctor",
        "surgery_cmd",
    ]
    for mod_name in expected_modules:
        mod = getattr(cli, mod_name)
        assert mod is not None, f"Failed to access neuralops.cli.{mod_name}"


def test_list_command_parser_and_execution(capsys):
    """Verify list command parser produces help and cmd_list runs cleanly."""
    parser = build_list_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--type" in help_text
    assert "--backend" in help_text

    # Run cmd_list with table output
    args = argparse.Namespace(type="attention", family=None, backend="all", json=False)
    ret = cmd_list(args)
    assert ret == 0

    # Run cmd_list with JSON output
    args_json = argparse.Namespace(type="compressor", family=None, backend="all", json=True)
    ret_json = cmd_list(args_json)
    assert ret_json == 0


def test_info_command_execution(capsys):
    """Verify cmd_info executes without error and displays system information."""
    ret = cmd_info()
    assert ret == 0


def test_doctor_command_execution():
    """Verify cmd_doctor executes and returns a valid integer status code."""
    ret = cmd_doctor()
    assert ret in (0, 1)


def test_compress_command_parser_and_validation():
    """Verify compress parser produces help and validates arguments."""
    parser = build_compress_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--compressor" in help_text
    assert "--rank" in help_text

    # Missing model returns error code 1
    args_no_model = argparse.Namespace(model=None)
    assert cmd_compress(args_no_model) == 1

    # Invalid rank raises ValueError
    args_bad_rank = argparse.Namespace(model="gpt2", rank=-5, compressor="direct_svd")
    with pytest.raises(ValueError, match="--rank must be a positive integer"):
        cmd_compress(args_bad_rank)

    # Invalid target-ratio raises ValueError
    args_bad_ratio = argparse.Namespace(model="gpt2", rank=16, target_ratio=2.5, compressor="direct_svd")
    with pytest.raises(ValueError, match="--target-ratio must be in the interval"):
        cmd_compress(args_bad_ratio)


def test_train_command_parser():
    """Verify train subcommand parser builds and exposes required arguments."""
    parser = build_train_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--model" in help_text
    assert "--dataset" in help_text
    assert "--epochs" in help_text
    assert "--lr" in help_text


def test_eval_command_parser():
    """Verify eval subcommand parser builds and exposes required arguments."""
    parser = build_eval_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--model" in help_text
    assert "--dataset" in help_text
    assert "--batch-size" in help_text


def test_serve_command_parser():
    """Verify serve subcommand parser builds and exposes server arguments."""
    parser = build_serve_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--model" in help_text
    assert "--host" in help_text
    assert "--port" in help_text
    assert "--engine" in help_text


def test_test_command_parser():
    """Verify test subcommand parser builds and exposes test suite arguments."""
    parser = build_test_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "--all" in help_text
    assert "--health" in help_text
    assert "--kernels" in help_text
    assert "--surgery" in help_text


def test_surgery_command_parser_and_dispatch():
    """Verify surgery subparser attaches correctly and dispatches."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    add_surgery_subparser(subparsers)

    help_text = parser.format_help()
    assert "surgery-recipe" in help_text

    # Missing action returns error code 1
    args_no_action = argparse.Namespace(surgery_action=None, model=None, attention=None)
    ret = handle_surgery_command(args_no_action)
    assert ret == 1

    # cmd_surgery default fallback
    ret_default = cmd_surgery(args_no_action)
    assert ret_default == 0
