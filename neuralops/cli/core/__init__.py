#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Modular CLI package for NeuralOps."""

from neuralops.cli.core.app import app_main, build_app_parser, dispatch_command
from neuralops.cli.core.config_merge import load_effective_config
from neuralops.cli.core.devices import resolve_device
from neuralops.cli.core.errors import CliError, ExitCode, emit_cli_error
from neuralops.cli.core.locks import create_run_directory, run_directory_lock
from neuralops.cli.core.output import CliReporter

__all__ = [
    "app_main",
    "build_app_parser",
    "dispatch_command",
    "ExitCode",
    "CliError",
    "emit_cli_error",
    "resolve_device",
    "load_effective_config",
    "create_run_directory",
    "run_directory_lock",
    "CliReporter",
]
