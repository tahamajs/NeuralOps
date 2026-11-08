#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying that all console scripts route to the canonical CLI entry point
and that canonical CLI handlers import and execute without errors.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def test_console_scripts_resolve_to_canonical_parser():
    """Assert all console scripts in pyproject.toml resolve to neuralops.benchmarks.cli:main."""
    root_dir = Path(__file__).resolve().parents[2]
    pyproject_path = root_dir / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml missing"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    scripts = data.get("project", {}).get("scripts", {})
    expected_scripts = ["neuralops", "neuralops-cli", "nops", "nops-cli"]

    for script_name in expected_scripts:
        assert script_name in scripts, f"Console script '{script_name}' not defined in pyproject.toml"
        target = scripts[script_name]
        assert target == "neuralops.benchmarks.cli:main", (
            f"Script '{script_name}' points to '{target}' instead of 'neuralops.benchmarks.cli:main'"
        )

        # Verify entry point target is dynamically importable
        mod_name, fn_name = target.split(":")
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, fn_name), f"Module '{mod_name}' has no attribute '{fn_name}'"
        assert callable(getattr(mod, fn_name)), f"Attribute '{fn_name}' in '{mod_name}' is not callable"


def test_doctor_info_compat_handlers_import():
    """Verify that canonical CLI handlers for doctor, info, and compat import and execute cleanly."""
    # 1. Test canonical doctor handler
    from neuralops.cli.doctor import cmd_doctor
    assert callable(cmd_doctor)

    # 2. Test canonical info handler
    from neuralops.cli.info import cmd_info
    assert callable(cmd_info)

    # 3. Test canonical compat handler
    from neuralops.export.runtime_compat import check_compatibility
    assert callable(check_compatibility)

    # 4. Run CLI subcommands with --help to confirm exit code 0
    res_doctor = subprocess.run(
        [sys.executable, "-m", "neuralops", "doctor", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_doctor.returncode == 0, f"neuralops doctor --help failed: {res_doctor.stderr}"
    assert "doctor" in res_doctor.stdout

    res_info = subprocess.run(
        [sys.executable, "-m", "neuralops", "info", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_info.returncode == 0, f"neuralops info --help failed: {res_info.stderr}"
    assert "info" in res_info.stdout

    res_compat = subprocess.run(
        [sys.executable, "-m", "neuralops", "compat", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_compat.returncode == 0, f"neuralops compat --help failed: {res_compat.stderr}"
    assert "compat" in res_compat.stdout


def test_ui_subcommand_reaches_web_ui(monkeypatch):
    """Verify that neuralops ui parses web UI options and dispatches to cmd_ui."""
    from neuralops.benchmarks.cli import build_parser, _dispatch
    from neuralops.ui.cli import cmd_ui

    parser = build_parser()
    args = parser.parse_args(["ui", "--port", "9090", "--host", "127.0.0.1", "--no-browser"])
    assert args.subcommand == "ui"
    assert args.port == 9090
    assert args.host == "127.0.0.1"
    assert args.no_browser is True

    called = {}

    def mock_cmd_ui(parsed_args):
        called["args"] = parsed_args
        return 0

    monkeypatch.setattr("neuralops.ui.cli.cmd_ui", mock_cmd_ui)
    ret = _dispatch(args)
    assert ret == 0
    assert "args" in called
    assert called["args"].port == 9090


def test_tui_subcommand_reaches_terminal_ui(monkeypatch):
    """Verify that neuralops tui parses TUI options and dispatches to run_interactive_ui."""
    from neuralops.benchmarks.cli import build_parser, _dispatch

    parser = build_parser()
    args = parser.parse_args(["tui", "--snapshot", "--inspect", "gated_delta_net"])
    assert args.subcommand == "tui"
    assert args.snapshot is True
    assert args.inspect == "gated_delta_net"

    called = {}

    def mock_run_interactive_ui(inspect_target=None, snapshot=False):
        called["inspect_target"] = inspect_target
        called["snapshot"] = snapshot

    monkeypatch.setattr("neuralops.benchmarks.terminal_ui.run_interactive_ui", mock_run_interactive_ui)
    ret = _dispatch(args)
    assert ret == 0
    assert called.get("inspect_target") == "gated_delta_net"
    assert called.get("snapshot") is True


def test_no_shadow_cli_imports_missing_modules():
    """Verify all CLI modules across packages import without ModuleNotFoundError (F-50)."""
    cli_modules = [
        "neuralops.benchmarks.cli",
        "neuralops.cli.doctor",
        "neuralops.cli.info",
        "neuralops.cli.main",
        "neuralops.cli.core.app",
        "neuralops.cli.core.commands.system",
        "neuralops.cli.core.commands.benchmark",
        "neuralops.cli.core.commands.model",
        "neuralops.cli.core.commands.module",
        "neuralops.cli.core.commands.results",
        "neuralops.export.runtime_compat",
    ]
    for mod_name in cli_modules:
        try:
            mod = importlib.import_module(mod_name)
            assert mod is not None
        except ModuleNotFoundError as exc:
            pytest.fail(f"Module {mod_name} failed to import due to missing dependency/module: {exc}")


def test_shadow_cli_shims_emit_deprecation_warning():
    """Verify that accessing shadow CLIs emits a clear DeprecationWarning and delegates to canonical parser."""
    import warnings
    from neuralops.cli import main as shadow_main
    from neuralops.cli.core import app as shadow_app

    with pytest.deprecated_call(match="deprecated and will be removed in v3.0.0"):
        parser1 = shadow_main.build_main_parser()
        assert parser1 is not None

    with pytest.deprecated_call(match="deprecated and will be removed in v3.0.0"):
        parser2 = shadow_app.build_app_parser()
        assert parser2 is not None
