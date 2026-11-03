#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Module command family: list, inspect, validate, scaffold."""

from __future__ import annotations

import argparse
import json

from neuralops.cli.core.errors import ExitCode, NumericalGateError
from neuralops.cli.core.output import CliReporter


def handle_list(args: argparse.Namespace) -> int:
    """Lists registered attention mechanisms and compression engines."""
    from neuralops.cli.list_cmd import cmd_list
    ret = cmd_list(args)
    return int(ret) if ret is not None else int(ExitCode.SUCCESS)


def handle_inspect(args: argparse.Namespace) -> int:
    """Inspect mathematical specification, shapes, and quickstart code for a module."""
    from neuralops.core.inspector import inspect_architecture, render_architecture_card
    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    spec = inspect_architecture(args.name)
    if getattr(args, "json", False):
        reporter.print_json(spec.to_dict())
    elif getattr(args, "code_only", False):
        print(f"# === Quickstart: {spec.name} ({spec.category}) ===")
        print(spec.quickstart_code.get("pytorch_direct", ""))
        print("\n# === Model Surgery & Micro-Healing ===")
        print(spec.quickstart_code.get("surgical_transplant", ""))
    else:
        print(render_architecture_card(spec))
        if getattr(args, "roofline", False):
            from neuralops.core.roofline import profile_roofline, render_roofline_terminal
            print("\n" + "=" * 74)
            print(f"  🔬 LIVE HARDWARE ROOFLINE ANALYSIS: {spec.name.upper()}")
            print("=" * 74)
            try:
                rep = profile_roofline(module_or_name=spec.name, batch_size=2, seq_len=1024)
                print(render_roofline_terminal(rep))
            except Exception as e_roof:
                print(f"[-] Roofline profiling skipped or failed: {e_roof}")
    return int(ExitCode.SUCCESS)


def handle_validate(args: argparse.Namespace) -> int:
    """Validate a module against correctness, numerical stability, and causal masking gates."""
    from neuralops.core.extensibility import validate_module
    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    reporter.log(f"[+] Running compliance validation for module: '{args.name}'...")
    res = validate_module(
        args.name,
        batch_size=getattr(args, "batch_size", 2),
        seq_len=getattr(args, "seq_len", 128),
        dim=getattr(args, "dim", 256),
        num_heads=getattr(args, "heads", 4),
        rank=getattr(args, "rank", 16),
        device=getattr(args, "device", "cpu"),
    )
    if getattr(args, "json", False):
        reporter.print_json(res)
    else:
        status_str = "PASS" if res.get("all_passed") else "FAIL"
        reporter.log(f"[{status_str}] Module validation completed: {res.get('checks_passed')}/{res.get('total_checks')} checks satisfied.")
        for check, outcome in res.get("details", {}).items():
            reporter.log(f"  - {check}: {outcome}")

    if not res.get("all_passed", False):
        raise NumericalGateError(f"Validation failed for module '{args.name}'", exit_code=ExitCode.GATE_VERIFICATION_FAILED)
    return int(ExitCode.SUCCESS)


def handle_scaffold(args: argparse.Namespace) -> int:
    """Scaffold boilerplate code for a new attention or compressor module."""
    from neuralops.core.extensibility import scaffold_module
    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    reporter.log(f"[+] Scaffolding {args.category} module '{args.name}' (family: {args.family})...")
    code = scaffold_module(args.name, category=args.category, family=args.family, output_file=args.output)
    if args.output:
        reporter.log(f"[✓] Scaffold written to '{args.output}'.")
    else:
        print(code)
    return int(ExitCode.SUCCESS)
