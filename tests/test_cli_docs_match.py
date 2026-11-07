#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying that all CLI examples in README.md and AGENTS.md match the canonical CLI parser (F-83)."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from neuralops.benchmarks.cli import _build_parser

ROOT_DIR = Path(__file__).resolve().parent.parent


def _extract_commands(md_file: Path) -> list[str]:
    """Extract all standalone neuralops CLI command lines from markdown shell blocks."""
    lines = md_file.read_text(encoding="utf-8").splitlines()
    commands: list[str] = []
    in_code_block = False
    current_cmd = ""

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if not in_code_block and current_cmd:
                commands.append(current_cmd)
                current_cmd = ""
            continue

        if not in_code_block:
            continue

        if not stripped or stripped.startswith("#"):
            continue

        line = stripped
        if line.startswith("$"):
            line = line[1:].strip()

        if current_cmd:
            current_cmd += " " + line
        elif line.startswith(("neuralops ", "python -m neuralops ", "nops ")):
            current_cmd = line

        if current_cmd:
            if current_cmd.endswith("\\"):
                current_cmd = current_cmd[:-1].strip()
            else:
                commands.append(current_cmd)
                current_cmd = ""

    return commands


def test_readme_cli_examples_parse() -> None:
    """Extract and assert that CLI invocations in README.md and AGENTS.md parse cleanly against _build_parser()."""
    parser = _build_parser()

    target_files = [ROOT_DIR / "README.md", ROOT_DIR / "AGENTS.md"]
    tested_count = 0

    for doc_path in target_files:
        assert doc_path.exists(), f"Missing documentation file: {doc_path}"
        cmds = _extract_commands(doc_path)

        for cmd in cmds:
            # Strip runner prefixes
            cleaned = re.sub(r"^python\s+-m\s+neuralops\s+", "", cmd)
            cleaned = re.sub(r"^(?:neuralops|nops)\s+", "", cleaned)
            # Remove inline comments, shell redirects and pipes
            cleaned = cleaned.split("#")[0].split("|")[0].split(">")[0].strip()

            tokens = shlex.split(cleaned)
            if not tokens or tokens[0] in ("--help", "-h", "--version", "-v", "install", "pip", "build"):
                continue

            try:
                parser.parse_args(tokens)
                tested_count += 1
            except SystemExit as exc:
                pytest.fail(f"Failed to parse CLI example from {doc_path.name}: '{cmd}' (tokens: {tokens}). Error code: {exc.code}")

    assert tested_count > 0, "No CLI commands were extracted for verification"
