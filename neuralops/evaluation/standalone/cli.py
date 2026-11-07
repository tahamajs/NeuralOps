#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Standalone Evaluation Command-Line Interface (F-63).

Official alias and thin delegator to canonical `neuralops eval` (`neuralops.cli.eval_cmd`),
preventing divergent evaluation logic across the codebase.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from neuralops.cli.eval_cmd import build_eval_parser, cmd_eval


def _build_parser() -> argparse.ArgumentParser:
    """Build the standalone evaluation parser harmonised with canonical `neuralops eval`."""
    return build_eval_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute evaluation by delegating directly to canonical cmd_eval."""
    parser = _build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)
    return cmd_eval(args)


__all__ = ["_build_parser", "build_eval_parser", "cmd_eval", "main"]

if __name__ == "__main__":
    sys.exit(main())
