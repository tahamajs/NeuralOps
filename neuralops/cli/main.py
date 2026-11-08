#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Unified CLI Main Router (Deprecated Shim).

DEPRECATION NOTICE:
  This module is a backward-compatibility shim. The canonical CLI entrypoint
  for NeuralOps is `neuralops.benchmarks.cli:main` (Decision 3).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence
import warnings

from neuralops.benchmarks.cli import _build_parser as _canonical_build_parser
from neuralops.benchmarks.cli import main as _canonical_main


def _emit_deprecation_warning() -> None:
    warnings.warn(
        "neuralops.cli.main is deprecated and will be removed in v3.0.0; "
        "use neuralops.benchmarks.cli:main instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def build_main_parser() -> argparse.ArgumentParser:
    """Return canonical CLI argument parser."""
    _emit_deprecation_warning()
    return _canonical_build_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute canonical CLI with deprecation notice."""
    _emit_deprecation_warning()
    args_list = list(argv) if argv is not None else None
    return _canonical_main(args_list)


if __name__ == "__main__":
    sys.exit(main())
