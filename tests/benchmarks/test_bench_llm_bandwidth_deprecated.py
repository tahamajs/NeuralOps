#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests verifying deprecation warnings for bench_llm_bandwidth argparse/CLI."""

import pytest

from neuralops.benchmarks.bench_llm_bandwidth import (
    build_parser,
    flush_l2_cache,
    main,
)


def test_build_parser_emits_deprecation_warning():
    """Verify that build_parser emits a clear DeprecationWarning pointing to neuralops benchmark."""
    with pytest.deprecated_call(match="bench_llm_bandwidth CLI/argparse is deprecated"):
        parser = build_parser()
    assert parser is not None
    # Verify default device is auto
    args = parser.parse_args([])
    assert args.device == "auto"


def test_main_cli_emits_deprecation_warning(monkeypatch):
    """Verify that invoking main emits DeprecationWarning and executes correctly."""
    monkeypatch.setattr(
        "neuralops.benchmarks.bench_llm_bandwidth.run_bandwidth_benchmark",
        lambda **kwargs: [],
    )

    with pytest.deprecated_call(match="bench_llm_bandwidth CLI/argparse is deprecated"):
        exit_code = main(["--quick"])
    assert exit_code == 0


def test_flush_l2_cache_cpu_noop():
    """Verify flush_l2_cache handles CPU / non-CUDA device safely without error."""
    # Should be a no-op on CPU or when CUDA unavailable
    flush_l2_cache(device="cpu")
