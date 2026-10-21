#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests that sweep reports only derive metrics present in the input data."""

import pandas as pd
import pytest

from neuralops.analysis.generate_sweep_publication_artifacts import (
    load_and_enrich_summary,
)


def test_missing_baseline_is_not_replaced_with_a_fabricated_score(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame(
        [{"module_name": "cosformer", "status": "PASS", "accuracy_pct": 87.0}]
    ).to_csv(summary, index=False)

    result = load_and_enrich_summary(summary)

    assert pd.isna(result.loc[0, "baseline_acc"])
    assert pd.isna(result.loc[0, "delta_acc"])
    assert pd.isna(result.loc[0, "retention_pct"])


def test_summary_without_required_metric_columns_is_rejected(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame([{"module_name": "cosformer", "status": "PASS"}]).to_csv(
        summary, index=False
    )

    with pytest.raises(ValueError, match="accuracy_pct"):
        load_and_enrich_summary(summary)
