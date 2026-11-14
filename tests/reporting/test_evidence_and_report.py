#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Evidence Record serialization, loading, and Markdown report rendering.
"""

from neuralops.executor.result import PipelineResult, StageRecord
from neuralops.recipe.loader import loads_recipe
from neuralops.reporting.evidence import build_evidence
from neuralops.reporting.report import render_report

SAMPLE_RECIPE = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: evidence-test
  version: 1.0.0
  description: "Evidence test recipe"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: step-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""


def test_build_evidence_and_render_report(tmp_path):
    recipe = loads_recipe(SAMPLE_RECIPE)
    res = PipelineResult(
        final_model=None,
        success=True,
        stage_records=[
            StageRecord(
                operation_name="weights.compress",
                duration=0.15,
                memory_before=1000,
                memory_after=500,
                stage_id="step-1",
                success=True,
                evidence={"compression_ratio": 0.49, "modules_compressed": 4},
            )
        ],
        total_time=0.18,
    )

    ev = build_evidence(recipe, res, seed=42)
    assert ev.success
    assert ev.recipe_name == "evidence-test"
    assert ev.environment["seed"] == 42
    assert ev.metrics["compression_ratio"] == 0.49

    # Save and reload
    out_file = tmp_path / "evidence.json"
    ev.save(out_file)
    assert out_file.exists()

    loaded = ev.load(out_file)
    assert loaded.recipe_hash == ev.recipe_hash
    assert loaded.success == ev.success

    # Render Markdown report
    md = render_report(ev)
    assert "# NeuralOps Transformation Report: evidence-test" in md
    assert "✓ PASS" in md
    assert "0.4900" in md
