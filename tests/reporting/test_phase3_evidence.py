#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Structured Evidence and Reporting (Sub-Agent C3).
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from neuralops.reporting import (
    build_evidence,
    capture_environment,
    load_evidence,
    render_report,
    save_evidence,
    save_report,
)


@dataclass
class DummyStageRecord:
    stage_id: str
    operation_name: str
    duration: float
    success: bool
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "operation_name": self.operation_name,
            "duration": self.duration,
            "success": self.success,
            "evidence": self.evidence,
        }


@dataclass
class DummyPipelineResult:
    final_model: nn.Module
    stage_records: List[DummyStageRecord]
    recipe_hash: str
    total_time: float
    success: bool


def test_c3_capture_environment_cpu():
    """Verify capture_environment captures all schema keys on CPU."""
    env = capture_environment(seed=42)

    assert "neuralops" in env
    assert "torch" in env
    assert "transformers" in env
    assert "cuda" in env
    assert "gpu" in env
    assert "hostname" in env
    assert "git_sha" in env
    assert env["seed"] == 42

    if not torch.cuda.is_available():
        assert env["cuda"] is None
        assert env["gpu"] is None


def test_c3_build_evidence_schema():
    """Verify build_evidence constructs schema with all required top-level and nested fields."""
    model = nn.Linear(10, 2)
    stages = [
        DummyStageRecord(
            stage_id="s1",
            operation_name="weights.compress",
            duration=0.15,
            success=True,
            evidence={"compression_ratio": 2.0},
        )
    ]
    res = DummyPipelineResult(
        final_model=model,
        stage_records=stages,
        recipe_hash="sha256:abc12345",
        total_time=0.25,
        success=True,
    )

    ev = build_evidence(res)

    assert ev["recipe_hash"] == "sha256:abc12345"
    assert ev["status"] == "SUCCESS"
    assert ev["total_time_seconds"] == 0.25

    # Model info
    assert "model" in ev
    assert "source" in ev["model"]
    assert "revision" in ev["model"]
    assert ev["model"]["final_parameter_count"] == sum(p.numel() for p in model.parameters())

    # Environment
    assert "environment" in ev
    assert ev["environment"]["neuralops"] is not None

    # Reproducibility
    assert "reproducibility" in ev
    assert ev["reproducibility"]["seed"] == 42
    assert ev["reproducibility"]["deterministic"] is True
    assert ev["reproducibility"]["language_version"] == "v1.0.0"
    assert "weights.compress" in ev["reproducibility"]["op_versions"]

    # Stages
    assert len(ev["stages"]) == 1
    assert ev["stages"][0]["stage_id"] == "s1"


def test_c3_evidence_save_and_load_round_trip(tmp_path):
    """Verify save_evidence and load_evidence round-trip byte-for-byte / value parity."""
    model = nn.Linear(8, 2)
    res = DummyPipelineResult(
        final_model=model,
        stage_records=[],
        recipe_hash="sha256:test999",
        total_time=1.0,
        success=True,
    )
    ev = build_evidence(res)

    save_path = tmp_path / "evidence.json"
    save_evidence(ev, save_path)

    loaded_ev = load_evidence(save_path)
    assert loaded_ev["recipe_hash"] == ev["recipe_hash"]
    assert loaded_ev["status"] == ev["status"]
    assert loaded_ev["total_time_seconds"] == ev["total_time_seconds"]
    assert loaded_ev["model"] == ev["model"]
    assert loaded_ev["environment"] == ev["environment"]


def test_c3_render_report_sections(tmp_path):
    """Verify render_report produces Markdown with required sections: Environment, Model, Stages, Repro."""
    model = nn.Linear(8, 2)
    stages = [
        DummyStageRecord(
            stage_id="s1",
            operation_name="weights.compress",
            duration=0.5,
            success=True,
            evidence={},
        )
    ]
    res = DummyPipelineResult(
        final_model=model,
        stage_records=stages,
        recipe_hash="sha256:fedcba",
        total_time=0.5,
        success=True,
    )
    ev = build_evidence(res)

    md = render_report(ev)
    assert "## Model" in md
    assert "## Environment" in md
    assert "## Stages" in md
    assert "## Repro" in md

    report_path = tmp_path / "report.md"
    save_report(ev, report_path)
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8") == md
