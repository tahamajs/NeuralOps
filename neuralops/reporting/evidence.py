#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Evidence Record and Lineage Provenance Artifact.

Specification Section 9: Execution Evidence.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neuralops.executor.result import PipelineResult
from neuralops.recipe.canonical import recipe_hash
from neuralops.recipe.schema import Recipe
from neuralops.reporting.environment import capture_environment


@dataclass
class EvidenceRecord:
    """Verifiable cryptographic and empirical transformation artifact."""

    recipe_hash: str
    recipe_name: str
    model_source: str
    timestamp: str
    environment: Dict[str, Any]
    stages: List[Dict[str, Any]] = field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> EvidenceRecord:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def build_evidence(
    recipe: Recipe,
    pipeline_result: PipelineResult,
    seed: Optional[int] = None,
) -> EvidenceRecord:
    """Construct an EvidenceRecord from an executed recipe result."""
    r_hash = recipe_hash(recipe)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env = capture_environment(seed=seed)

    stages_data: List[Dict[str, Any]] = []
    total_comp_ratio = 1.0

    for s in pipeline_result.stage_records:
        sd = s.to_dict()
        stages_data.append(sd)
        ev = s.evidence or {}
        if "compression_ratio" in ev:
            total_comp_ratio = float(ev["compression_ratio"])

    metrics = {
        "duration_seconds": pipeline_result.duration_seconds,
        "compression_ratio": total_comp_ratio,
        "stages_executed": len(pipeline_result.stage_records),
    }

    val_data = pipeline_result.validation.to_dict() if pipeline_result.validation else None

    return EvidenceRecord(
        recipe_hash=r_hash,
        recipe_name=recipe.metadata.name,
        model_source=recipe.model.source or recipe.model.family or "unknown",
        timestamp=now_iso,
        environment=env,
        stages=stages_data,
        validation=val_data,
        metrics=metrics,
        success=pipeline_result.success,
    )


__all__ = ["EvidenceRecord", "build_evidence"]
