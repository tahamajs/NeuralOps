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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from neuralops.reporting.environment import capture_environment


def _count_parameters(model: Optional[nn.Module]) -> int:
    if model is None:
        return 0
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:  # allowed-swallow: model parameter inspection fallback
        return 0


class EvidenceRecord(dict):
    """Frozen Phase 3 Evidence dictionary with backward-compatible attribute access."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        if "metrics" not in self:
            self["metrics"] = {}

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self, indent=indent, default=str)

    def save(self, path: Union[str, Path]) -> None:
        save_evidence(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> EvidenceRecord:
        return cls(load_evidence(path))


def build_evidence(
    arg1: Any,
    arg2: Optional[Any] = None,
    seed: Optional[int] = 42,
    deterministic: bool = True,
    language_version: str = "v1.0.0",
) -> EvidenceRecord:
    """Construct a frozen Evidence schema dict from an executed pipeline result.

    Supports both signatures:
    - build_evidence(pipeline_result, recipe=None, ...)
    - build_evidence(recipe, pipeline_result, ...)
    """
    # Disambiguate arguments
    recipe = None
    pipeline_result = None

    from neuralops.recipe.schema import Recipe

    if isinstance(arg1, Recipe):
        recipe = arg1
        pipeline_result = arg2
    else:
        pipeline_result = arg1
        recipe = arg2

    # 1. Resolve pipeline result attributes
    if isinstance(pipeline_result, dict):
        res_dict = pipeline_result
        success = res_dict.get("success", True)
        total_time = float(res_dict.get("total_time", res_dict.get("total_time_seconds", 0.0)))
        raw_stages = res_dict.get("stage_records", res_dict.get("stages", []))
        rec_hash = res_dict.get("recipe_hash", "unknown")
        final_model = res_dict.get("final_model")
        val_data = res_dict.get("validation")
    elif pipeline_result is not None:
        success = bool(getattr(pipeline_result, "success", True))
        total_time = float(
            getattr(
                pipeline_result,
                "total_time",
                getattr(pipeline_result, "duration_seconds", 0.0),
            )
        )
        raw_stages = getattr(
            pipeline_result,
            "stage_records",
            getattr(pipeline_result, "stages", []),
        )
        rec_hash = str(getattr(pipeline_result, "recipe_hash", "unknown"))
        final_model = getattr(pipeline_result, "final_model", None)
        val_obj = getattr(pipeline_result, "validation", None)
        val_data = val_obj.to_dict() if hasattr(val_obj, "to_dict") else val_obj
    else:
        success = True
        total_time = 0.0
        raw_stages = []
        rec_hash = "unknown"
        final_model = None
        val_data = None

    # 2. Derive status: SUCCESS / FAILED / PARTIAL
    stages_data: List[Dict[str, Any]] = []
    success_count = 0
    total_comp_ratio = 1.0

    for s in raw_stages:
        sd = s.to_dict() if hasattr(s, "to_dict") else dict(s)
        stages_data.append(sd)
        if sd.get("success", True):
            success_count += 1
        ev = sd.get("evidence", {})
        if isinstance(ev, dict) and "compression_ratio" in ev:
            total_comp_ratio = float(ev["compression_ratio"])

    if success:
        status = "SUCCESS"
    elif success_count > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"

    # 3. Model information
    model_source = "unknown"
    model_rev = "main"
    recipe_name = "unnamed_recipe"

    if recipe is not None:
        recipe_name = getattr(getattr(recipe, "metadata", None), "name", "unnamed_recipe")
        m_spec = getattr(recipe, "model", None)
        if m_spec:
            model_source = getattr(m_spec, "source", getattr(m_spec, "family", "unknown")) or "unknown"
            model_rev = getattr(m_spec, "revision", "main") or "main"
        if rec_hash == "unknown":
            try:
                from neuralops.recipe.canonical import recipe_hash

                rec_hash = recipe_hash(recipe)
            except Exception:  # allowed-swallow: optional recipe hash derivation
                pass

    final_params = _count_parameters(final_model)

    # 4. Environment capture
    env = capture_environment(seed=seed)

    # 5. Reproducibility metadata
    op_versions: Dict[str, str] = {}
    for sd in stages_data:
        op_name = sd.get("operation_name", sd.get("operation_type", "unknown"))
        op_versions[op_name] = "1.0.0"

    initial_metrics = {
        "duration_seconds": total_time,
        "compression_ratio": total_comp_ratio,
        "stages_executed": len(stages_data),
    }

    evidence = EvidenceRecord({
        "recipe_hash": rec_hash,
        "recipe_name": recipe_name,
        "model_source": model_source,
        "model": {
            "source": model_source,
            "revision": model_rev,
            "final_parameter_count": final_params,
        },
        "environment": env,
        "reproducibility": {
            "seed": seed,
            "deterministic": deterministic,
            "language_version": language_version,
            "op_versions": op_versions,
        },
        "stages": stages_data,
        "validation": val_data,
        "metrics": initial_metrics,
        "total_time_seconds": total_time,
        "status": status,
        "success": success,
    })

    return evidence


def save_evidence(evidence: Dict[str, Any], path: Union[str, Path]) -> None:
    """Serialize evidence dictionary to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, default=str)


def load_evidence(path: Union[str, Path]) -> EvidenceRecord:
    """Load and parse evidence dictionary from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvidenceRecord(data)


__all__ = ["EvidenceRecord", "build_evidence", "load_evidence", "save_evidence"]
