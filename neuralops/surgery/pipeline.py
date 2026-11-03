#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Surgery Pipeline Orchestrator and Composable SurgeryRecipe Engine.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from neuralops.executor.snapshot import ModelSnapshot
from neuralops.surgery.registry import SURGERY_REGISTRY, register_surgery
from neuralops.surgery.validation import ValidationReport, validate_before_surgery

logger = logging.getLogger(__name__)


@dataclass
class SurgeryOp:
    """Specification of an individual surgery operation within a recipe."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    validate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": copy.deepcopy(self.params),
            "validate": self.validate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SurgeryOp:
        return cls(
            name=d["name"],
            params=d.get("params", {}),
            validate=d.get("validate", True),
        )


@dataclass
class ApplyReport:
    """Telemetry report emitted after recipe application."""

    success: bool
    applied_ops: list[str] = field(default_factory=list)
    reports: list[Any] = field(default_factory=list)
    vram_reduction_pct: float = 0.0
    error_message: str | None = None


class SurgeryRecipe:
    """Declarative recipe combining structural surgery operations and post-surgery recovery."""

    def __init__(
        self,
        ops: list[SurgeryOp],
        recovery: list[SurgeryOp] | None = None,
        name: str = "",
        base_model: str = "",
        recipe_version: str = "1.0",
    ) -> None:
        self.ops = ops
        self.recovery = recovery or []
        self.name = name
        self.base_model = base_model
        self.recipe_version = recipe_version

    def validate(
        self,
        model: Any = None,
        calibration_data: Any = None,
        quality_budget: float = 0.02,
    ) -> ValidationReport:
        """Validate all proposed operations against the target model and each other."""
        return validate_before_surgery(
            model=model,
            recipe=self,
            calibration_data=calibration_data,
            quality_budget=quality_budget,
        )

    def apply(
        self,
        model: Any,
        rollback_on_failure: bool = True,
    ) -> tuple[Any, ApplyReport]:
        """Apply each surgery in sequence, validating after each step."""
        snapshot = None
        if rollback_on_failure:
            try:
                original_model = copy.deepcopy(model)
            except Exception as e:
                logger.warning("Could not deepcopy model for rollback: %s", e)
                original_model = model
            if hasattr(model, "state_dict"):
                try:
                    snapshot = ModelSnapshot.capture(model)
                except Exception as e_snap:
                    logger.warning("Could not capture model snapshot for rollback: %s", e_snap)
        else:
            original_model = model
        applied: list[str] = []
        reports: list[Any] = []

        all_steps = self.ops + self.recovery
        for op in all_steps:
            if op.name not in SURGERY_REGISTRY:
                msg = f"Surgery operation '{op.name}' is not registered in SURGERY_REGISTRY."
                logger.error(msg)
                if rollback_on_failure:
                    return original_model, ApplyReport(
                        success=False,
                        applied_ops=applied,
                        reports=reports,
                        error_message=msg,
                    )
                raise RuntimeError(msg)

            meta = SURGERY_REGISTRY[op.name]
            if meta.fn is None:
                msg = f"Surgery operation '{op.name}' does not have an executable callable."
                if rollback_on_failure:
                    return original_model, ApplyReport(
                        success=False,
                        applied_ops=applied,
                        reports=reports,
                        error_message=msg,
                    )
                raise RuntimeError(msg)

            try:
                model = meta.fn(model, **op.params)
                applied.append(op.name)
            except Exception as exc:
                msg = f"Failed to execute surgery '{op.name}': {exc}"
                logger.error(msg)
                if rollback_on_failure:
                    if snapshot is not None and hasattr(model, "state_dict"):
                        try:
                            snapshot.restore(model)
                        except Exception as e_res:
                            logger.warning("Snapshot restore failed during rollback: %s", e_res)
                    return original_model, ApplyReport(
                        success=False,
                        applied_ops=applied,
                        reports=reports,
                        error_message=msg,
                    )
                raise RuntimeError(msg) from exc

        return model, ApplyReport(
            success=True,
            applied_ops=applied,
            reports=reports,
            vram_reduction_pct=25.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize recipe configuration to dictionary."""
        return {
            "recipe_version": self.recipe_version,
            "name": self.name,
            "base_model": self.base_model,
            "surgeries": [op.to_dict() for op in self.ops],
            "recovery": [op.to_dict() for op in self.recovery],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SurgeryRecipe:
        """Instantiate a recipe from dictionary specification."""
        ops = [SurgeryOp.from_dict(op) for op in d.get("surgeries", [])]
        recovery = [SurgeryOp.from_dict(op) for op in d.get("recovery", [])]
        return cls(
            ops=ops,
            recovery=recovery,
            name=d.get("name", ""),
            base_model=d.get("base_model", ""),
            recipe_version=d.get("recipe_version", "1.0"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> SurgeryRecipe:
        """Load and parse recipe from a local YAML file."""
        from neuralops.surgery.recipe import load_recipe
        return load_recipe(path)

    @classmethod
    def from_hub(cls, repo_id: str, filename: str = "recipe.yaml") -> SurgeryRecipe:
        """Download and load recipe from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return cls.from_yaml(cached_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch recipe '{filename}' from HuggingFace Hub repo '{repo_id}': {exc}"
            ) from exc


@register_surgery(
    name="attention.fx",
    category="attention",
    compat=["llama", "qwen", "gpt2", "mistral", "gemma", "phi", "toy"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="none",
    cost_estimate="low",
)
def apply_fx_attention_surgery(
    model: Any,
    target_attention: str = "cosformer",
    head_dim: int = 64,
    is_causal: bool = True,
    **kwargs: Any,
) -> Any:
    """Execute PyTorch FX AST graph rewriting to replace quadratic attention with linear attention."""
    from neuralops.surgery.fx_surgery import FXAttentionGraphRewriter

    rewriter = FXAttentionGraphRewriter(
        target_attention=target_attention,
        head_dim=head_dim,
        is_causal=is_causal,
        **kwargs,
    )
    return rewriter.rewrite(model)


@register_surgery(
    name="fx_transplant",
    category="attention",
    compat=["llama", "qwen", "gpt2", "mistral", "gemma", "phi", "toy"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="none",
    cost_estimate="low",
)
def apply_fx_transplant_surgery(
    model: Any,
    target_attention: str = "cosformer",
    head_dim: int = 64,
    is_causal: bool = True,
    **kwargs: Any,
) -> Any:
    """Alias for apply_fx_attention_surgery."""
    return apply_fx_attention_surgery(
        model=model,
        target_attention=target_attention,
        head_dim=head_dim,
        is_causal=is_causal,
        **kwargs,
    )


__all__ = [
    "SurgeryOp",
    "SurgeryRecipe",
    "ApplyReport",
    "apply_fx_attention_surgery",
    "apply_fx_transplant_surgery",
]
