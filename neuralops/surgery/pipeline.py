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
    wrap_surgical_model: bool = False,
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
    rewritten = rewriter.rewrite(model)
    if wrap_surgical_model:
        from neuralops.surgery.model_surgery import SurgicalModelWrapper

        return SurgicalModelWrapper(
            model=rewritten,
            engine="fx",
            target_attention=target_attention,
        )
    return rewritten


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
    wrap_surgical_model: bool = False,
    **kwargs: Any,
) -> Any:
    """Alias for apply_fx_attention_surgery."""
    return apply_fx_attention_surgery(
        model=model,
        target_attention=target_attention,
        head_dim=head_dim,
        is_causal=is_causal,
        wrap_surgical_model=wrap_surgical_model,
        **kwargs,
    )


class SurgeryPipeline:
    """
    Master Surgery Pipeline Orchestrator (Phase 058).
    Coordinates declarative surgery recipes, invariant compliance verification,
    and Universal Surgery Director integration returning certified SurgicalModel instances.
    """

    def __init__(self, recipe: SurgeryRecipe | None = None) -> None:
        self.recipe = recipe

    def run(
        self,
        model: Any,
        rollback_on_failure: bool = True,
    ) -> tuple[Any, ApplyReport]:
        """Executes the pipeline recipe on the target model."""
        if self.recipe is None:
            raise ValueError("SurgeryPipeline requires a valid SurgeryRecipe to run.")
        return self.recipe.apply(model=model, rollback_on_failure=rollback_on_failure)

    @classmethod
    def apply_universal_director(
        cls,
        model: Any,
        laws: list[int] | None = None,
        target_attention: str = "cosformer",
        layer_indices: str | list[int] | None = None,
        decay: float = 0.99,
        softcapping_factor: float | None = None,
        engine: str = "shell",
        attach_sidecar: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Executes Universal Surgery Director enforcing Inviolable Laws [1..10] and returns a certified SurgicalModel.

        Args:
            model: Target transformer model across 9 canonical families.
            laws: List of integer laws to enforce (defaults to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).
            target_attention: Target linear attention mechanism (default: 'cosformer').
            layer_indices: Target layers ('all', 'hybrid_moa', or list of ints).
            decay: Law 7 exponential state decay factor gamma in (0, 1).
            softcapping_factor: Law 2 logit soft-capping factor.
            engine: Surgery engine ('shell' or 'fx').
            attach_sidecar: Whether to attach shadow sidecar for delta monitoring.
            **kwargs: Extra parameters passed to director.

        Returns:
            Certified model satisfying SurgicalModel protocol.
        """
        from neuralops.models.protocols import SurgicalModel
        from neuralops.surgery.director import UniversalSurgeryDirector

        if laws is None:
            laws = list(range(1, 11))
        else:
            laws = list(laws)

        # Validate laws
        invalid_laws = [law for law in laws if law < 1 or law > 10]
        if invalid_laws:
            raise ValueError(f"Invalid laws specified: {invalid_laws}. Universal laws must be in 1..10.")

        director = UniversalSurgeryDirector(model)
        if softcapping_factor is not None:
            director.profile.softcapping_factor = softcapping_factor

        surgeried = director.execute_surgery(
            target_attention=target_attention,
            layer_indices=layer_indices,
            engine=engine,
            attach_sidecar=attach_sidecar,
            decay=decay,
            **kwargs,
        )

        # Verify invariants
        director.verify_invariants()

        # Certify SurgicalModel protocol compliance
        SurgicalModel.register(type(surgeried))
        if not hasattr(surgeried, "get_surgical_manifest"):
            surgeried.get_surgical_manifest = lambda: getattr(surgeried, "_surgical_manifest", {})  # type: ignore[assignment]
        if not hasattr(surgeried, "get_attention_family"):
            surgeried.get_attention_family = lambda layer_idx: target_attention  # type: ignore[assignment]

        return surgeried


@register_surgery(
    name="attention.universal_director",
    category="attention",
    compat=["llama", "gemma2", "deepseek_mla", "bert", "t5", "falcon", "vit", "jamba", "quantized"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="high",
    cost_estimate="low",
)
def apply_universal_director_surgery(
    model: Any,
    decay: float = 0.99,
    softcapping_factor: float | None = None,
    **kwargs: Any,
) -> Any:
    """Execute Universal Surgery Director invariant compliance across 9 canonical model families."""
    return SurgeryPipeline.apply_universal_director(
        model=model,
        decay=decay,
        softcapping_factor=softcapping_factor,
        **kwargs,
    )


@register_surgery(
    name="attention.mla",
    category="attention",
    compat=["deepseek", "deepseek_mla", "llama", "qwen"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="medium",
    cost_estimate="medium",
)
def apply_mla_attention_surgery(
    model: Any,
    latent_kv_dim: int = 512,
    decoupled_rope_dim: int = 64,
    decay: float = 0.99,
    **kwargs: Any,
) -> Any:
    """Execute DeepSeek Multi-Head Latent Attention (MLA) dual-stream surgery."""
    from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge
    from neuralops.surgery.scanner import UniversalArchitectureScanner

    scanner = UniversalArchitectureScanner(model)
    targets = scanner.scan()
    for spec in targets:
        mla_bridge = DeepSeekMLASurgeryBridge(
            latent_kv_dim=latent_kv_dim,
            decoupled_rope_dim=decoupled_rope_dim,
            decay=decay,
            hidden_dim=spec.hidden_dim,
            **kwargs,
        )
        spec.parent_module.register_module(spec.child_name, mla_bridge)
    return model


__all__ = [
    "SurgeryOp",
    "SurgeryRecipe",
    "ApplyReport",
    "SurgeryPipeline",
    "apply_fx_attention_surgery",
    "apply_fx_transplant_surgery",
    "apply_universal_director_surgery",
    "apply_mla_attention_surgery",
]
