#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
ModelSurgery — Fluent API for Transactional Neural Network Transformation.

Specification Section 8: ModelSurgery API.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from neuralops.executor.executor import execute_recipe
from neuralops.executor.snapshot import ModelSnapshot
from neuralops.operations.registry import get_operation
from neuralops.recipe.loader import load_recipe, loads_recipe
from neuralops.recipe.schema import Recipe
from neuralops.validation.result import ValidationResult
from neuralops.validation.runner import run_validation_gates

logger = logging.getLogger("NeuralOps.Surgery")


class ModelSurgery:
    """Fluent transactional orchestrator for model transformations and recipes."""

    def __init__(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        recipe: Optional[Recipe] = None,
    ) -> None:
        self.model: nn.Module = model
        try:
            self.baseline_model: Optional[nn.Module] = (
                baseline_model if baseline_model is not None else copy.deepcopy(model).cpu()
            )
        except Exception as e:
            logger.warning("Could not deepcopy model for baseline_model: %s", e)
            self.baseline_model = baseline_model

        self.recipe: Optional[Recipe] = recipe
        self._initial_snapshot: ModelSnapshot = ModelSnapshot.capture(model)
        self._history: List[ModelSnapshot] = [self._initial_snapshot]
        self._evidence_records: List[Dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device_map: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelSurgery:
        """Load model via HuggingFace transformers or torch and wrap in ModelSurgery."""
        logger.info("Loading model from pretrained '%s'...", model_name_or_path)
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=device_map,
                **kwargs,
            )
        except Exception as e:
            logger.warning("AutoModel.from_pretrained failed (%s); trying torch.load fallback", e)
            model = torch.load(model_name_or_path, map_location="cpu", weights_only=False)

        return cls(model=model)

    @classmethod
    def from_recipe(
        cls,
        recipe_or_path: Union[str, Path, Recipe, Dict[str, Any]],
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> ModelSurgery:
        """Instantiate ModelSurgery configured by a declarative transformation recipe."""
        if isinstance(recipe_or_path, Recipe):
            recipe = recipe_or_path
        elif isinstance(recipe_or_path, (str, Path)):
            p = str(recipe_or_path)
            if p.strip().startswith(("{", "---", "apiVersion", "version")):
                recipe = loads_recipe(p)
            else:
                recipe = load_recipe(p)
        elif isinstance(recipe_or_path, dict):
            recipe = Recipe.from_dict(recipe_or_path)
        else:
            raise TypeError(f"Invalid recipe type: {type(recipe_or_path).__name__}")

        if model is None:
            # Attempt to load target model declared in recipe
            source = recipe.model.source
            if not source:
                raise ValueError("Recipe does not specify model.source; please pass 'model' explicitly")
            return cls.from_pretrained(source, recipe=recipe, **kwargs)

        return cls(model=model, recipe=recipe)

    def apply(
        self,
        operation_or_recipe: Optional[Union[str, Recipe]] = None,
        version: str = "1.0.0",
        layer_parallel: bool = False,
        **config: Any,
    ) -> ModelSurgery:
        """Apply an operation or execute a full recipe in place."""
        if layer_parallel:
            from neuralops.distributed.context import DistributedContext
            from neuralops.executor.layer_parallel import LayerParallelExecutor

            ctx = DistributedContext()
            if ctx.is_distributed() or ctx.available_gpus > 1:
                lexec = LayerParallelExecutor(ctx)
                if isinstance(operation_or_recipe, str):
                    op_lower = operation_or_recipe.lower()
                    if "attention" in op_lower or "transplant" in op_lower:
                        target_att = config.get("target_attention", config.get("attention", "full_attention"))
                        self.model = lexec.transplant(self.model, target_attention=target_att, **config)
                        return self
                    elif "compress" in op_lower or "svd" in op_lower:
                        method = config.get("method", "svd")
                        target_ratio = config.get("target_ratio", config.get("ratio", 0.5))
                        self.model = lexec.compress(self.model, method=method, target_ratio=target_ratio, **config)
                        return self
        if operation_or_recipe is None and self.recipe is not None:
            # Execute attached recipe
            pipeline_result = execute_recipe(
                recipe=self.recipe,
                model=self.model,
                baseline_model=self.baseline_model,
            )
            self.model = pipeline_result.model
            for stage in pipeline_result.stage_records:
                if stage.evidence:
                    self._evidence_records.append(stage.evidence)
            if not pipeline_result.success:
                raise RuntimeError(f"Recipe execution failed: {pipeline_result.error}")
            return self

        if isinstance(operation_or_recipe, Recipe):
            pipeline_result = execute_recipe(
                recipe=operation_or_recipe,
                model=self.model,
                baseline_model=self.baseline_model,
            )
            self.model = pipeline_result.model
            for stage in pipeline_result.stage_records:
                if stage.evidence:
                    self._evidence_records.append(stage.evidence)
            if not pipeline_result.success:
                raise RuntimeError(f"Recipe execution failed: {pipeline_result.error}")
            return self

        if isinstance(operation_or_recipe, str):
            op_name = operation_or_recipe
            # Support inline param syntax e.g. "weights.compress[method=aasvd]"
            if "[" in op_name and op_name.endswith("]"):
                base_name, params_str = op_name[:-1].split("[", 1)
                op_name = base_name.strip()
                for pair in params_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        config[k.strip()] = v.strip()

            if op_name in ("attention.fx", "fx_transplant", "attention.rewrite_fx") or config.get("engine") == "fx":
                target_att = config.get("target_attention", config.get("target", "cosformer"))
                return self.replace_attention(target_attention=target_att, engine="fx", **config)

            # Capture snapshot before mutation for transactional rollback
            snapshot_before = ModelSnapshot.capture(self.model)
            self._history.append(snapshot_before)
            op_impl = get_operation(op_name, version=version)
            try:
                self.model, evidence = op_impl.apply(self.model, config)
            except Exception as exc:
                logger.error(
                    "Surgery operation '%s' failed mid-transplant: %s. Rolling back to snapshot.",
                    op_name,
                    exc,
                )
                self.model = snapshot_before.restore(self.model)
                if self._history and self._history[-1] is snapshot_before:
                    self._history.pop()
                raise
            if evidence:
                self._evidence_records.append(evidence)
            return self

        raise ValueError("Must provide operation name or Recipe to apply()")

    def replace_attention(
        self,
        target_attention: str = "cosformer",
        layer_indices: Optional[List[int]] = None,
        engine: str = "shell",
        **config: Any,
    ) -> ModelSurgery:
        """Replace quadratic attention mechanisms using Python module mutation ('shell')

        or PyTorch FX AST graph rewriting ('fx').
        """
        if engine == "fx":
            from neuralops.surgery.fx_surgery import FXAttentionGraphRewriter
            rewriter = FXAttentionGraphRewriter(
                target_attention=target_attention,
                head_dim=config.get("head_dim", 64),
                is_causal=config.get("is_causal", True),
                leaf_modules=config.get("leaf_modules", None),
            )
            self.model = rewriter.rewrite(self.model, concrete_args=config.get("concrete_args", None))
            return self
        elif engine == "shell":
            return self.apply(
                "attention.transplant",
                target_attention=target_attention,
                layer_indices=layer_indices,
                **config,
            )
        else:
            raise ValueError(f"Unknown surgery engine '{engine}'. Choose 'shell' or 'fx'.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invokes the underlying model's forward pass."""
        return self.model(*args, **kwargs)

    def compile(
        self,
        mode: str = "max-autotune",
        fullgraph: bool = True,
        backend: str = "inductor",
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Compiles the surgical model with PyTorch Inductor for maximum fusion and throughput."""
        logger.info(
            "Compiling model via torch.compile(mode='%s', fullgraph=%s, backend='%s')...",
            mode,
            fullgraph,
            backend,
        )
        self.model = torch.compile(
            self.model,
            mode=mode,
            fullgraph=fullgraph,
            backend=backend,
            options=options,
            **kwargs,
        )
        return self.model

    def validate(
        self,
        profile: str = "standard",
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run validation gates against the transformed model."""
        return run_validation_gates(
            model=self.model,
            baseline_model=self.baseline_model,
            profile=profile,
            context=context,
        )

    def commit(self) -> ModelSurgery:
        """Commit current transformation state as the new baseline."""
        try:
            self.baseline_model = copy.deepcopy(self.model).cpu()
        except Exception as e:
            logger.warning("Could not deepcopy model for baseline on commit: %s", e)
            self.baseline_model = None
        self._initial_snapshot = ModelSnapshot.capture(self.model)
        self._history = [self._initial_snapshot]
        return self

    def rollback(self) -> ModelSurgery:
        """Rollback model to the initial baseline snapshot."""
        logger.info("Rolling back ModelSurgery to initial baseline snapshot...")
        self.model = self._initial_snapshot.restore(self.model)
        self._history = [self._initial_snapshot]
        return self

    def get_evidence(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded transformation evidence records."""
        return list(self._evidence_records)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save transformed model weights and configuration."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_dir)
        else:
            torch.save(self.model.state_dict(), out_dir / "pytorch_model.bin")
        logger.info("Saved transformed model to %s", out_dir)


__all__ = ["ModelSurgery"]
