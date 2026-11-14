#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Master Transformer Composite Pipeline (Surgery -> SVD -> Prune -> Quant -> Export)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn

from neuralops.compression.pruning.wanda import WandaCompressor
from neuralops.compression.quantization.awq import AWQCompressor
from neuralops.compression.quantization.fp8_compressor import FP8Compressor
from neuralops.compression.svd.direct_svd import DirectSVDCompressor
from neuralops.export.safetensors_exporter import SafeTensorsExporter

logger = logging.getLogger(__name__)


class MasterTransformerPipeline:
    """End-to-End Composite Industrial Transformation Pipeline.

    Sequentially executes:
      1. Attention Surgery (Transplanting target attention architecture)
      2. SVD Factorization (Low-rank linear decomposition)
      3. Structured/Unstructured Pruning (Magnitude and activation-aware pruning)
      4. Quantization (W8A8/FP8 or INT4 weight quantization)
      5. Serving Export (SafeTensors serialization and manifest creation)
    """

    def __init__(self, target_architecture: str = "cosformer", quant_bits: int = 8) -> None:
        """Initialize composite pipeline with target architecture and quantization bits.

        Args:
            target_architecture: Attention mechanism to graft into the model.
            quant_bits: Quantization bit-width (8 for FP8/W8A8, 4 for INT4 AWQ).
        """
        self.target_architecture = target_architecture
        self.quant_bits = quant_bits
        self.pipeline_stages: List[str] = [
            "attention_surgery",
            "svd_factorization",
            "structured_pruning",
            "quantization_w8a8",
            "serving_export",
        ]

    def execute_stage_surgery(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        inherit_weights: bool = True,
        engine: str = "shell",
        strict: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """Stage 1: Surgical Attention Transplant via UniversalSurgeryDirector.

        Args:
            model: PyTorch module undergoing attention surgery.
            layer_indices: Optional specific block indices to graft.
            inherit_weights: Whether to inherit projections from base attention.
            engine: Execution engine ("shell", "torch", "inductor").
            strict: If True, re-raise exceptions; otherwise log warning.
            **kwargs: Extra arguments forwarded to surgery director.

        Returns:
            nn.Module with transplanted attention mechanism.
        """
        logger.info("Executing Stage 1: Surgical Attention Transplant -> %s", self.target_architecture)
        try:
            from neuralops.surgery.director import UniversalSurgeryDirector

            director = UniversalSurgeryDirector(model)
            model = director.execute_surgery(
                target_attention=self.target_architecture,
                layer_indices=layer_indices,
                inherit_weights=inherit_weights,
                engine=engine,
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "UniversalSurgeryDirector execution fell back with exception: %s.",
                exc,
            )
            if strict:
                raise
        return model

    def execute_stage_svd(
        self,
        model: nn.Module,
        rank_ratio: float = 0.5,
        **kwargs: Any,
    ) -> nn.Module:
        """Stage 2: SVD Low-Rank Compression.

        Args:
            model: PyTorch module to compress.
            rank_ratio: Target rank ratio (0.0 to 1.0) for linear decompositions.
            **kwargs: Additional parameters passed to DirectSVDCompressor.

        Returns:
            Compressed nn.Module with factorized linear layers.
        """
        logger.info("Executing Stage 2: SVD Low-Rank Compression with ratio %.2f", rank_ratio)
<<<<<<< HEAD
        raise NotImplementedError("execute_stage_svd is not yet implemented.")
=======
        compressor = DirectSVDCompressor(
            model,
            rank_ratio=rank_ratio,
            skip_attention_qkv=kwargs.pop("skip_attention_qkv", False),
            **kwargs,
        )
        compressed_model, stats = compressor.compress()
        logger.info("Stage 2 SVD compression completed: %s", stats)
        return compressed_model
>>>>>>> fix/benchmarks-remediation

    def execute_stage_prune(
        self,
        model: nn.Module,
        sparsity: float = 0.25,
        **kwargs: Any,
    ) -> nn.Module:
        """Stage 3: Structured / Activation-Aware Sparsity & Pruning.

        Args:
            model: PyTorch module to prune.
            sparsity: Target sparsity fraction (0.0 to 1.0).
            **kwargs: Additional parameters passed to WandaCompressor.

        Returns:
            Pruned nn.Module.
        """
        logger.info("Executing Stage 3: Structured Pruning with sparsity %.2f", sparsity)
<<<<<<< HEAD
        raise NotImplementedError("execute_stage_prune is not yet implemented.")
=======
        compressor = WandaCompressor(model, sparsity=sparsity, **kwargs)
        pruned_model, stats = compressor.compress()
        logger.info("Stage 3 pruning completed: %s", stats)
        return pruned_model
>>>>>>> fix/benchmarks-remediation

    def execute_stage_quantize(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        """Stage 4: Quantization (FP8/W8A8 or INT4).

        Args:
            model: PyTorch module to quantize.
            **kwargs: Additional parameters for quantization compressor.

        Returns:
            Quantized nn.Module.

        Raises:
            NotImplementedError: If quant_bits is not supported (only 8 and 4 supported).
        """
        logger.info("Executing Stage 4: Quantization to %d bits", self.quant_bits)
<<<<<<< HEAD
        raise NotImplementedError("execute_stage_quantize is not yet implemented.")
=======
        if self.quant_bits == 8:
            scale_type = kwargs.pop("scale_type", "per_tensor")
            fp8_format = kwargs.pop("fp8_format", "e4m3")
            compressor = FP8Compressor(model, scale_type=scale_type, fp8_format=fp8_format, **kwargs)
            quantized_model, stats = compressor.compress()
            logger.info("Stage 4 FP8 quantization completed: %s", stats)
            return quantized_model
        elif self.quant_bits == 4:
            group_size = kwargs.pop("group_size", 128)
            awq_compressor = AWQCompressor(model, bits=4, group_size=group_size, **kwargs)
            quantized_model, stats = awq_compressor.compress()
            logger.info("Stage 4 INT4 AWQ quantization completed: %s", stats)
            return quantized_model
        else:
            raise NotImplementedError(
                f"Quantization to {self.quant_bits} bits is not supported. Supported bit widths: 4, 8."
            )
>>>>>>> fix/benchmarks-remediation

    def execute_stage_export(
        self,
        model: nn.Module,
        output_dir: str = "results/artifacts",
        filename_prefix: str = "master_transformed_model",
    ) -> Dict[str, str]:
        """Stage 5: Serving Export to SafeTensors and Manifest.

        Args:
            model: Transformed model to serialize.
            output_dir: Directory where artifacts are saved.
            filename_prefix: Base filename for weights and manifest.

        Returns:
            Dict mapping artifact types to filesystem paths.
        """
        logger.info("Executing Stage 5: Exporting to %s", output_dir)
        exporter = SafeTensorsExporter(output_dir=output_dir)
        return exporter.export_model(model, filename_prefix=filename_prefix)

    def run_full_pipeline(
        self,
        model: nn.Module,
        output_dir: str = "results/artifacts",
    ) -> Dict[str, Any]:
        """Run all 5 transformation stages sequentially.

        Args:
            model: Starting PyTorch model.
            output_dir: Directory where final exported artifacts are stored.

        Returns:
            Dictionary containing final model, exported artifacts, and execution status.
        """
        m = self.execute_stage_surgery(model)
        m = self.execute_stage_svd(m)
        m = self.execute_stage_prune(m)
        m = self.execute_stage_quantize(m)
        export_artifacts = self.execute_stage_export(m, output_dir=output_dir)
        return {
            "model": m,
            "artifacts": export_artifacts,
            "status": "SUCCESS",
        }


__all__ = ["MasterTransformerPipeline"]
