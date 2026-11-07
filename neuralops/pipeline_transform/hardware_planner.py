#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware-Aware Resource Planning and VRAM/Latency Estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from neuralops.pipeline_transform.plan import ExecutionPlan
from neuralops.recipe.schema import Recipe

logger = logging.getLogger("NeuralOps.PipelineTransform.HardwarePlanner")


@dataclass
class HardwareProfile:
    """Hardware capability and accelerator performance profile."""

    name: str
    vram_gb: float
    compute_capability: str
    memory_bandwidth_gbps: float
    flops_fp16_tflops: float
    device_type: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vram_gb": self.vram_gb,
            "compute_capability": self.compute_capability,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "flops_fp16_tflops": self.flops_fp16_tflops,
            "device_type": self.device_type,
        }


KNOWN_HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "rtx4090": HardwareProfile(
        name="NVIDIA GeForce RTX 4090",
        vram_gb=24.0,
        compute_capability="8.9",
        memory_bandwidth_gbps=1008.0,
        flops_fp16_tflops=165.2,
    ),
    "a100": HardwareProfile(
        name="NVIDIA A100-SXM4-80GB",
        vram_gb=80.0,
        compute_capability="8.0",
        memory_bandwidth_gbps=2039.0,
        flops_fp16_tflops=312.0,
    ),
    "h100": HardwareProfile(
        name="NVIDIA H100 80GB HBM3",
        vram_gb=80.0,
        compute_capability="9.0",
        memory_bandwidth_gbps=3350.0,
        flops_fp16_tflops=989.0,
    ),
    "t4": HardwareProfile(
        name="NVIDIA Tesla T4",
        vram_gb=16.0,
        compute_capability="7.5",
        memory_bandwidth_gbps=320.0,
        flops_fp16_tflops=65.0,
    ),
    "cpu": HardwareProfile(
        name="Host CPU",
        vram_gb=16.0,
        compute_capability="0.0",
        memory_bandwidth_gbps=50.0,
        flops_fp16_tflops=2.0,
        device_type="cpu",
    ),
}


@dataclass
class ResourceEstimate:
    """Hardware resource requirements and parameter impact estimate."""

    peak_vram_mb: float
    total_time_seconds: float
    parameters_before: int
    parameters_after: int
    parameter_delta_pct: float
    flops_delta_pct: float
    estimated_svd_time_seconds: float
    likely_needs_finetuning: bool
    confidence: str = "medium"  # "high", "medium", "low"
    warnings: List[str] = field(default_factory=list)
    hardware_profile: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_vram_mb": self.peak_vram_mb,
            "total_time_seconds": self.total_time_seconds,
            "parameters_before": self.parameters_before,
            "parameters_after": self.parameters_after,
            "parameter_delta_pct": self.parameter_delta_pct,
            "flops_delta_pct": self.flops_delta_pct,
            "estimated_svd_time_seconds": self.estimated_svd_time_seconds,
            "likely_needs_finetuning": self.likely_needs_finetuning,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "hardware_profile": self.hardware_profile,
        }


def _resolve_hardware_profile(
    profile_arg: Optional[Union[str, HardwareProfile, Dict[str, Any]]],
) -> Tuple[HardwareProfile, str, List[str]]:
    warnings: List[str] = []
    confidence = "medium"

    if isinstance(profile_arg, HardwareProfile):
        return profile_arg, "high", warnings

    if isinstance(profile_arg, dict):
        hp = HardwareProfile(
            name=profile_arg.get("name", "Custom GPU"),
            vram_gb=float(profile_arg.get("vram_gb", 16.0)),
            compute_capability=str(profile_arg.get("compute_capability", "8.0")),
            memory_bandwidth_gbps=float(profile_arg.get("memory_bandwidth_gbps", 500.0)),
            flops_fp16_tflops=float(profile_arg.get("flops_fp16_tflops", 100.0)),
            device_type=profile_arg.get("device_type", "cuda"),
        )
        return hp, "medium", warnings

    if isinstance(profile_arg, str):
        key = profile_arg.lower().replace(" ", "").replace("-", "").replace("_", "")
        for k, p in KNOWN_HARDWARE_PROFILES.items():
            if k in key or key in k:
                return p, "high", warnings

        # Unknown profile string
        warnings.append(
            f"Unknown hardware profile '{profile_arg}'; falling back to conservative defaults (confidence: low)."
        )
        fallback = HardwareProfile(
            name=f"Unknown GPU ({profile_arg})",
            vram_gb=16.0,
            compute_capability="7.0",
            memory_bandwidth_gbps=400.0,
            flops_fp16_tflops=50.0,
        )
        return fallback, "low", warnings

    # Auto-detect or default
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_key = gpu_name.lower().replace(" ", "").replace("-", "")
        for k, p in KNOWN_HARDWARE_PROFILES.items():
            if k in gpu_key:
                return p, "high", warnings
        total_vram = float(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        detected = HardwareProfile(
            name=gpu_name,
            vram_gb=total_vram,
            compute_capability="8.0",
            memory_bandwidth_gbps=800.0,
            flops_fp16_tflops=100.0,
        )
        return detected, "medium", warnings

    # Host is CPU only
    warnings.append("No CUDA GPU detected; resource estimates assume target CUDA hardware.")
    return KNOWN_HARDWARE_PROFILES["rtx4090"], "medium", warnings


def estimate_resources(
    plan_or_recipe: Union[ExecutionPlan, Recipe, Dict[str, Any]],
    hardware_profile: Optional[Union[str, HardwareProfile, Dict[str, Any]]] = None,
    base_parameters: Optional[int] = None,
) -> ResourceEstimate:
    """Estimate peak VRAM, transformation latency, FLOP delta, and parameter count."""
    hw_prof, confidence, warnings = _resolve_hardware_profile(hardware_profile)

    # Extract stages
    stages = []
    if isinstance(plan_or_recipe, ExecutionPlan):
        stages = plan_or_recipe.stages
    elif isinstance(plan_or_recipe, Recipe):
        stages = plan_or_recipe.operations
    elif isinstance(plan_or_recipe, dict):
        stages = plan_or_recipe.get("stages", plan_or_recipe.get("operations", []))

    # Analytical model parameter count determination
    if base_parameters is not None:
        params_before = int(base_parameters)
    else:
        model_src = ""
        if isinstance(plan_or_recipe, Recipe) and plan_or_recipe.model:
            model_src = plan_or_recipe.model.source or ""
        elif isinstance(plan_or_recipe, dict):
            model_src = plan_or_recipe.get("model", {}).get("source", "")
        elif (
            isinstance(plan_or_recipe, ExecutionPlan)
            and hasattr(plan_or_recipe, "recipe")
            and plan_or_recipe.recipe
            and plan_or_recipe.recipe.model
        ):
            model_src = plan_or_recipe.recipe.model.source or ""

        src_lower = model_src.lower()
        if "70b" in src_lower:
            params_before = 70_000_000_000
        elif "13b" in src_lower:
            params_before = 13_000_000_000
        elif "8b" in src_lower:
            params_before = 8_000_000_000
        elif "7b" in src_lower:
            params_before = 7_000_000_000
        elif "5b" in src_lower:
            params_before = 5_000_000_000
        elif "3b" in src_lower:
            params_before = 3_000_000_000
        elif "1b" in src_lower:
            params_before = 1_000_000_000
        elif "distilbert" in src_lower:
            params_before = 66_000_000
        elif "gpt2" in src_lower:
            params_before = 124_000_000
        else:
            params_before = 125_000_000

    params_after = params_before
    estimated_svd_time = 0.0
    total_time = 0.0
    likely_needs_finetuning = False

    # Baseline VRAM model:
    # Model weights (FP16: 2 bytes/param) + Optimizer/Activations/Peak scratch space
    base_model_vram_mb = (params_before * 2.0) / (1024 * 1024)
    peak_vram_mb = base_model_vram_mb * 1.15

    for s in stages:
        op_type = getattr(s, "operation_type", getattr(s, "type", "unknown"))
        cfg = getattr(s, "config", {}) if hasattr(s, "config") else s.get("config", {})

        if op_type == "weights.compress":
            target_ratio = float(cfg.get("target_ratio", 0.5))
            # SVD reduces linear parameters
            compressed_portion = params_before * 0.65  # linear layers typically ~65% of LLM
            delta_params = int(compressed_portion * (1.0 - target_ratio))
            params_after -= delta_params

            # SVD runtime: proportional to matrix size and flops
            bandwidth_factor = 1000.0 / max(hw_prof.memory_bandwidth_gbps, 1.0)
            compute_factor = 100.0 / max(hw_prof.flops_fp16_tflops, 1.0)
            stage_svd_sec = (params_before / 100_000_000.0) * (2.5 * bandwidth_factor + 1.5 * compute_factor)
            estimated_svd_time += stage_svd_sec
            total_time += stage_svd_sec

            # Peak scratch VRAM during SVD decomposition
            svd_scratch_mb = base_model_vram_mb * (0.1 + 0.8 * target_ratio)
            peak_vram_mb = max(peak_vram_mb, base_model_vram_mb + svd_scratch_mb)

            if target_ratio < 0.6:
                likely_needs_finetuning = True

        elif op_type == "attention.replace":
            total_time += 1.5
            peak_vram_mb = max(peak_vram_mb, base_model_vram_mb * 1.3)
            likely_needs_finetuning = True

        else:
            total_time += 1.0

    param_delta_pct = ((params_after - params_before) / max(1, params_before)) * 100.0
    flops_delta_pct = param_delta_pct * 0.95  # FLOPs track parameter reductions in transformer layers

    # Check against hardware profile VRAM limit
    hw_vram_mb = hw_prof.vram_gb * 1024.0
    if peak_vram_mb > hw_vram_mb:
        warnings.append(
            f"Estimated peak VRAM ({peak_vram_mb:.0f} MB) exceeds available device VRAM ({hw_vram_mb:.0f} MB) on {hw_prof.name}."
        )

    return ResourceEstimate(
        peak_vram_mb=round(peak_vram_mb, 2),
        total_time_seconds=round(max(total_time, 0.1), 2),
        parameters_before=params_before,
        parameters_after=params_after,
        parameter_delta_pct=round(param_delta_pct, 2),
        flops_delta_pct=round(flops_delta_pct, 2),
        estimated_svd_time_seconds=round(estimated_svd_time, 2),
        likely_needs_finetuning=likely_needs_finetuning,
        confidence=confidence,
        warnings=warnings,
        hardware_profile=hw_prof.to_dict(),
    )


__all__ = [
    "HardwareProfile",
    "KNOWN_HARDWARE_PROFILES",
    "ResourceEstimate",
    "estimate_resources",
]
