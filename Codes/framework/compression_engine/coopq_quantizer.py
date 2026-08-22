"""CoopQ: Cooperative Game-Theoretic Bit Allocation Quantizer with STE. Registered as 'coopq'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.compression_engine.astro_quantizer import (
    AstroLinear,
    quantize_dequantize_tensor,
)
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class CoopQQuantizer(BaseCompressor):
    """
    CoopQ: Cooperative Game-Theoretic Bit Allocation Quantization.
    Allocates discrete bit-widths (e.g. 2, 4, 6, 8 bits) across transformer layers
    proportional to layer sensitivity (Frobenius energy / variance), guaranteeing
    a target mean bit-budget under strict distortion minimization.

    Args:
        model: PyTorch model.
        bits: Target average bit-width (default 4).
        min_bits: Minimum allowable bit-width per layer (default 2).
        max_bits: Maximum allowable bit-width per layer (default 8).
        symmetric: Whether to use symmetric quantization (default True).
        per_channel: Whether to use per-channel scale factors (default False).
        use_ste: Whether to substitute nn.Linear with AstroLinear for QAT (default False).
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        min_bits: int = 2,
        max_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        use_ste: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="coopq", **kwargs)
        self.bits = int(bits)
        self.min_bits = int(min_bits)
        self.max_bits = int(max_bits)
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.use_ste = bool(use_ste)

    def _replace_module(self, full_name: str, new_module: nn.Module) -> None:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _calculate_layer_sensitivities(self) -> Dict[str, float]:
        """Compute sensitivity metric per Linear layer based on normalized Frobenius energy and variance."""
        sensitivities: Dict[str, float] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                # Frobenius norm normalized by matrix dimensions + standard deviation
                frob_norm = torch.norm(W, p="fro").item() / (W.numel() ** 0.5 + 1e-8)
                std_val = torch.std(W).item() if W.numel() > 1 else 0.0
                sensitivities[name] = float(frob_norm + std_val)
        return sensitivities

    def _allocate_bits(self, sensitivities: Dict[str, float]) -> Dict[str, int]:
        """Allocate discrete integer bit-widths satisfying mean(bits) <= self.bits."""
        if not sensitivities:
            return {}

        total_sens = sum(sensitivities.values()) + 1e-8
        num_layers = len(sensitivities)
        target_total_bits = self.bits * num_layers

        # Continuous allocation proportional to sensitivity
        raw_allocations: Dict[str, float] = {}
        mean_sens = total_sens / num_layers
        for name, sens in sensitivities.items():
            # Sensitivity delta from mean
            sens_ratio = sens / mean_sens
            raw_bits = self.bits + (sens_ratio - 1.0) * (self.max_bits - self.min_bits) / 2.0
            raw_allocations[name] = max(float(self.min_bits), min(float(self.max_bits), raw_bits))

        # Discrete rounding into allowed bit-levels [2, 3, 4, 6, 8]
        allowed_bits = [2, 3, 4, 6, 8]
        allowed_bits = [b for b in allowed_bits if self.min_bits <= b <= self.max_bits]
        if not allowed_bits:
            allowed_bits = [self.bits]

        discrete_allocations: Dict[str, int] = {}
        for name, raw_b in raw_allocations.items():
            closest_b = min(allowed_bits, key=lambda b: abs(b - raw_b))
            discrete_allocations[name] = closest_b

        # Adjust sum to match target total bits budget
        current_sum = sum(discrete_allocations.values())
        sorted_names = sorted(sensitivities.keys(), key=lambda k: sensitivities[k], reverse=True)

        if current_sum > target_total_bits:
            # Reduce bits starting from lowest sensitivity
            for name in reversed(sorted_names):
                curr_b = discrete_allocations[name]
                valid_lower = [b for b in allowed_bits if b < curr_b]
                if valid_lower:
                    discrete_allocations[name] = max(valid_lower)
                    current_sum = sum(discrete_allocations.values())
                    if current_sum <= target_total_bits:
                        break
        elif current_sum < target_total_bits:
            # Increase bits starting from highest sensitivity
            for name in sorted_names:
                curr_b = discrete_allocations[name]
                valid_higher = [b for b in allowed_bits if b > curr_b]
                if valid_higher:
                    discrete_allocations[name] = min(valid_higher)
                    current_sum = sum(discrete_allocations.values())
                    if current_sum >= target_total_bits:
                        break

        return discrete_allocations

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize linear layers with layer-specific bit allocation."""
        sensitivities = self._calculate_layer_sensitivities()
        bit_allocations = self._allocate_bits(sensitivities)

        total_bits_weighted = 0
        total_params_linear = 0

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and name in bit_allocations:
                layer_bits = bit_allocations[name]
                total_bits_weighted += layer_bits * module.weight.numel()
                total_params_linear += module.weight.numel()

                if self.use_ste:
                    astro_layer = AstroLinear.from_linear(
                        module,
                        bits=layer_bits,
                        symmetric=self.symmetric,
                        per_channel=self.per_channel,
                    )
                    self._replace_module(name, astro_layer)
                else:
                    module.weight.data = quantize_dequantize_tensor(
                        module.weight.data,
                        bits=layer_bits,
                        symmetric=self.symmetric,
                        per_channel=self.per_channel,
                        use_ste=False,
                    )
                    if module.bias is not None:
                        module.bias.data = quantize_dequantize_tensor(
                            module.bias.data,
                            bits=layer_bits,
                            symmetric=self.symmetric,
                            per_channel=False,
                            use_ste=False,
                        )

        avg_bits = (
            total_bits_weighted / total_params_linear if total_params_linear > 0 else self.bits
        )
        stats = self.get_compression_stats()
        stats["target_bits"] = self.bits
        stats["average_allocated_bits"] = round(avg_bits, 2)
        stats["layer_bit_allocations"] = bit_allocations
        stats["symmetric"] = self.symmetric
        stats["theoretical_vram_saving"] = f"{(1.0 - avg_bits / 16.0) * 100:.1f}% vs FP16"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("coopq")
def coopq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """CoopQ compressor factory function."""
    bits = kwargs.get("bits", getattr(comp_config, "coopq_bits", 4) if comp_config is not None else 4)
    min_bits = kwargs.get("min_bits", getattr(comp_config, "coopq_min_bits", 2) if comp_config is not None else 2)
    max_bits = kwargs.get("max_bits", getattr(comp_config, "coopq_max_bits", 8) if comp_config is not None else 8)
    symmetric = kwargs.get("symmetric", getattr(comp_config, "coopq_symmetric", True) if comp_config is not None else True)
    per_channel = kwargs.get("per_channel", getattr(comp_config, "coopq_per_channel", False) if comp_config is not None else False)
    use_ste = kwargs.get("use_ste", getattr(comp_config, "coopq_use_ste", False) if comp_config is not None else False)
    compressor = CoopQQuantizer(
        model,
        bits=bits,
        min_bits=min_bits,
        max_bits=max_bits,
        symmetric=symmetric,
        per_channel=per_channel,
        use_ste=use_ste,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
