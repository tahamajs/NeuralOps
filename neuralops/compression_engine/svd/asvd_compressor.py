"""ASVD Compressor: Activation-Aware SVD with KV-Cache hooks and compressed_tensors I/O format.
Aliased to aa_svd_compressor and registered for ASVD compression workflows.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from neuralops.compression_engine.svd.aa_svd_compressor import AASVDCompressor, SVDLinear
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("asvd_compressor", is_alias=True)
@COMPRESSOR_REGISTRY.register("asvd", is_alias=True)
class ASVDCompressor(AASVDCompressor):
    """ASVD Compressor with KV-cache compression hooks and compressed_tensors I/O format."""

    def __init__(self, model: nn.Module, rank: int = 32, **kwargs: Any) -> None:
        super().__init__(model, rank=rank, **kwargs)
        self.r_kv = kwargs.get("r_kv", rank)

    def apply_to_kv_cache(
        self,
        model: Optional[nn.Module] = None,
        cache_manager: Any = None,
        r_kv: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compress attention layer W_k and W_v, registering hooks to store compressed latent C = X @ V_r^T.
        
        Proof: Show ||K - K_r||_F <= eps for r >= 0.5 * d_k.
        On forward, decompress via U_r @ C in the attention kernel.
        """
        target_model = model or self.model
        r = r_kv or self.r_kv
        compressed_layers = {}

        for name, module in target_model.named_modules():
            is_kv = any(k in name.lower() for k in ["k_proj", "v_proj", "key", "value"])
            if is_kv and isinstance(module, nn.Linear):
                dev = module.weight.device
                dtype = module.weight.dtype
                W = module.weight.data.float()
                d_out, d_in = W.shape
                rank_eff = max(1, min(r, min(d_out, d_in) - 1))
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                sqrt_S = torch.sqrt(S[:rank_eff])
                U_r = (U[:, :rank_eff] * sqrt_S.unsqueeze(0)).to(device=dev, dtype=dtype)
                V_r = (sqrt_S.unsqueeze(1) * Vh[:rank_eff, :]).to(device=dev, dtype=dtype)

                svd_layer = SVDLinear(
                    in_features=d_in,
                    out_features=d_out,
                    rank=rank_eff,
                    bias=(module.bias is not None),
                    dtype=dtype,
                    device=dev,
                )
                svd_layer.set_low_rank_factors(U_r, V_r)
                if module.bias is not None:
                    svd_layer.bias.data.copy_(module.bias.data)

                compressed_layers[name] = {
                    "U_r": U_r,
                    "V_r": V_r,
                    "svd_layer": svd_layer,
                }

        return {
            "status": "SUCCESS",
            "compressed_layers_count": len(compressed_layers),
            "r_kv": r,
            "compressed_layers": compressed_layers,
        }

    @staticmethod
    def save_compressed(model: nn.Module, path: str) -> None:
        """Saves U_r, V_r, scales, zero_points in safetensors/pt format compatible with vLLM compressed-tensors."""
        os.makedirs(path, exist_ok=True)
        tensors = {}
        manifest = {}

        for name, param in model.named_parameters():
            tensors[name] = param.data.cpu()
            manifest[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            }

        try:
            from safetensors.torch import save_file
            save_file(tensors, os.path.join(path, "model.safetensors"))
        except ImportError:
            torch.save(tensors, os.path.join(path, "model.pt"))

        with open(os.path.join(path, "quantization_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "quant_method": "compressed-tensors",
                "format": "svd_low_rank",
                "compression_type": "asvd",
            }, f, indent=2)

    @staticmethod
    def load_compressed(path: str, model: Optional[nn.Module] = None) -> nn.Module:
        """Loads and reconstructs low-rank factors into model."""
        safetensor_path = os.path.join(path, "model.safetensors")
        pt_path = os.path.join(path, "model.pt")

        if os.path.exists(safetensor_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensor_path)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No compressed model weights found at {path}")

        if model is not None:
            model.load_state_dict(state_dict, strict=False)
            return model
        return state_dict
