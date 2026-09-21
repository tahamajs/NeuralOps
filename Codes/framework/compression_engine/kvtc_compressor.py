"""
KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive quantization
      + entropy coding for compact model weight / KV cache storage.

Paper: "KV Cache Transform Coding for Compact Storage in LLM Inference"
       Staniszewski & Lancucki, ICLR 2026.

NOTE ON SCOPE:
In this codebase, KVTC operates primarily as a static weight-matrix compressor
(PCA low-rank projection + uniform integer quantization + optional LoRA adaptation)
applied to linear layers at initialization time, rather than dynamic runtime KV cache
interception during autoregressive generation. For runtime KV cache transform coding,
the `KVTCStore` class is provided as a standalone utility.

Registered in COMPRESSOR_REGISTRY as:
  - 'kvtc' (primary)
  - 'kvtc_quantizer'
  - 'pca_quant'
  - 'pca_quant_compressor'
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin",
}


class SimpleEntropyCoder:
    """Lightweight entropy coder using empirical symbol frequency estimation."""

    @staticmethod
    def compute_entropy(tensor: torch.Tensor) -> float:
        """Compute empirical Shannon entropy in bits per symbol."""
        if tensor.numel() == 0:
            return 0.0
        unique_vals, counts = torch.unique(tensor, return_counts=True)
        probs = counts.float() / float(tensor.numel())
        entropy = -torch.sum(probs * torch.log2(torch.clamp(probs, min=1e-12))).item()
        return float(entropy)

    @staticmethod
    def compute_compression_ratio(tensor: torch.Tensor, uncompressed_bits: int = 16) -> float:
        """Estimate theoretical compression ratio relative to FP16/FP32 baseline."""
        entropy = SimpleEntropyCoder.compute_entropy(tensor)
        return float(uncompressed_bits / max(entropy, 1e-4))


class KVTCStore:
    is_custom_attention: bool = True
    """
    Runtime KV Cache Transform Coding engine.
    Applies PCA decorrelation across head feature dimensions and adaptive quantization,
    supporting dynamic sequence lengths across prefill and decoding phases.
    """

    def __init__(self, bits: int = 4, pca_energy: float = 0.95) -> None:
        self.bits = int(bits)
        self.pca_energy = float(pca_energy)
        self.pca_basis_k: Optional[torch.Tensor] = None
        self.pca_basis_v: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset PCA basis to prevent cross-sequence memory leaks."""
        self.pca_basis_k = None
        self.pca_basis_v = None

    @torch.no_grad()
    def calibrate_basis(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Calibrate PCA projection basis from sample K/V tensor (B, H, N, D)."""
        k_flat = k.detach().float().reshape(-1, k.shape[-1])
        v_flat = v.detach().float().reshape(-1, v.shape[-1])

        # SVD on centered channels in FP32
        _, _, Vt_k = torch.linalg.svd(k_flat - k_flat.mean(dim=0, keepdim=True), full_matrices=False)
        _, _, Vt_v = torch.linalg.svd(v_flat - v_flat.mean(dim=0, keepdim=True), full_matrices=False)

        self.pca_basis_k = Vt_k.t().contiguous().to(device=k.device, dtype=k.dtype)
        self.pca_basis_v = Vt_v.t().contiguous().to(device=v.device, dtype=v.dtype)

    @torch.no_grad()
    def compress_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Transform and quantize dynamic KV cache tensors.
        Returns reconstructed (k_hat, v_hat) and coding statistics.
        """
        orig_shape = k.shape
        D = orig_shape[-1]

        if self.pca_basis_k is None or self.pca_basis_v is None:
            self.calibrate_basis(k, v)

        k_flat = k.reshape(-1, D)
        v_flat = v.reshape(-1, D)

        k_trans = torch.matmul(k_flat, self.pca_basis_k)
        v_trans = torch.matmul(v_flat, self.pca_basis_v)

        qmax = (2 ** (self.bits - 1)) - 1
        scale_k = torch.clamp(k_trans.abs().max() / float(qmax), min=1e-6)
        scale_v = torch.clamp(v_trans.abs().max() / float(qmax), min=1e-6)

        q_k = torch.clamp(torch.round(k_trans / scale_k), -qmax, qmax)
        q_v = torch.clamp(torch.round(v_trans / scale_v), -qmax, qmax)

        h_k = SimpleEntropyCoder.compute_entropy(q_k)
        h_v = SimpleEntropyCoder.compute_entropy(q_v)

        k_hat = torch.matmul(q_k * scale_k, self.pca_basis_k.t().contiguous()).reshape(orig_shape)
        v_hat = torch.matmul(q_v * scale_v, self.pca_basis_v.t().contiguous()).reshape(orig_shape)

        mean_h = (h_k + h_v) / 2.0
        stats: Dict[str, Any] = {
            "entropy_k_bits": h_k,
            "entropy_v_bits": h_v,
            "mean_entropy_bits": mean_h,
            "entropy_lower_bound_bits": mean_h,
            "note": "Theoretical lower bound; actual bitstream not implemented.",
            "theoretical_compression_ratio": 16.0 / max(mean_h, 1e-4),
        }
        return k_hat, v_hat, stats


class KVTCQuantizedLoRALinear(nn.Module):
    """
    Linear layer containing true integer quantized base weights (int8/int16 buffer)
    with scale and zero_point buffers, dequantized on the fly during forward(),
    plus optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        int_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.compute_dtype = compute_dtype
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else int_weight.device

        # Store true integer weights as buffer (not float Parameter)
        self.register_buffer("int_weight", int_weight.to(device=target_device))
        self.register_buffer("scale", scale.to(device=target_device, dtype=torch.float32))
        self.register_buffer("zero_point", zero_point.to(device=target_device, dtype=torch.float32))

        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=compute_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=compute_dtype, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=compute_dtype, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Backward-compatibility property returning on-the-fly dequantized weight."""
        return (self.int_weight.to(dtype=self.compute_dtype) * self.scale + self.zero_point)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        int_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> KVTCQuantizedLoRALinear:
        target_mod = getattr(linear, "base_layer", linear)
        bias = target_mod.bias.detach() if hasattr(target_mod, "bias") and target_mod.bias is not None else None
        in_features = getattr(target_mod, "in_features", int_weight.shape[1])
        out_features = getattr(target_mod, "out_features", int_weight.shape[0])

        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)
        effective_lora_r = lora_r
        effective_lora_alpha = lora_alpha
        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            effective_lora_r = w_A.shape[0]
            if hasattr(linear, "scaling") and linear.scaling:
                effective_lora_alpha = linear.scaling * effective_lora_r
            elif hasattr(linear, "lora_alpha"):
                effective_lora_alpha = linear.lora_alpha

        if compute_dtype is None:
            if hasattr(target_mod, "weight") and target_mod.weight is not None:
                compute_dtype = target_mod.weight.dtype
            else:
                compute_dtype = torch.float32

        new_layer = cls(
            in_features=in_features,
            out_features=out_features,
            int_weight=int_weight,
            scale=scale,
            zero_point=zero_point,
            bias=bias,
            lora_r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
            device=int_weight.device,
            compute_dtype=compute_dtype,
        )

        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if hasattr(w_A, "data") and hasattr(w_B, "data") and new_layer.lora_A is not None and new_layer.lora_B is not None:
                new_layer.lora_A.data.copy_(w_A.data.to(dtype=new_layer.lora_A.dtype, device=new_layer.lora_A.device))
                new_layer.lora_B.data.copy_(w_B.data.to(dtype=new_layer.lora_B.dtype, device=new_layer.lora_B.device))
                new_layer.lora_A.requires_grad = True
                new_layer.lora_B.requires_grad = True

        return new_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = (self.int_weight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype) + self.zero_point.to(dtype=x.dtype))
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        out = F.linear(x, w_eff, bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(
                F.linear(x.to(dtype=self.lora_A.dtype), self.lora_A),
                self.lora_B
            ).to(dtype=out.dtype)
            out = out + self.scaling * lora_out
        return out


@COMPRESSOR_REGISTRY.register("kvtc")
@COMPRESSOR_REGISTRY.register("kvtc_quantizer")
@COMPRESSOR_REGISTRY.register("pca_quant")
@COMPRESSOR_REGISTRY.register("pca_quant_compressor")
class KVTCCompressor(BaseCompressor):
    """
    KVTC: KV Cache Transform Coding compressor (Static Weight PCA + Quantization).
    Compresses linear projection weights using PCA decorrelation,
    percentile clipping, uniform integer quantization, and entropy coding estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        pca_energy: float = 0.95,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        target_attention_only: bool = False,
        skip_attention_qkv: bool = True,
        min_rank: int = 4,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="kvtc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        self.cfg = cfg
        self.bits = int(_cfg_get(cfg, "kvtc_bits", _cfg_get(cfg, "bits", bits)))
        self.pca_energy = float(_cfg_get(cfg, "kvtc_pca_energy", _cfg_get(cfg, "pca_energy", pca_energy)))
        self.max_error = float(_cfg_get(cfg, "kvtc_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.fallback_to_original = bool(_cfg_get(cfg, "kvtc_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)))
        self.lora_r = int(_cfg_get(cfg, "lora_r", lora_r))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.target_attention_only = bool(
            _cfg_get(cfg, "kvtc_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.skip_attention_qkv = bool(_cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.min_rank = max(1, int(_cfg_get(cfg, "min_rank", min_rank)))
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    @staticmethod
    def _percentile_clip(W: torch.Tensor, low: float = 0.001, high: float = 0.999) -> torch.Tensor:
        """Clip extreme outliers from weight tensor before computing quantization scale."""
        if W.numel() == 0:
            return W
        q_low = torch.quantile(W.float(), low)
        q_high = torch.quantile(W.float(), high)
        return torch.clamp(W, min=q_low, max=q_high)

    def _quantize_to_int(
        self, W: torch.Tensor, bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous tensor to true integer format (torch.int8 or torch.int16).
        Uses signed symmetric quantization around 0:
            int_w in [-2^(bits-1), 2^(bits-1) - 1]
            scale = max(|W|) / (2^(bits-1) - 1)
            W_dequant ≈ int_w * scale + zero_point (zero_point = 0.0)
        """
        qmax = (2 ** (bits - 1)) - 1
        qmin = -(2 ** (bits - 1))

        max_abs = torch.clamp(W.abs().max(), min=1e-6)
        scale = (max_abs / float(qmax)).item()
        zero_point = 0.0

        q_float = torch.round(W / scale)
        q_clamped = torch.clamp(q_float, qmin, qmax)

        scale_t = torch.tensor(scale, dtype=torch.float32, device=W.device)
        zp_t = torch.tensor(zero_point, dtype=torch.float32, device=W.device)

        if bits <= 8:
            int_w = q_clamped.to(torch.int8)
        else:
            int_w = q_clamped.to(torch.int16)

        return int_w, scale_t, zp_t

    def _dequantize_from_int(
        self, int_w: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize integer weight back to floating point."""
        return int_w.to(dtype=scale.dtype) * scale + zero_point

    def _estimate_actual_bytes(self, model: nn.Module) -> int:
        """
        Estimate physical memory storage in bytes across all parameters and buffers,
        taking into account actual tensor element size (int8=1 byte, fp16=2 bytes, fp32=4 bytes).
        Uses tensor storage data_ptr() to avoid duplicate counting of shared weights.
        """
        total_bytes = 0
        seen_ptrs: Set[int] = set()

        for p in model.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total_bytes += p.numel() * p.element_size()

        for b in model.buffers():
            ptr = b.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total_bytes += b.numel() * b.element_size()

        return total_bytes

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        if self._is_gqa_asymmetric(name, module):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear_like = isinstance(target_mod, nn.Linear) or "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not is_linear_like:
            return False
        if isinstance(target_mod, KVTCQuantizedLoRALinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using PCA decorrelation and true integer quantization."""
        total_entropy = 0.0
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        layer_details: Dict[str, Any] = {}
        seen_weights: Set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Physical byte tracking before compression
        initial_actual_bytes = self._estimate_actual_bytes(self.model)

        # Freeze non-target linear/conv1d layers when target_attention_only is True
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_mod.weight.data
            orig_params = raw_w.numel()
            if getattr(target_mod, "bias", None) is not None:
                orig_params += target_mod.bias.numel()
            total_orig_params += orig_params

            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            min_dim = min(d_out, d_in)

            try:
                # Step 1: PCA decorrelation via SVD in FP32
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                total_energy = (S ** 2).sum()
                if total_energy > 0:
                    cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                    idx = torch.where(cum_energy >= self.pca_energy)[0]
                    rank = int(idx[0].item()) + 1 if len(idx) > 0 else min_dim
                else:
                    rank = min_dim

                effective_min_rank = min(self.min_rank, min_dim)
                rank = max(effective_min_rank, min(rank, min_dim))

                # Reconstruct low-rank approximation
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vt_r = Vt[:rank, :]
                W_lowrank = (U_r * S_r.unsqueeze(0)) @ Vt_r

                # Step 2: Percentile outlier clipping before quantization
                W_clipped = self._percentile_clip(W_lowrank, low=0.001, high=0.999)

                # Step 3: Integer quantization
                int_w, scale, zp = self._quantize_to_int(W_clipped, bits=self.bits)
                W_quant = self._dequantize_from_int(int_w, scale, zp)
                W_quant = torch.nan_to_num(W_quant, nan=0.0)

                # Reconstruction error check against original W
                norm_orig = torch.norm(W, p="fro").item()
                rel_err = (torch.norm(W - W_quant, p="fro") / max(norm_orig, 1e-12)).item()
                self.layer_errors[name] = float(rel_err)

                if not math.isfinite(rel_err) or rel_err > self.max_error:
                    logger.warning(
                        "KVTC reconstruction error %.4f on %s exceeded max_error %.4f%s",
                        rel_err,
                        name,
                        self.max_error,
                        ". Keeping original layer." if self.fallback_to_original else ". Proceeding with quantized weight.",
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        total_comp_params += orig_params
                        continue

                # Step 4: Empirical Shannon entropy calculation on discrete integer symbols
                entropy = SimpleEntropyCoder.compute_entropy(int_w)
                total_entropy += entropy

                new_layer = KVTCQuantizedLoRALinear.from_linear(
                    module,
                    int_weight=int_w,
                    scale=scale,
                    zero_point=zp,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    compute_dtype=dtype,
                )

                # Effective compressed parameters count for reporting
                quant_equiv_params = int(orig_params * (self.bits / 16.0))
                lora_params = (
                    (new_layer.lora_A.numel() + new_layer.lora_B.numel())
                    if (new_layer.lora_A is not None and new_layer.lora_B is not None)
                    else 0
                )
                total_comp_params += (quant_equiv_params + lora_params)

                replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
                layer_details[name] = {
                    "rank": rank,
                    "rel_error": rel_err,
                    "entropy": entropy,
                    "shape": tuple(int_w.shape),
                }
            except Exception as e:
                logger.warning("KVTC compression exception on %s: %s. Keeping original layer.", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params
                self.layer_errors[name] = 1.0

        # Physical byte tracking after compression
        final_actual_bytes = self._estimate_actual_bytes(self.model)

        # Real post-compression forward sanity check
        forward_check_success = False
        forward_check_error = None
        try:
            p0 = next(self.model.parameters(), None)
            if p0 is None:
                p0 = next(self.model.buffers(), None)
            dev = p0.device if p0 is not None else torch.device("cpu")
            self.model.eval()

            # Try text classification / sequence classification format: input_ids (1, 8)
            with torch.no_grad():
                test_passed = False
                try:
                    dummy_ids = torch.randint(0, 100, (1, 8), device=dev)
                    _ = self.model(dummy_ids)
                    test_passed = True
                except Exception:
                    pass

                if not test_passed:
                    # Try continuous hidden tensor (1, 8, d_model)
                    dummy_x = torch.randn(1, 8, 64, device=dev)
                    _ = self.model(dummy_x)
                    test_passed = True

            forward_check_success = test_passed
        except Exception as e:
            forward_check_error = str(e)
            logger.error("KVTC post-compression forward pass failed: %s", e)
            if not kwargs.get("ignore_forward_check", False):
                raise RuntimeError(
                    f"KVTC produced a model that fails the forward pass sanity check: {e}"
                ) from e

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_entropy = total_entropy / max(len(self.compressed_layers), 1)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        param_ratio = round(total_comp_params / max(total_orig_params, 1), 4) if total_orig_params > 0 else 1.0

        bytes_reduction_pct = (
            (initial_actual_bytes - final_actual_bytes) / float(initial_actual_bytes) * 100.0
            if initial_actual_bytes > 0
            else 0.0
        )

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "bits": self.bits,
            "pca_energy": self.pca_energy,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "skip_attention_qkv": self.skip_attention_qkv,
            "avg_entropy_bits": round(avg_entropy, 2),
            "entropy_lower_bound_bits": round(avg_entropy, 2),
            "entropy_coding_note": "Theoretical lower bound; actual bitstream not implemented.",
            "avg_reconstruction_error": avg_err,
            "effective_parameter_ratio": param_ratio,
            "initial_actual_bytes": initial_actual_bytes,
            "final_actual_bytes": final_actual_bytes,
            "actual_bytes_reduction_pct": round(bytes_reduction_pct, 2),
            "theoretical_entropy_compression_ratio": round(16.0 / max(avg_entropy, 1e-4), 2),
            "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
            "layer_errors": self.layer_errors,
            "layer_details": layer_details,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "forward_check_error": forward_check_error,
            "method": "kvtc_pca_quant",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("kvtc_factory")
@COMPRESSOR_REGISTRY.register("pca_quant_factory")
def kvtc_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """KVTC compressor factory function."""
    bits = kwargs.pop("bits", _cfg_get(comp_config, "kvtc_bits", _cfg_get(comp_config, "bits", 4)))
    pca_energy = kwargs.pop("pca_energy", _cfg_get(comp_config, "kvtc_pca_energy", _cfg_get(comp_config, "pca_energy", 0.95)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "kvtc_max_error", _cfg_get(comp_config, "max_error", 0.50)))
    fallback_to_original = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "kvtc_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 16))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "kvtc_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv",
        _cfg_get(comp_config, "skip_attention_qkv", True),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = KVTCCompressor(
        model,
        bits=bits,
        pca_energy=pca_energy,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        skip_attention_qkv=skip_attention_qkv,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


kvtc_compressor_factory = kvtc_factory
pca_quant_factory = kvtc_factory

__all__ = [
    "KVTCQuantizedLoRALinear",
    "KVTCCompressor",
    "KVTCStore",
    "SimpleEntropyCoder",
    "kvtc_factory",
    "kvtc_compressor_factory",
    "pca_quant_factory",
]
