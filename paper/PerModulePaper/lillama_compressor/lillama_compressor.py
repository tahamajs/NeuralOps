"""
LiLLaMA: Local Distillation via Activation Cross-Covariance Singular Value Decomposition.
Registered in COMPRESSOR_REGISTRY as 'lillama' and 'lillama_compressor'.

Mathematical Formulation:
1. Input and Output Activations:
   $$X \\in \\mathbb{R}^{N \\times d_{in}}, \\quad Y \\in \\mathbb{R}^{N \\times d_{out}}$$
2. Empirical Cross-Covariance Matrix:
   $$C_{YX} = \\frac{1}{N} Y^\\top X \\in \\mathbb{R}^{d_{out} \\times d_{in}}$$
3. Truncated Singular Value Decomposition:
   $$C_{YX} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
4. Optimal Distilled Weight Reconstruction:
   $$W_{distill} = U_r \\Sigma_r V_r^\\top$$

Production Hardening:
- Fix 1: Reconstruction error validation with configurable max_error (default 0.20) and safe fallback.
- Fix 2: Support for Conv1D (GPT-2) and fused QKV projections (Qwen/LLaMA).
- Fix 3: FP32 computations with cross-covariance SVD and weight SVD fallback.
- Fix 4: Comprehensive per-layer statistics including ranks, errors, and cross-covariance usage flags.
- Fix 5: Configurable target modules / skip_attention flag and critical-layer protection via should_compress_linear.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import get_calibration_tensors
from framework.utils.compat import should_compress_linear

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin", "self_attn"
}


@COMPRESSOR_REGISTRY.register("lillama")
@COMPRESSOR_REGISTRY.register("lillama_compressor")
class LiLLaMACompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    LiLLaMA Compressor: Local activation cross-covariance SVD distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        rank: int = 32,
        calibration_texts: Optional[List[str]] = None,
        calibration_loader: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        skip_attention: bool = False,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lillama", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "lillama_rank", _cfg_get(cfg, "rank", rank))
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            skip_attention = _cfg_get(cfg, "skip_attention", skip_attention)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.rank = max(1, int(rank))
        self.calibration_texts = calibration_texts
        self.calibration_loader = calibration_loader
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.skip_attention = bool(skip_attention)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False

        if self.target_attention_only:
            is_attn = any(k in name.lower() for k in _ATTN_NAMES)
            if not is_attn:
                return False

        if self.skip_attention:
            attn_keywords = {"query", "key", "value", "q_lin", "k_lin", "v_lin", "q_proj", "k_proj", "v_proj", "out_lin", "out_proj", "dense", "o_proj", "c_attn"}
            last_name = name.split(".")[-1].lower()
            if last_name in attn_keywords or any(k in name.lower() for k in ["self_attn", "attn.c_attn"]):
                return False

        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _collect_io_activations(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect input and output activations across target layers."""
        self.model.eval()
        activations: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                inp = inputs[0].detach() if inputs and isinstance(inputs[0], torch.Tensor) else None
                out = output.detach() if isinstance(output, torch.Tensor) else (output[0].detach() if isinstance(output, (tuple, list)) else None)
                if inp is not None and out is not None:
                    if name not in activations:
                        activations[name] = {"in": [], "out": []}
                    activations[name]["in"].append(inp.cpu())
                    activations[name]["out"].append(out.cpu())
            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    calib_inputs = {k: v.to(self.target_device) if isinstance(v, torch.Tensor) else v for k, v in calib_inputs.items()}
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs.to(self.target_device))
                elif isinstance(calib_inputs, (list, tuple)):
                    calib_inputs = [v.to(self.target_device) if isinstance(v, torch.Tensor) else v for v in calib_inputs]
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("LiLLaMA IO activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, data in activations.items():
            if data["in"] and data["out"]:
                cat_in = torch.cat(data["in"], dim=0)
                cat_out = torch.cat(data["out"], dim=0)
                aggregated[name] = (cat_in, cat_out)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes LiLLaMA cross-covariance SVD distillation."""
        activations_map = self._collect_io_activations()
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors = {}
        layer_details = {}

        total_orig_params = sum(p.numel() for p in self.model.parameters())
        total_comp_params = 0

        for name, module in self.model.named_modules():
            if not self._is_target_module(name, module):
                # Do NOT add to layer_errors; only track compressed layers
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue

            # Support LoRALinear wrapper
            target_layer = getattr(module, "base_layer", module)
            raw_w = target_layer.weight.data
            cls_name = target_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
            device = raw_w.device
            dtype = target_layer.weight.dtype

            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            used_cross_cov = False

            if name in activations_map:
                inp, out = activations_map[name]
                inp_flat = inp.reshape(-1, inp.shape[-1]).float()
                out_flat = out.reshape(-1, out.shape[-1]).float()

                if inp_flat.shape[-1] == d_in and out_flat.shape[-1] == d_out:
                    C = torch.matmul(out_flat.T, inp_flat) / max(len(inp_flat), 1)
                    try:
                        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
                        r = min(self.rank, min(d_out, d_in), len(S))
                        cum_e = (S[:r] ** 2).sum() / ((S ** 2).sum() + 1e-12)
                        if cum_e < 0.95:
                            cum_ratio = torch.cumsum(S ** 2, 0) / ((S ** 2).sum() + 1e-12)
                            r_needed = int((cum_ratio < 0.95).sum().item()) + 1
                            r = min(r_needed, len(S))
                        W_new = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
                        used_cross_cov = True
                    except Exception:
                        W_new = W
                else:
                    W_new = W
            else:
                # SVD on weight itself as fallback
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    r = min(self.rank, len(S))
                    cum_e = (S[:r] ** 2).sum() / ((S ** 2).sum() + 1e-12)
                    if cum_e < 0.95:
                        cum_ratio = torch.cumsum(S ** 2, 0) / ((S ** 2).sum() + 1e-12)
                        r_needed = int((cum_ratio < 0.95).sum().item()) + 1
                        r = min(r_needed, len(S))
                    W_new = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
                except Exception:
                    W_new = W

            # Relative Frobenius reconstruction error check
            norm_orig = torch.norm(W, p="fro")
            rel_err = (torch.norm(W - W_new, p="fro") / (norm_orig + 1e-8)).item()
            self.layer_errors[name] = rel_err

            if rel_err > self.max_error:
                logger.warning(
                    "LiLLaMA reconstruction error %.4f on %s exceeded max_error %.4f%s",
                    rel_err,
                    name,
                    self.max_error,
                    ". Keeping original layer." if self.fallback_to_original else ". Proceeding with distilled weights.",
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False
                        total_comp_params += p.numel()
                    continue

            final_w = W_new.t().contiguous() if is_conv1d else W_new
            target_layer.weight.data.copy_(final_w.to(dtype=dtype, device=device))
            # Freeze base weight, preserve LoRA adapter trainability
            target_layer.weight.requires_grad = False
            if hasattr(target_layer, "bias") and target_layer.bias is not None:
                target_layer.bias.requires_grad = False

            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for p in [module.lora_A, module.lora_B]:
                    if hasattr(p, "weight"):
                        p.weight.requires_grad = True
                    elif isinstance(p, nn.Parameter):
                        p.requires_grad = True

            self.compressed_layers.append(name)
            for p in module.parameters(recurse=False):
                total_comp_params += p.numel()

            layer_details[name] = {
                "rank": r if 'r' in locals() else self.rank,
                "rel_error": rel_err,
                "used_cross_cov": used_cross_cov,
                "shape": tuple(final_w.shape),
            }

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.layer_errors if k in self.compressed_layers)
            / max(1, len([k for k in self.layer_errors if k in self.compressed_layers]))
            if self.compressed_layers
            else 0.0
        )
        eff_ratio = total_comp_params / max(1, total_orig_params)
        stats.update({
            "distilled_layers": len(self.compressed_layers),
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "rank": self.rank,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "layer_errors": self.layer_errors,
            "layer_details": layer_details,
            "effective_parameter_ratio": eff_ratio,
            "method": "lillama_cross_covariance_svd",
        })
        self.stats = stats
        logger.info(
            "LiLLaMA compression complete: %d layers distilled, avg error: %.4f",
            len(self.compressed_layers),
            avg_err,
        )
        return self.model, stats


def lillama_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for LiLLaMA compressor."""
    rank = _cfg_get(comp_config, "lillama_rank", _cfg_get(comp_config, "rank", kwargs.get("rank", 32)))
    max_error = _cfg_get(comp_config, "max_error", kwargs.get("max_error", 0.20))
    fallback_to_original = _cfg_get(comp_config, "fallback_to_original", kwargs.get("fallback_to_original", True))
    skip_attention = _cfg_get(comp_config, "skip_attention", kwargs.get("skip_attention", False))
    target_attention_only = _cfg_get(comp_config, "target_attention_only", kwargs.get("target_attention_only", False))
    tokenizer = getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", kwargs.get("tokenizer", None))

    compressor = LiLLaMACompressor(
        model,
        tokenizer=tokenizer,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        skip_attention=skip_attention,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "LiLLaMACompressor",
    "lillama_factory",
]
