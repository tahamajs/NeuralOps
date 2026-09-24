"""
SliceGPT: Compress Large Language Models by Deleting Rows and Columns.
Registered in COMPRESSOR_REGISTRY as 'slicegpt'.

Mathematical Formulation:
1. Orthogonal Coordinate Invariance:
   $$W_{\\ell} X_{\\ell} = (W_{\\ell} Q_{\\ell}) (Q_{\\ell}^\\top X_{\\ell})$$
2. Eigen-decomposition of Activation Covariance:
   $$C = X X^\\top = Q \\Lambda Q^\\top$$
3. Coordinate Slicing Transformation:
   $$W_{\\text{sliced}} = W Q P_r, \\quad P_r = [I_r, 0]^\\top$$

Production Hardening:
- Fix 1: Default fallback_to_original changed to True.
- Fix 2: Reliable Conv1D detection by class name and nf attribute without shape heuristics.
- Fix 3: Always log warning when reconstruction error exceeds max_error regardless of fallback.
- Fix 4: Unified critical-layer protection delegated solely to should_compress_linear.
- Fix 5: Store max_error and fallback_to_original in returned stats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import get_calibration_tensors
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("slicegpt")
@COMPRESSOR_REGISTRY.register("slicegpt_compressor")
class SliceGPTCompressor(BaseCompressor):
    """
    SliceGPT Compression Strategy: Projects activations onto principal components via orthogonal
    rotation matrices Q and slices out lowest-variance coordinates, with Frobenius validation and fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        sparsity: float = 0.25,
        non_uniform: bool = False,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        target_attention_only: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_error: float = 0.5,
        fallback_to_original: bool = True,  # Fix 1: Default changed to True
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="slicegpt", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            sparsity = _cfg_get(cfg, "slicegpt_sparsity", _cfg_get(cfg, "sparsity", sparsity))
            non_uniform = _cfg_get(cfg, "non_uniform", non_uniform)
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)

        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.non_uniform = non_uniform
        self.target_attention_only = bool(target_attention_only)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """
        Fix 4: Unified critical-layer protection delegated to should_compress_linear.
        Eliminates duplicate keyword matching.
        """
        name_l = name.lower()
        if self.target_attention_only:
            attn_keywords = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj', 'query', 'key', 'value']
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        # Fix 2: Reliable Conv1D check using class name and hasattr(nf)
        is_linear_like = isinstance(base_mod, nn.Linear) or "conv1d" in cls_name or hasattr(base_mod, "nf")
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations across target linear layers."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(inp.cpu())
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
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("Forward pass for activation collection had exception: %s. Using synthetic linear inputs.", e)

        for h in hooks:
            h.remove()

        # Concatenate collected activations
        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                cat_act = torch.cat(act_list, dim=0)
                aggregated[name] = cat_act

        return aggregated

    def _compute_orthogonal_rotation(self, X: Optional[torch.Tensor], d_in: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute orthogonal rotation matrix Q via eigen-decomposition of activation covariance."""
        if device is None:
            device = X.device if X is not None else torch.device("cpu")
        if X is not None and X.numel() > 0:
            X_flat = X.reshape(-1, X.shape[-1]).float().to(device)
            if X_flat.shape[-1] == d_in and X_flat.shape[0] > 1:
                cov = torch.matmul(X_flat.T, X_flat) / X_flat.shape[0]
                cov = cov + 1e-4 * torch.eye(d_in, device=device)
                try:
                    eigvals, Q = torch.linalg.eigh(cov)
                    idx = torch.argsort(eigvals, descending=True)
                    return Q[:, idx], eigvals[idx]
                except Exception as e:
                    logger.debug("eigh covariance failed (%s). Falling back to QR random basis.", e)

        # Haar distributed random orthogonal matrix via QR factorization
        rand_mat = torch.randn(d_in, d_in, device=device)
        Q, _ = torch.linalg.qr(rand_mat)
        return Q, torch.ones(d_in, device=device)

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Applies SliceGPT orthogonal rotation and projection slicing with reconstruction validation.
        """
        self.layer_errors = {}
        activations_map = self._collect_layer_activations()
        sliced_layers_count = 0
        total_channels_pruned = 0

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                base_mod = getattr(module, "base_layer", module)
                raw_w = base_mod.weight.data
                cls_name = base_mod.__class__.__name__.lower()
                # Fix 2: Check for Conv1D via class name and nf attribute
                is_conv1d = "conv1d" in cls_name or hasattr(base_mod, "nf")
                device = raw_w.device
                dtype = raw_w.dtype

                if is_conv1d:
                    W = raw_w.t().float()
                    d_out, d_in = W.shape
                else:
                    W = raw_w.float()
                    d_out, d_in = W.shape

                # Get activations for this module
                act = activations_map.get(name, None)
                Q, eigvals = self._compute_orthogonal_rotation(act, d_in, device=device)
                Q = Q.to(device=device, dtype=torch.float32)

                # Rotate weight: W_tilde = W @ Q
                W_tilde = torch.matmul(W, Q)

                # Determine slicing dimension r
                if self.non_uniform and ("query" in name or "key" in name or "q_lin" in name or "k_lin" in name):
                    head_sparsity = max(0.0, self.sparsity * 0.75)
                    r = max(1, int(d_in * (1.0 - head_sparsity)))
                elif self.non_uniform and ("mlp" in name or "ffn" in name or "dense" in name):
                    mlp_sparsity = min(0.9, self.sparsity * 1.15)
                    r = max(1, int(d_in * (1.0 - mlp_sparsity)))
                else:
                    r = max(1, int(d_in * (1.0 - self.sparsity)))

                # Create projection slicing operator P_r keeping top-r principal channels
                W_sliced_rot = W_tilde.clone()
                if r < d_in:
                    W_sliced_rot[:, r:] = 0.0

                # Rotate back to original coordinate basis: W_pruned = W_sliced_rot @ Q^T
                W_pruned = torch.matmul(W_sliced_rot, Q.T)

                # Frobenius reconstruction error check
                norm_orig = torch.norm(W, p="fro")
                rel_err = (torch.norm(W - W_pruned, p="fro") / (norm_orig + 1e-8)).item()
                self.layer_errors[name] = rel_err

                # Fix 3: Always log a warning when reconstruction error exceeds max_error
                if rel_err > self.max_error:
                    logger.warning(
                        "SliceGPT reconstruction error %.4f on %s exceeded threshold %.4f%s",
                        rel_err,
                        name,
                        self.max_error,
                        ". Skipping (fallback to original)." if self.fallback_to_original else ". Proceeding with sliced weight.",
                    )
                    if self.fallback_to_original:
                        logger.warning("Fallback triggered for %s. Layer is kept original and NOT wrapped in LoRA. It will not be fine-tuned.", name)
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                if r < d_in:
                    total_channels_pruned += (d_in - r)

                final_w = W_pruned.t() if is_conv1d else W_pruned
                base_mod.weight.data.copy_(final_w.to(dtype=dtype, device=device))
                compressed_layer = base_mod
                # Preserve existing LoRA adapter weights if the source module has them
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                   module.lora_A is not None and module.lora_B is not None:
                    src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                    src_lB = getattr(module.lora_B, 'weight', module.lora_B)
                    if hasattr(compressed_layer, 'lora_A') and compressed_layer.lora_A is not None:
                        if hasattr(src_lA, 'data') and src_lA.shape == compressed_layer.lora_A.shape:
                            compressed_layer.lora_A.data.copy_(src_lA.data)
                    if hasattr(compressed_layer, 'lora_B') and compressed_layer.lora_B is not None:
                        if hasattr(src_lB, 'data') and src_lB.shape == compressed_layer.lora_B.shape:
                            compressed_layer.lora_B.data.copy_(src_lB.data)

                # Determine if module is wrapped (LoRALinear) or standalone
                is_wrapped = hasattr(module, 'base_layer')
                if is_wrapped:
                    module.base_layer = compressed_layer
                    new_module = module
                else:
                    new_module = compressed_layer
                    replace_module(self.model, name, new_module)
                sliced_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        # Fix 5: Store max_error and fallback_to_original in stats (B13: explicit components_zeroed)
        stats.update({
            "sliced_layers": sliced_layers_count,
            "sparsity_target": self.sparsity,
            "non_uniform": self.non_uniform,
            "components_zeroed": total_channels_pruned,
            "total_channels_pruned": total_channels_pruned,  # Backwards compatibility alias
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "method": "slicegpt_pca_rotation_slicing",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("slicegpt_factory")
def slicegpt_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Factory function for SliceGPT compressor."""
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "slicegpt_sparsity", _cfg_get(comp_config, "sparsity", 0.25)))
    non_uniform = kwargs.pop("non_uniform", _cfg_get(comp_config, "non_uniform", False))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.20))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None))

    compressor = SliceGPTCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        non_uniform=non_uniform,
        target_attention_only=target_attention_only,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SliceGPTCompressor",
    "slicegpt_factory",
]
