"""
SparseGPT: One-Shot Layer-Wise Weight Pruning with Second-Order Error Compensation.
Registered in COMPRESSOR_REGISTRY as 'sparsegpt'.

Mathematical Formulation:
1. Empirical Hessian Formulation:
   $$H = 2 X X^\\top + \\lambda I \\in \\mathbb{R}^{d_{in} \\times d_{in}}$$
2. Cholesky Factorization & Inversion:
   $$H = L L^\\top \\implies H^{-1} = (L^{-1})^\\top L^{-1}$$
3. OBS Saliency Metric for Column $j$:
   $$S_{ij} = \\frac{|W_{ij}|^2}{2 [H^{-1}]_{jj}}$$
4. 2:4 Structured or Unstructured Mask $\\mathcal{M}$:
   $$W_{pruned} = W \\odot \\mathcal{M}$$
5. Optimal Brain Surgeon Column Update:
   $$W_{:, j+1:d_{in}} \\leftarrow W_{:, j+1:d_{in}} - (W_{:, j} - W_{pruned, :, j}) \\frac{[H^{-1}]_{j, j+1:d_{in}}}{[H^{-1}]_{jj}}$$
"""

from __future__ import annotations

from typing import Any
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear
from framework.core.pruning_utils import (
    compute_empirical_hessian,
    get_calibration_tensors,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class SparseGPTCompressor(BaseCompressor):
    """
    In-memory PyTorch SparseGPT Compressor.
    
    Performs layer-by-layer second-order pruning with Optimal Brain Surgeon (OBS) weight
    compensation, supporting unstructured sparsity, block sparsity, and strict 2:4 structured sparsity.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        sparsity: float = 0.5,
        prunen: int = 2,
        prunem: int = 4,
        block_size: int = 128,
        percdamp: float = 0.01,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        target_attention_only: bool = False,
        device: str | torch.device | None = None,
        structured_24: bool = False,
        max_error: float = 0.5,
        fallback_to_original: bool = True,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="sparsegpt", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            sparsity = _cfg_get(cfg, "sparsegpt_sparsity", _cfg_get(cfg, "sparsity", sparsity))
            structured_24 = _cfg_get(cfg, "sparsegpt_structured_24", _cfg_get(cfg, "structured_24", structured_24))
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.prunen = prunen
        self.prunem = prunem
        self.block_size = block_size
        self.percdamp = percdamp
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_attention_only = bool(target_attention_only)
        self.structured_24 = structured_24 or (prunen == 2 and prunem == 4 and self.sparsity == 0.5)
        if self.structured_24:
            self.sparsity = 0.5
            self.prunen = 2
            self.prunem = 4
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        if self.target_attention_only:
            attn_keywords = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj', 'query', 'key', 'value']
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        is_linear_like = isinstance(base_mod, nn.Linear) or "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _collect_layer_activations(self) -> dict[str, torch.Tensor]:
        """Capture layer inputs via forward hooks."""
        self.model.eval()
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any):
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
                logger.debug("SparseGPT activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def _compress_linear_module(
        self,
        name: str,
        module: nn.Module,
        activations: torch.Tensor | None,
    ) -> bool:
        """Executes second-order SparseGPT pruning on a single Linear or Conv1D layer."""
        base_mod = getattr(module, "base_layer", module)
        raw_w = base_mod.weight.data
        cls_name = base_mod.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf") or (
            raw_w.ndim == 2
            and hasattr(base_mod, "in_features")
            and base_mod.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
        d_out, d_in = W.shape
        device = W.device

        # Compute empirical Hessian and inverse
        if activations is not None and activations.numel() > 0:
            act_flat = activations.reshape(-1, activations.shape[-1]).float()
            if act_flat.shape[-1] == d_in:
                _, H_inv = compute_empirical_hessian(act_flat, damping=self.percdamp, device=device)
            else:
                H = 2.0 * torch.eye(d_in, device=device)
                H_inv = 0.5 * torch.eye(d_in, device=device)
        else:
            H = 2.0 * torch.eye(d_in, device=device)
            H_inv = 0.5 * torch.eye(d_in, device=device)

        H_inv = H_inv.to(device=device, dtype=torch.float32)

        # Iterate over columns in blocks
        W_pruned = W.clone()
        block_cols = min(self.block_size, d_in)

        for i1 in range(0, d_in, block_cols):
            i2 = min(i1 + block_cols, d_in)
            count = i2 - i1

            W_block_init = W_pruned[:, i1:i2].clone()
            W_block = W_block_init.clone()
            H_inv_block = H_inv[i1:i2, i1:i2].clone()

            for j in range(count):
                w_col = W_block[:, j].clone()
                diag_inv = max(H_inv_block[j, j].item(), 1e-8)
                saliency_col = (w_col ** 2) / (2.0 * diag_inv)

                # N:M structured or unstructured pruning with OBS error compensation (B10, H25)
                if self.structured_24 or (self.prunem > 0 and self.prunen > 0):
                    M = max(1, self.prunem)
                    N = max(1, min(self.prunen, M))
                    # At the end of every M-group or block boundary, apply N:M mask across the group
                    if (j + 1) % M == 0 or (j + 1 == count):
                        start_c = (j // M) * M
                        sub_w = W_block[:, start_c : j + 1]
                        sub_h_diag = torch.diagonal(H_inv_block[start_c : j + 1, start_c : j + 1]).clamp(min=1e-8)
                        sub_saliency = (sub_w ** 2) / (2.0 * sub_h_diag.unsqueeze(0))
                        k_keep = min(N, sub_w.shape[1])
                        _, top_idx = torch.topk(sub_saliency, k=k_keep, dim=1, largest=True)
                        sub_mask = torch.zeros_like(sub_w)
                        sub_mask.scatter_(1, top_idx, 1.0)
                        pruned_sub = sub_w * sub_mask
                        err_sub = sub_w - pruned_sub
                        W_block[:, start_c : j + 1] = pruned_sub

                        # Optimal Brain Surgeon error compensation update
                        if j < count - 1:
                            err_col = err_sub[:, -1]
                            h_slice = H_inv_block[j, j + 1 : count] / diag_inv
                            W_block[:, j + 1 : count] -= torch.outer(err_col, h_slice)
                else:
                    k_prune = int(d_out * self.sparsity)
                    if k_prune > 0:
                        _, prune_idx = torch.topk(saliency_col, k=k_prune, largest=False)
                        mask_col = torch.ones(d_out, device=device)
                        mask_col[prune_idx] = 0.0
                        pruned_w_col = w_col * mask_col
                        err = w_col - pruned_w_col
                        W_block[:, j] = pruned_w_col

                        # Optimal Brain Surgeon error compensation update
                        if j < count - 1:
                            h_slice = H_inv_block[j, j + 1 : count] / diag_inv
                            W_block[:, j + 1 : count] -= torch.outer(err, h_slice)

            W_pruned[:, i1:i2] = W_block

            # Update remaining columns outside the current block
            if i2 < d_in:
                err_block = W_block_init - W_block
                h_cross = H_inv[i1:i2, i2:]  # [count, d_in - i2]
                inv_diag_block = torch.linalg.pinv(H_inv_block)
                update = torch.matmul(err_block, torch.matmul(inv_diag_block, h_cross))
                W_pruned[:, i2:] -= update

        # Relative Frobenius error check
        norm_orig = torch.norm(W, p="fro")
        rel_err = (torch.norm(W - W_pruned, p="fro") / (norm_orig + 1e-8)).item()
        self.layer_errors[name] = rel_err

        if self.fallback_to_original and rel_err > self.max_error:
            logger.warning(
                "SparseGPT reconstruction error %.4f on %s exceeded threshold %.4f. Skipping.",
                rel_err,
                name,
                self.max_error,
            )
            logger.warning("Fallback triggered for %s. Layer is kept original and NOT wrapped in LoRA. It will not be fine-tuned.", name)
            for p in module.parameters():
                p.requires_grad = False
            return False

        final_w = W_pruned.t() if is_conv1d else W_pruned
        base_mod.weight.data = final_w.to(dtype=base_mod.weight.dtype, device=base_mod.weight.device)
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
        return True

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Executes SparseGPT compression across all target linear layers."""
        self.layer_errors = {}
        activations_map = self._collect_layer_activations()
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                act = activations_map.get(name, None)
                if self._compress_linear_module(name, module, act):
                    pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "layer_errors": self.layer_errors,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "max_error": self.max_error,
            "method": "sparsegpt_second_order",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("sparsegpt")
def sparsegpt_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Factory function for SparseGPT compressor."""
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "sparsegpt_sparsity", _cfg_get(comp_config, "sparsity", 0.5)))
    structured_24 = kwargs.pop("structured_24", _cfg_get(comp_config, "sparsegpt_structured_24", _cfg_get(comp_config, "structured_24", False)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.5))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None))

    compressor = SparseGPTCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SparseGPTCompressor",
    "sparsegpt_factory",
]
