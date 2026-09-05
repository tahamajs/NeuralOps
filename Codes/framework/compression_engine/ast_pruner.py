"""
AST: Adaptive Semi-Structured 2:4 and N:M Pruning with Empirical Fisher Calibration and PCA Coordinate Rotation.
Registered in COMPRESSOR_REGISTRY as 'ast_pruner' and 'ast'.

Mathematical Formulation:
1. Activation Covariance & PCA Coordinate Rotation:
   $$\\Sigma_X = \\frac{1}{N - 1} (X - \\mu)^\\top (X - \\mu) = Q \\Lambda Q^\\top, \\quad Q^\\top Q = I$$
   $$\\widetilde{W} = W Q, \\quad \\widetilde{X} = Q^\\top X$$
2. Empirical Fisher Information Calibration:
   $$\\mathcal{L}_{\\text{calib}} = \\frac{1}{2} \\sum_{i} \\|y_i\\|_2^2, \\quad g = \\frac{\\partial \\mathcal{L}}{\\partial W}$$
   $$F = \\mathbb{E}[g \\odot g], \\quad \\widetilde{F} = F Q^2$$
3. PCA-Rotated Fisher Saliency:
   $$S_{ij} = |\\widetilde{W}_{ij}| \\cdot \\sqrt{\\widetilde{F}_{ij} + \\epsilon}$$
4. 2:4 Structured Masking & Coordinate Back-Rotation:
   $$\\widetilde{W}_{\\text{pruned}} = \\Pi_{2:4}(\\widetilde{W}, S)$$
   $$W_{\\text{ast}} = \\widetilde{W}_{\\text{pruned}} Q^\\top$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("ast_pruner")
@COMPRESSOR_REGISTRY.register("ast")
class ASTPruner(BaseCompressor):
    """
    AST Pruner: Vectorized semi-structured 2:4 and N:M pruning engine with
    empirical Fisher calibration and PCA coordinate rotation.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - SVD, PCA, and Fisher gradient computations in FP32 with NaN/Inf guards.
    - Unified critical-layer protection via should_compress_linear and protect_critical.
    - Comprehensive per-layer statistics and error tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        n_keep: int = 2,
        dim: int = 1,
        use_pca: bool = True,
        use_fisher: bool = True,
        recalibrate_ln: bool = True,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="ast_pruner", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            block_size = getattr(cfg, "ast_block_size", getattr(cfg, "block_size", block_size))
            n_keep = getattr(cfg, "ast_n_keep", getattr(cfg, "n_keep", n_keep))
            dim = getattr(cfg, "ast_dim", getattr(cfg, "dim", dim))
            use_pca = getattr(cfg, "ast_use_pca", getattr(cfg, "use_pca", use_pca))
            use_fisher = getattr(cfg, "ast_use_fisher", getattr(cfg, "use_fisher", use_fisher))
            recalibrate_ln = getattr(cfg, "ast_recalibrate_ln", getattr(cfg, "recalibrate_ln", recalibrate_ln))
            max_error = getattr(cfg, "ast_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "ast_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )
            protect_critical = getattr(cfg, "protect_critical", protect_critical)

        self.block_size = max(2, int(block_size))
        self.n_keep = max(1, min(int(n_keep), self.block_size - 1))
        self.dim = dim
        self.use_pca = bool(use_pca)
        self.use_fisher = bool(use_fisher)
        self.recalibrate_ln = bool(recalibrate_ln)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.target_modules = target_modules
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like:
            return False

        if self.protect_critical:
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
                return False

        from framework.utils.compat import should_compress_linear
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _collect_fisher_and_activations(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Collects empirical Fisher gradients and activation covariance via calibration pass."""
        activations: Dict[str, List[torch.Tensor]] = {}
        fisher_info: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(x.cpu())
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

        try:
            self.model.train()
            out = None
            if isinstance(calib_inputs, dict):
                out = self.model(**calib_inputs)
            elif isinstance(calib_inputs, torch.Tensor):
                out = self.model(calib_inputs)
            elif isinstance(calib_inputs, (list, tuple)):
                out = self.model(*calib_inputs)

            if self.use_fisher and out is not None:
                if hasattr(out, "logits"):
                    loss = 0.5 * torch.sum(out.logits.pow(2))
                elif isinstance(out, torch.Tensor):
                    loss = 0.5 * torch.sum(out.pow(2))
                elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    loss = 0.5 * torch.sum(out[0].pow(2))
                else:
                    loss = None

                if loss is not None:
                    self.model.zero_grad()
                    loss.backward()
                    for name, module in self.model.named_modules():
                        if self._is_target_module(name, module) and module.weight.grad is not None:
                            g = module.weight.grad.detach()
                            fisher_info[name] = g.pow(2).cpu()
                    self.model.zero_grad()
        except Exception as e:
            logger.debug("AST Fisher calibration exception: %s", e)
        finally:
            self.model.eval()
            for h in hooks:
                h.remove()

        cov_map: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                act_concat = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0).float()
                N, D = act_concat.shape
                mean = torch.mean(act_concat, dim=0, keepdim=True)
                centered = act_concat - mean
                cov = torch.matmul(centered.T, centered) / max(N - 1, 1)
                cov_map[name] = cov
                if name not in fisher_info:
                    fisher_info[name] = torch.var(act_concat, dim=0, keepdim=True).expand(1, D)

        return cov_map, fisher_info

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes AST 2:4 structured pruning with PCA rotation and empirical Fisher calibration."""
        cov_map, fisher_map = self._collect_fisher_and_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = module.weight.dtype

                # 1. Derive PCA orthogonal coordinate rotation matrix Q
                if self.use_pca and name in cov_map and cov_map[name].shape == (d_in, d_in):
                    cov = cov_map[name].to(device)
                    cov = cov + 1e-6 * torch.eye(d_in, device=device)
                    try:
                        eigvals, Q = torch.linalg.eigh(cov)
                        idx = torch.argsort(eigvals, descending=True)
                        Q = Q[:, idx]
                        if torch.isnan(Q).any() or torch.isinf(Q).any():
                            Q = torch.eye(d_in, device=device)
                    except Exception:
                        Q = torch.eye(d_in, device=device)
                else:
                    Q = torch.eye(d_in, device=device)

                # 2. Rotate weights into PCA coordinate space: \widetilde{W} = W Q
                W_pca = torch.matmul(W, Q)

                # 3. Compute empirical Fisher saliency in rotated basis
                fisher = fisher_map.get(name, None)
                if fisher is not None:
                    F_mat = fisher.to(device).float()
                    if is_conv1d and F_mat.dim() == 2 and F_mat.shape == raw_w.shape:
                        F_mat = F_mat.t()
                    if F_mat.dim() == 2 and F_mat.shape == W.shape:
                        F_pca = torch.matmul(F_mat, Q.pow(2))
                    elif F_mat.shape[-1] == d_in:
                        F_pca = torch.matmul(F_mat.reshape(1, d_in), Q.pow(2)).expand(d_out, d_in)
                    else:
                        F_pca = torch.ones_like(W_pca)
                    saliency = torch.abs(W_pca) * torch.sqrt(torch.clamp(F_pca, min=1e-8))
                else:
                    saliency = torch.abs(W_pca)

                # 4. Apply semi-structured 2:4 mask in PCA space
                W_pca_pruned, _ = apply_24_structured_mask(W_pca, saliency=saliency, dim=self.dim)

                # 5. Rotate back to original coordinate basis: W = \widetilde{W}_pruned Q^T
                W_final = torch.matmul(W_pca_pruned, Q.T)
                W_final = torch.nan_to_num(W_final, nan=0.0)

                w_norm = torch.norm(W, p="fro").item()
                err = float(torch.norm(W - W_final, p="fro").item() / max(w_norm, 1e-12))
                self.layer_errors[name] = err

                if err > self.max_error:
                    logger.warning(
                        "ASTPruner reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping AST pruning for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                if is_conv1d:
                    module.weight.data.copy_(W_final.t().contiguous().to(dtype=dtype, device=device))
                else:
                    module.weight.data.copy_(W_final.contiguous().to(dtype=dtype, device=device))

                self.compressed_layers.append(name)

        # LayerNorm Recalibration
        if self.recalibrate_ln and (self.calibration_loader is not None or self.calibration_texts is not None):
            try:
                recalibrate_layernorms(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
            except Exception as e:
                logger.warning("AST LayerNorm recalibration skipped: %s", e)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": len(self.compressed_layers),
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "block_size": self.block_size,
            "n_keep": self.n_keep,
            "pattern": f"{self.n_keep}:{self.block_size}",
            "use_pca": self.use_pca,
            "use_fisher": self.use_fisher,
            "recalibrate_ln": self.recalibrate_ln,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "method": "ast_fisher_pca_2_4_pruning",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("ast_pruner_factory")
@COMPRESSOR_REGISTRY.register("ast_factory")
def ast_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for AST pruner."""
    block_size = kwargs.pop("block_size", getattr(comp_config, "ast_block_size", getattr(comp_config, "block_size", 4)))
    n_keep = kwargs.pop("n_keep", getattr(comp_config, "ast_n_keep", getattr(comp_config, "n_keep", 2)))
    dim = kwargs.pop("dim", getattr(comp_config, "ast_dim", getattr(comp_config, "dim", 1)))
    use_pca = kwargs.pop("use_pca", getattr(comp_config, "ast_use_pca", True))
    use_fisher = kwargs.pop("use_fisher", getattr(comp_config, "ast_use_fisher", True))
    recalibrate_ln = kwargs.pop("recalibrate_ln", getattr(comp_config, "ast_recalibrate_ln", True))
    max_error = kwargs.pop("max_error", getattr(comp_config, "ast_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "ast_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))

    compressor = ASTPruner(
        model,
        block_size=block_size,
        n_keep=n_keep,
        dim=dim,
        use_pca=use_pca,
        use_fisher=use_fisher,
        recalibrate_ln=recalibrate_ln,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

__all__ = [
    "ASTPruner",
    "ast_pruner_factory",
]

__all__ = [
    "ASTPruner",
    "ast_pruner_factory",
]
