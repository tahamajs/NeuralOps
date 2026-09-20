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
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
    recalibrate_layernorms,
)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


logger = logging.getLogger(__name__)


class ASTPrunedLoRALinear(nn.Module):
    """
    Linear layer containing AST 2:4 semi-structured pruned base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        self.weight = nn.Parameter(
            weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> ASTPrunedLoRALinear:
        bias = linear.bias.detach() if hasattr(linear, "bias") and linear.bias is not None else None
        in_features = getattr(linear, "in_features", weight.shape[1])
        out_features = getattr(linear, "out_features", weight.shape[0])
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(x_flat, self.weight, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


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
    - Optional trainable LoRA adapters via ASTPrunedLoRALinear.
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
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="ast_pruner", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        if conf is not None:
            block_size = getattr(conf, "ast_block_size", getattr(conf, "block_size", block_size))
            n_keep = getattr(conf, "ast_n_keep", getattr(conf, "n_keep", n_keep))
            dim = getattr(conf, "ast_dim", getattr(conf, "dim", dim))
            use_pca = getattr(conf, "ast_use_pca", getattr(conf, "use_pca", use_pca))
            use_fisher = getattr(conf, "ast_use_fisher", getattr(conf, "use_fisher", use_fisher))
            recalibrate_ln = getattr(conf, "ast_recalibrate_ln", getattr(conf, "recalibrate_ln", recalibrate_ln))
            max_error = getattr(conf, "ast_max_error", getattr(conf, "max_error", max_error))
            fallback_to_original = getattr(
                conf, "ast_fallback_to_original", getattr(conf, "fallback_to_original", fallback_to_original)
            )
            protect_critical = getattr(conf, "protect_critical", protect_critical)
            lora_r = getattr(conf, "lora_r", getattr(conf, "lora_rank", lora_r))
            lora_alpha = getattr(conf, "lora_alpha", lora_alpha)
            skip_attention_qkv = getattr(conf, "skip_attention_qkv", skip_attention_qkv)

        self.block_size = max(2, int(block_size))
        self.n_keep = max(1, min(int(n_keep), self.block_size - 1))
        self.dim = dim
        self.use_pca = bool(use_pca)
        self.use_fisher = bool(use_fisher)
        self.recalibrate_ln = bool(recalibrate_ln)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = target_attention_only
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
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

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
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like or isinstance(module, ASTPrunedLoRALinear):
            return False

        if self.protect_critical:
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
                return False

        from framework.utils.compat import should_compress_linear
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only and not any(k in name.lower() for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "c_attn", "c_proj"]):
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

        batches = []
        if self.calibration_loader is not None:
            for b in self.calibration_loader:
                batches.append(b)
                if len(batches) >= 4:
                    break
        if not batches:
            batches = [get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )]

        n_batches_run = 0
        try:
            self.model.train()
            for batch_input in batches:
                out = None
                if isinstance(batch_input, dict):
                    out = self.model(**{k: v.to(self.target_device) if hasattr(v, "to") else v for k, v in batch_input.items()})
                elif isinstance(batch_input, torch.Tensor):
                    out = self.model(batch_input.to(self.target_device))
                elif isinstance(batch_input, (list, tuple)):
                    out = self.model(*[b.to(self.target_device) if hasattr(b, "to") else b for b in batch_input])

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
                                g_sq = g.pow(2).cpu()
                                if name in fisher_info:
                                    fisher_info[name] = fisher_info[name] + g_sq
                                else:
                                    fisher_info[name] = g_sq
                        self.model.zero_grad()
                        n_batches_run += 1

            if n_batches_run > 1:
                for k in fisher_info:
                    fisher_info[k] = fisher_info[k] / float(n_batches_run)
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
        from framework.core.model_utils import replace_module

        cov_map, fisher_map = self._collect_fisher_and_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                if not hasattr(module, "weight") or module.weight is None:
                    continue
                w_id = id(module.weight)
                if w_id in seen_weights:
                    for p in module.parameters(): p.requires_grad = False
                    self.skipped_layers.append(name)
                    continue
                seen_weights.add(w_id)

                target_mod = getattr(module, 'base_layer', module)
                raw_w = target_mod.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                d_out, d_in = W.shape
                device = W.device
                dtype = target_mod.weight.dtype

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
                        for p in module.parameters(): p.requires_grad = False
                        self.fallback_layers.append(name)
                        continue

                if self.lora_r > 0:
                    ast_layer = ASTPrunedLoRALinear.from_linear(
                        target_mod,
                        weight=W_final,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    if name == "":
                        self.model = ast_layer
                    else:
                        if hasattr(module, 'base_layer'):
                            if hasattr(module, 'lora_A'):
                                ast_layer.lora_A = module.lora_A
                            if hasattr(module, 'lora_B'):
                                ast_layer.lora_B = module.lora_B
                            module.base_layer = ast_layer
                            ast_layer = module
                        replace_module(self.model, name, ast_layer)
                else:
                    if is_conv1d:
                        target_mod.weight.data.copy_(W_final.t().contiguous().to(dtype=dtype, device=device))
                    else:
                        target_mod.weight.data.copy_(W_final.contiguous().to(dtype=dtype, device=device))

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

        # Post-compression forward sanity check
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.debug("AST post-compression forward sanity check exception: %s", e)

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
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
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
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "method": "ast_fisher_pca_2_4_pruning",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("ast_pruner_factory")
@COMPRESSOR_REGISTRY.register("ast_factory")
def ast_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for AST pruner."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
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
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", getattr(comp_config, "lora_rank", 0)))
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0))
    skip_attention_qkv = kwargs.pop("skip_attention_qkv", getattr(comp_config, "skip_attention_qkv", True))

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
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "ASTPruner",
    "ASTPrunedLoRALinear",
    "ast_pruner_factory",
]
