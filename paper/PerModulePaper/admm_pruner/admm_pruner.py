"""
ADMM Pruner: Alternating Direction Method of Multipliers for Structured & Semi-Structured Pruning
with Layer-Wise Reconstruction Distillation, STE, and Dynamic Rho.
Registered in COMPRESSOR_REGISTRY as 'admm_pruner' and 'admm_pruner_distill'.

Mathematical Formulation:
$$\\min_W \\frac{1}{2} \\|W - W_0\\|_F^2 + \\lambda \\cdot \\mathbb{E}_{x \\sim \\mathcal{D}} \\left[ \\| \\text{layer}(x; W, \\text{mask}) - \\text{layer}(x; W_0) \\|_2^2 \\right] + g(Z) \\quad \\text{s.t.} \\quad W = Z$$

ADMM Updates with Dynamic Rho Scheduling & Distillation Gradient Step:
1. $Z^{(t+1)} = \\Pi_{\\mathcal{S}}(W^{(t)} + U^{(t)})$
2. $W^{(t+1/2)} = \\frac{1}{1 + \\rho^{(t)}} \\left( W_0 + \\rho^{(t)} (Z^{(t+1)} - U^{(t)}) \\right)$
3. Distillation step on $W$: $W^{(t+1)} = W^{(t+1/2)} - \\eta \\cdot \\lambda_{\\text{distill}} \\nabla_W (\\text{MSE}_{\\text{distill}})$
4. $U^{(t+1)} = U^{(t)} + W^{(t+1)} - Z^{(t+1)}$
5. Dynamic Rho Scheduling based on primal/dual residuals:
   $$\\rho^{(t+1)} = \\begin{cases} \\rho^{(t)} \\cdot \\tau_{\\text{incr}} & \\text{if } \\|r^{(t+1)}\\| > \\mu \\|s^{(t+1)}\\| \\\\ \\rho^{(t)} / \\tau_{\\text{decr}} & \\text{if } \\|s^{(t+1)}\\| > \\mu \\|r^{(t+1)}\\| \\\\ \\rho^{(t)} & \\text{otherwise} \\end{cases}$$
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
    recalibrate_layernorms,
)
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ADMMSTEPrune(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) autograd function for ADMM sparse weights.
    Forwards masked weights and passes gradients through uninhibited on backward pass.
    """

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(mask)
        return weight * mask

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (mask,) = ctx.saved_tensors
        grad_weight = grad_output.clone()
        return grad_weight, None


class ADMMMaskedLinear(nn.Module):
    """
    Linear layer maintaining continuous weights and an ADMM-derived binary mask with STE gradient backprop.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.register_buffer("mask", torch.ones(out_features, in_features, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, mask: torch.Tensor) -> ADMMMaskedLinear:
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        layer.weight.data.copy_(linear.weight.data)
        layer.mask.copy_(mask.to(dtype=linear.weight.dtype, device=linear.weight.device))
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_masked = ADMMSTEPrune.apply(self.weight, self.mask)
        return F.linear(x, w_masked, self.bias)


@COMPRESSOR_REGISTRY.register("admm_pruner")
@COMPRESSOR_REGISTRY.register("admm_pruner_distill")
class ADMMPruner(BaseCompressor):
    """
    Production-Hardened ADMM Pruner with Layer-Wise Reconstruction Distillation,
    2:4 semi-structured tensor core acceleration, STE gradients, and dynamic rho scheduling.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.45).
    - Safe fallback to original weights when error exceeds threshold with fallback layer freezing.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Robust LoRALinear support: operates on base_layer.weight, preserves LoRA adapters intact and trainable.
    - target_attention_only flag: filters out MLP/FFN layers and targets attention projections.
    - Layer-Wise Reconstruction Distillation: captures input calibration activations and optimizes W
      to minimize MSE output deviation ||layer(x; W*mask) - layer(x; W0)||^2.
    - FP32 numerical stability with NaN/Inf guards and best-iterate tracking.
    - Comprehensive per-layer statistics (actual_sparsity, distill_loss, reconstruction error).
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        rho: float = 1e-3,
        max_iter: int = 6,
        structured_24: bool = True,
        block_size: Tuple[int, int] = (4, 4),
        dynamic_rho: bool = True,
        use_ste: bool = True,
        recalibrate_ln: bool = True,
        max_error: float = 0.45,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        lambda_distill: float = 0.1,
        distill_batch_size: int = 32,
        distill_use_activation: bool = False,
        distill_lr: float = 0.1,
        distill_steps: int = 2,
        calibration_inputs: Optional[Dict[str, torch.Tensor]] = None,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="admm_pruner", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))

        self.sparsity = max(
            0.0,
            min(
                float(_cfg_get(cfg, "admm_pruner_sparsity", _cfg_get(cfg, "admm_sparsity", _cfg_get(cfg, "sparsity", sparsity)))),
                0.95,
            ),
        )
        self.initial_rho = float(_cfg_get(cfg, "admm_pruner_rho", _cfg_get(cfg, "admm_rho", _cfg_get(cfg, "rho", rho))))
        self.max_iter = max(
            1,
            int(_cfg_get(cfg, "admm_pruner_max_iter", _cfg_get(cfg, "admm_max_iter", _cfg_get(cfg, "max_iter", max_iter)))),
        )
        self.structured_24 = bool(
            _cfg_get(cfg, "admm_pruner_structured_24", _cfg_get(cfg, "admm_structured_24", _cfg_get(cfg, "structured_24", structured_24)))
        )
        self.block_size = block_size
        self.dynamic_rho = bool(_cfg_get(cfg, "admm_pruner_dynamic_rho", _cfg_get(cfg, "dynamic_rho", dynamic_rho)))
        self.use_ste = bool(_cfg_get(cfg, "admm_pruner_use_ste", _cfg_get(cfg, "use_ste", use_ste)))
        self.recalibrate_ln = bool(_cfg_get(cfg, "admm_pruner_recalibrate_ln", _cfg_get(cfg, "recalibrate_ln", recalibrate_ln)))
        self.max_error = float(_cfg_get(cfg, "admm_pruner_max_error", _cfg_get(cfg, "admm_max_error", _cfg_get(cfg, "max_error", max_error))))
        self.fallback_to_original = bool(
            _cfg_get(cfg, "admm_pruner_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.protect_critical = bool(_cfg_get(cfg, "protect_critical", protect_critical))
        self.target_attention_only = bool(
            _cfg_get(cfg, "admm_pruner_target_attention_only", _cfg_get(cfg, "admm_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only)))
        )
        self.target_modules = target_modules

        # Distillation configuration
        self.lambda_distill = float(
            _cfg_get(cfg, "admm_pruner_lambda_distill", _cfg_get(cfg, "lambda_distill", lambda_distill))
        )
        self.distill_batch_size = int(
            _cfg_get(cfg, "admm_pruner_distill_batch_size", _cfg_get(cfg, "distill_batch_size", distill_batch_size))
        )
        self.distill_use_activation = bool(
            _cfg_get(cfg, "admm_pruner_distill_use_activation", _cfg_get(cfg, "distill_use_activation", distill_use_activation))
        )
        self.distill_lr = float(_cfg_get(cfg, "distill_lr", distill_lr))
        self.distill_steps = max(1, int(_cfg_get(cfg, "distill_steps", distill_steps)))

        self.calibration_inputs = calibration_inputs or {}
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.comp_config = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_sparsities: Dict[str, float] = {}
        self.layer_actual_sparsities: Dict[str, float] = {}
        self.layer_distill_losses: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Determines if a given module is eligible for ADMM pruning."""
        if isinstance(module, ADMMMaskedLinear):
            return False
        # Do not target inner submodules of LoRA or wrappers directly
        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        # Extract weight and ensure it is strictly a 2D matrix
        weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return False

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
            or "loralinear" in cls_name
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        # Skip classification / output heads
        if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score", "pooler", "pre_classifier"]):
            return False

        if self.target_attention_only:
            # Skip FFN / MLP layers
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            # RoBERTa layer.X.output.dense is FFN, while attention.output.dense is attention
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            # Match attention projections
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj", "query", "key", "value",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _collect_calibration_inputs(self, target_names: List[str]) -> Dict[str, torch.Tensor]:
        """Captures input activations to each target module during a calibration forward pass."""
        collected: Dict[str, List[torch.Tensor]] = {name: [] for name in target_names}

        # Check if pre-configured calibration inputs exist
        if self.calibration_inputs:
            for name in target_names:
                if name in self.calibration_inputs:
                    collected[name].append(self.calibration_inputs[name])

        has_pre_collected = any(len(v) > 0 for v in collected.values())
        if has_pre_collected:
            return {k: torch.cat(v, dim=0) for k, v in collected.items() if v}

        # If neither loader nor texts are present, return empty (will use synthetic fallback if needed)
        if self.calibration_loader is None and self.calibration_texts is None:
            return {}

        hooks = []
        name_to_mod = dict(self.model.named_modules())

        def make_hook(layer_name: str):
            def hook_fn(mod: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
                if not inp or inp[0] is None:
                    return
                cur_total = sum(t.shape[0] for t in collected[layer_name])
                if cur_total >= self.distill_batch_size:
                    return
                x = inp[0].detach()
                if x.ndim >= 2:
                    x_flat = x.reshape(-1, x.shape[-1]).float().cpu()
                    rem = self.distill_batch_size - cur_total
                    collected[layer_name].append(x_flat[:rem])
            return hook_fn

        for name in target_names:
            if name in name_to_mod:
                target_mod = getattr(name_to_mod[name], "base_layer", name_to_mod[name])
                hooks.append(target_mod.register_forward_hook(make_hook(name)))

        device = self.target_device
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                if self.calibration_loader is not None:
                    for batch in self.calibration_loader:
                        if isinstance(batch, dict):
                            batch = {
                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()
                            }
                            self.model(**batch)
                        elif isinstance(batch, (list, tuple)):
                            batch = [
                                v.to(device) if isinstance(v, torch.Tensor) else v
                                for v in batch
                            ]
                            self.model(*batch)
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch.to(device))
                        if all(sum(t.shape[0] for t in collected[n]) >= self.distill_batch_size for n in target_names):
                            break
                elif self.calibration_texts is not None and self.tokenizer is not None:
                    for text in self.calibration_texts[:4]:
                        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                        toks = {k: v.to(device) for k, v in toks.items()}
                        self.model(**toks)
        except Exception as e:
            logger.warning("ADMM calibration input collection encountered exception: %s", e)
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                self.model.train()

        result: Dict[str, torch.Tensor] = {}
        for name, tensor_list in collected.items():
            if tensor_list:
                result[name] = torch.cat(tensor_list, dim=0)
        return result

    def _project_sparse(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects continuous weights onto sparse constraint set and returns (pruned_tensor, binary_mask)."""
        if self.structured_24:
            pruned, mask = apply_24_structured_mask(tensor, saliency=torch.abs(tensor), dim=1)
            mask = (mask > 0.5).float()
            return pruned, mask
        elif self.block_size != (1, 1):
            pruned, mask = apply_block_mask(tensor, block_size=self.block_size, sparsity=self.sparsity)
            mask = (mask > 0.5).float()
            return pruned, mask
        else:
            flat = tensor.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (tensor.abs() >= thr).float()
            return tensor * mask, mask

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ADMM pruning with Layer-Wise Reconstruction Distillation, STE, and safe validation."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}
        self.layer_actual_sparsities = {}
        self.layer_distill_losses = {}

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("ADMMPruner: No compressible target modules found.")
            stats = self.get_compression_stats()
            self.stats = stats
            self.model.compression_stats = stats
            return self.model, stats

        target_names = [name for name, _ in targets]
        calib_inputs = self._collect_calibration_inputs(target_names) if self.lambda_distill > 0 else {}

        for name, module in targets:
            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(target_mod, "nf")
                or hasattr(module, "nf")
            )

            raw_w = target_mod.weight.data
            orig_shape = raw_w.shape
            device = raw_w.device
            dtype = raw_w.dtype

            # Standardize weight to [out_features, in_features] in FP32
            if is_conv1d:
                W0 = raw_w.t().float()
            else:
                W0 = raw_w.float()

            out_features, in_features = W0.shape
            w_norm = torch.norm(W0, p="fro").item()

            # Retrieve or generate calibration input activations for distillation
            X_calib = calib_inputs.get(name, None)
            if self.lambda_distill > 0:
                if X_calib is None or X_calib.numel() == 0:
                    # Synthetic calibration inputs if data loader was unavailable
                    torch.manual_seed(hash(name) % (2**31 - 1))
                    X_calib = torch.randn(min(32, self.distill_batch_size), in_features, device=device) / math.sqrt(in_features)
                else:
                    X_calib = X_calib.to(device=device, dtype=torch.float32)
                    if X_calib.ndim > 2:
                        X_calib = X_calib.reshape(-1, in_features)
                    if X_calib.shape[0] > self.distill_batch_size:
                        X_calib = X_calib[:self.distill_batch_size]

                # Pre-compute target uncompressed outputs: Y0 = X @ W0^T
                with torch.no_grad():
                    Y0 = X_calib @ W0.t()
            else:
                X_calib = None
                Y0 = None

            # Baseline 0-iteration projection candidate
            Z0, mask0 = self._project_sparse(W0)
            W_sparse_0 = W0 * mask0
            best_W_sparse = W_sparse_0
            best_mask = mask0
            base_w_err = float(torch.norm(W0 - W_sparse_0, p="fro").item() / max(w_norm, 1e-12))
            best_err = base_w_err
            best_score = base_w_err
            if self.lambda_distill > 0 and X_calib is not None and Y0 is not None:
                y0_norm = max(torch.norm(Y0, p="fro").item(), 1e-12)
                base_y_err = float(torch.norm((X_calib @ W_sparse_0.t()) - Y0, p="fro").item() / y0_norm)
                best_score = base_w_err + self.lambda_distill * base_y_err

            # ADMM State Initialization
            W = W0.clone()
            Z = Z0.clone()
            mask = mask0.clone()
            U = torch.zeros_like(W0)

            rho = self.initial_rho
            mu = 10.0
            tau_incr = 1.5
            tau_decr = 1.5
            rho_min, rho_max = 1e-4, 100.0
            final_distill_loss = 0.0

            try:
                for it in range(self.max_iter):
                    Z_prev = Z.clone()

                    # 1. Update Z (Sparse projection)
                    Z, mask = self._project_sparse(W + U)

                    # 2. Update W via closed-form ADMM
                    W = (W0 + rho * (Z - U)) / (1.0 + rho)

                    # 3. Layer-Wise Reconstruction Distillation step on W
                    if self.lambda_distill > 0 and X_calib is not None and Y0 is not None:
                        for _ in range(self.distill_steps):
                            if self.use_ste:
                                W_eff = ADMMSTEPrune.apply(W, mask)
                            else:
                                W_eff = W * mask

                            Y_pruned = X_calib @ W_eff.t()
                            E = Y_pruned - Y0  # [N, out_features]
                            N_pts, d_out = E.shape
                            grad_W = ((2.0 / max(1, N_pts * d_out)) * (E.t() @ X_calib)) * mask

                            grad_norm = torch.norm(grad_W)
                            if grad_norm > 1e-8:
                                # B17: Standard bounded gradient descent step without artificial scale distortion
                                lr_eff = self.distill_lr * self.lambda_distill
                                clip_coef = min(1.0, 1.0 / (grad_norm.item() + 1e-6))
                                step = lr_eff * (grad_W * clip_coef)
                                W = W - step
                            else:
                                break

                    # 4. Update Dual variable U
                    U = U + W - Z

                    # 5. Dynamic Rho Scheduling based on primal/dual residuals
                    if self.dynamic_rho:
                        primal_res = torch.norm(W - Z).item()
                        dual_res = (rho * torch.norm(Z - Z_prev)).item()
                        if primal_res > mu * (dual_res + 1e-8):
                            rho = min(rho * tau_incr, rho_max)
                        elif dual_res > mu * (primal_res + 1e-8):
                            rho = max(rho / tau_decr, rho_min)

                    # Check for NaN / Inf
                    if torch.isnan(W).any() or torch.isinf(W).any():
                        logger.warning("ADMMPruner: NaN/Inf detected at iter %d on %s", it, name)
                        break

                    # Monotonic best-iterate tracking
                    W_sparse_curr = W * mask
                    curr_w_err = float(torch.norm(W0 - W_sparse_curr, p="fro").item() / max(w_norm, 1e-12))
                    curr_score = curr_w_err
                    if self.lambda_distill > 0 and X_calib is not None and Y0 is not None:
                        curr_y_err = float(torch.norm((X_calib @ W_sparse_curr.t()) - Y0, p="fro").item() / y0_norm)
                        curr_score = curr_w_err + self.lambda_distill * curr_y_err

                    if curr_score < best_score:
                        best_score = curr_score
                        best_err = curr_w_err
                        best_W_sparse = W_sparse_curr
                        best_mask = mask

                # Compute final distillation loss on best sparse candidate
                if self.lambda_distill > 0 and X_calib is not None and Y0 is not None:
                    with torch.no_grad():
                        final_distill_loss = float(torch.mean(((X_calib @ best_W_sparse.t()) - Y0) ** 2).item())

            except Exception as e:
                logger.warning("ADMMPruner optimization failed on %s: %s", name, e)
                best_W_sparse = None
                best_err = 1.0

            final_err = best_err
            W_final = best_W_sparse
            self.layer_errors[name] = final_err
            self.layer_sparsities[name] = 0.5 if self.structured_24 else self.sparsity
            self.layer_distill_losses[name] = final_distill_loss

            if (
                W_final is None
                or torch.isnan(W_final).any()
                or torch.isinf(W_final).any()
                or final_err > self.max_error
            ):
                logger.warning(
                    "ADMMPruner reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping ADMM pruning for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    # Freeze fallback layers to ensure recovery fine-tuning does not diverge
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Transpose back if Conv1D
            if is_conv1d:
                res_w = W_final.t().contiguous().to(dtype=dtype, device=device)
            else:
                res_w = W_final.contiguous().to(dtype=dtype, device=device)

            if res_w.shape != orig_shape:
                logger.warning("Shape mismatch on %s: %s vs %s. Skipping.", name, res_w.shape, orig_shape)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

            # Respect use_ste flag: use ADMMMaskedLinear when True on standard nn.Linear
            if self.use_ste and not has_lora and not is_conv1d and isinstance(module, nn.Linear):
                admm_layer = ADMMMaskedLinear.from_linear(module, mask=best_mask)
                replace_module(self.model, name, admm_layer)
            else:
                target_mod.weight.data.copy_(res_w)

            # Record actual zero fraction
            actual_zero_frac = float((res_w == 0).sum().item() / max(1, res_w.numel()))
            self.layer_actual_sparsities[name] = actual_zero_frac

            # Preserve LoRA adapter parameters
            if has_lora:
                if hasattr(target_mod, "weight") and target_mod.weight is not None:
                    target_mod.weight.requires_grad = False
                if hasattr(target_mod, "bias") and target_mod.bias is not None:
                    target_mod.bias.requires_grad = False
                for p_name, p in module.named_parameters():
                    if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                        p.requires_grad = True

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
                logger.warning("ADMM LayerNorm recalibration skipped: %s", e)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        avg_actual_sparsity = (
            sum(self.layer_actual_sparsities[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else (0.5 if self.structured_24 else self.sparsity)
        )
        avg_distill_loss = (
            sum(self.layer_distill_losses[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
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
            "layer_distill_losses": self.layer_distill_losses,
            "avg_reconstruction_error": avg_err,
            "avg_distill_loss": avg_distill_loss,
            "sparsity_target": self.sparsity,
            "actual_sparsity": avg_actual_sparsity,
            "layer_actual_sparsities": self.layer_actual_sparsities,
            "structured_24": self.structured_24,
            "admm_iterations": self.max_iter,
            "dynamic_rho": self.dynamic_rho,
            "use_ste": self.use_ste,
            "recalibrate_ln": self.recalibrate_ln,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "lambda_distill": self.lambda_distill,
            "distill_batch_size": self.distill_batch_size,
            "distill_use_activation": self.distill_use_activation,
            "method": "admm_pruner_distill_ste_dynamic_rho",
        })
        self.stats = stats
        self.model.compression_stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("admm_pruner_factory")
def admm_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ADMM pruner with Layer-Wise Reconstruction Distillation."""
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "admm_pruner_sparsity", _cfg_get(comp_config, "admm_sparsity", _cfg_get(comp_config, "sparsity", 0.5))),
    )
    rho = kwargs.pop(
        "rho",
        _cfg_get(comp_config, "admm_pruner_rho", _cfg_get(comp_config, "admm_rho", _cfg_get(comp_config, "rho", 0.1))),
    )
    max_iter = kwargs.pop(
        "max_iter",
        _cfg_get(comp_config, "admm_pruner_max_iter", _cfg_get(comp_config, "admm_max_iter", _cfg_get(comp_config, "max_iter", 6))),
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(comp_config, "admm_pruner_structured_24", _cfg_get(comp_config, "admm_structured_24", _cfg_get(comp_config, "structured_24", True))),
    )
    dynamic_rho = kwargs.pop(
        "dynamic_rho",
        _cfg_get(comp_config, "admm_pruner_dynamic_rho", _cfg_get(comp_config, "dynamic_rho", True)),
    )
    use_ste = kwargs.pop(
        "use_ste",
        _cfg_get(comp_config, "admm_pruner_use_ste", _cfg_get(comp_config, "use_ste", True)),
    )
    recalibrate_ln = kwargs.pop(
        "recalibrate_ln",
        _cfg_get(comp_config, "admm_pruner_recalibrate_ln", _cfg_get(comp_config, "recalibrate_ln", True)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "admm_pruner_max_error", _cfg_get(comp_config, "admm_max_error", _cfg_get(comp_config, "max_error", 0.45))),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "admm_pruner_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "admm_pruner_target_attention_only", _cfg_get(comp_config, "admm_target_attention_only", _cfg_get(comp_config, "target_attention_only", True))),
    )
    lambda_distill = kwargs.pop(
        "lambda_distill",
        _cfg_get(comp_config, "admm_pruner_lambda_distill", _cfg_get(comp_config, "lambda_distill", 0.1)),
    )
    distill_batch_size = kwargs.pop(
        "distill_batch_size",
        _cfg_get(comp_config, "admm_pruner_distill_batch_size", _cfg_get(comp_config, "distill_batch_size", 32)),
    )
    distill_use_activation = kwargs.pop(
        "distill_use_activation",
        _cfg_get(comp_config, "admm_pruner_distill_use_activation", _cfg_get(comp_config, "distill_use_activation", False)),
    )

    compressor = ADMMPruner(
        model,
        sparsity=sparsity,
        rho=rho,
        max_iter=max_iter,
        structured_24=structured_24,
        dynamic_rho=dynamic_rho,
        use_ste=use_ste,
        recalibrate_ln=recalibrate_ln,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        lambda_distill=lambda_distill,
        distill_batch_size=distill_batch_size,
        distill_use_activation=distill_use_activation,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "ADMMSTEPrune",
    "ADMMMaskedLinear",
    "ADMMPruner",
    "admm_pruner_factory",
]
