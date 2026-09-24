"""
ARMOR: Adaptive Resilient Matrix Orthogonal Rank Pruning (2:4 Sparse Core + Low-Rank Correction).
Registered in COMPRESSOR_REGISTRY as 'armor_pruner', 'armor', and 'armor_compressor'.

Bug-fix map:
  AR-FIX-01  scaling computed from actual lora_A rank.
  AR-FIX-02  sparse_mask is non-persistent (regenerable).
  AR-FIX-03  forward cast dtype-safe for scaling.
  AR-FIX-04  apply_24_structured_mask mask handling robust (bool or float).
  AR-FIX-05  Conv1D detection uses class-name rather than nf-only.
  AR-FIX-06  finite-check via torch.isfinite().all() for speed.
  AR-FIX-07  seen_weights applied to the actual weight tensor.
  AR-FIX-08  recalibrate_layernorms guarded by presence of calibration data.
  AR-FIX-09  weight property dtype-consistent.
  AR-FIX-10  factory validates rank <= min(in, out).
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
from framework.core.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
    recalibrate_layernorms,
)
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# Saliency
# ===========================================================================

def compute_resilient_taylor_saliency(
    weight: torch.Tensor,
    grad: Optional[torch.Tensor] = None,
    activations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Resilient Taylor saliency with three-tier fallback:
      1. |W · g| if gradient is valid and nonzero.
      2. |W| · E[|x|] if activations available.
      3. |W| · (1 + σ(W)) as fallback.
    """
    W = weight.float()

    if grad is not None and torch.is_tensor(grad):
        g = grad.float()
        if torch.isfinite(g).all() and torch.count_nonzero(g) > 0:
            return torch.abs(W * g)

    if activations is not None and torch.is_tensor(activations) and activations.numel() > 0:
        X = activations.reshape(-1, activations.shape[-1]).float()
        if X.shape[-1] == W.shape[1]:
            act_scale = torch.mean(torch.abs(X), dim=0, keepdim=True).to(W.device)
            act_scale = torch.nan_to_num(act_scale, nan=1.0)
            return torch.abs(W) * act_scale

    w_std = torch.std(W).item() if W.numel() > 1 else 1.0
    if not math.isfinite(w_std) or w_std <= 0.0:
        w_std = 1.0
    return torch.abs(W) * (1.0 + w_std)


# ===========================================================================
# ARMORLinear
# ===========================================================================

class ARMORLinear(nn.Module):
    """
    Decoupled sparse + low-rank residual layer.

    Forward: y = x @ (sparse_weight ⊙ sparse_mask)^T
                + (scaling) * (x @ lora_A^T) @ lora_B^T
                + bias
    """
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.rank) if self.rank > 0 else 1.0

        self.sparse_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        # AR-FIX-02: non-persistent mask (regenerable from sparse_weight)
        self.register_buffer(
            "sparse_mask",
            torch.ones(out_features, in_features, dtype=dtype, device=device),
            persistent=False,
        )

        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        mask: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        lora_alpha: float = 1.0,
    ) -> "ARMORLinear":
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            w_data = raw_w.t().contiguous()
        else:
            out_f, in_f = raw_w.shape
            w_data = raw_w

        # AR-FIX-01/10: rank = actual lora_A rows, clamped to (in, out).
        requested_rank = int(lora_A.shape[0])
        actual_rank = max(1, min(requested_rank, in_f, out_f))

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=requested_rank,
            bias=(getattr(linear, "bias", None) is not None and linear.bias is not None),
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )
        mask_dev = mask.to(dtype=w_data.dtype, device=w_data.device)
        with torch.no_grad():
            layer.sparse_weight.data.copy_(w_data * mask_dev)
            layer.sparse_mask.copy_(mask_dev)
            layer.lora_A.data.copy_(lora_A[:actual_rank].to(dtype=raw_w.dtype, device=raw_w.device))
            layer.lora_B.data.copy_(lora_B[:, :actual_rank].to(dtype=raw_w.dtype, device=raw_w.device))
            if getattr(linear, "bias", None) is not None and linear.bias is not None and layer.bias is not None:
                layer.bias.data.copy_(linear.bias.data)
        return layer

    @property
    def weight(self) -> torch.Tensor:
        # AR-FIX-09: dtype-consistent.
        base = self.sparse_weight * self.sparse_mask
        if self.lora_A is not None and self.lora_B is not None and self.rank > 0:
            lora_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            return base + lora_w.to(base.dtype)
        return base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # 1. Sparse core.
        w_eff = self.sparse_weight * self.sparse_mask
        out_sparse = F.linear(x_flat, w_eff)

        # 2. Low-rank residual.
        lr_dtype = self.lora_A.dtype
        x_lr = x_flat.to(lr_dtype) if x_flat.dtype != lr_dtype else x_flat
        out_lr = F.linear(F.linear(x_lr, self.lora_A), self.lora_B)
        # AR-FIX-03: explicit dtype of scaling.
        out_lr = (out_lr * self.scaling).to(out_sparse.dtype)

        out = out_sparse + out_lr
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.view(*orig_shape[:-1], self.out_features)


# ===========================================================================
# ARMOR Pruner
# ===========================================================================

@COMPRESSOR_REGISTRY.register("armor_pruner")
@COMPRESSOR_REGISTRY.register("armor")
@COMPRESSOR_REGISTRY.register("armor_compressor")
class ARMORPruner(BaseCompressor):
    is_custom_attention: bool = True
    """
    Production-hardened ARMOR Pruner.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        rank: int = 16,
        sparsity: float = 0.5,
        structured_24: bool = True,
        decouple: bool = True,
        recalibrate_ln: bool = True,
        lora_alpha: float = 1.0,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        skip_attention_qkv: bool = False,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="armor_pruner", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)

        self.block_size = int(_cfg_get(conf, "armor_block_size",
                                       _cfg_get(conf, "block_size", block_size)))
        self.rank = max(1, int(_cfg_get(conf, "armor_rank",
                                        _cfg_get(conf, "rank", rank))))
        self.sparsity = max(0.0, min(float(_cfg_get(conf, "armor_sparsity",
                                            _cfg_get(conf, "sparsity", sparsity))), 0.95))
        self.structured_24 = bool(_cfg_get(conf, "armor_structured_24",
                                            _cfg_get(conf, "structured_24", structured_24)))
        self.decouple = bool(_cfg_get(conf, "armor_decouple",
                                       _cfg_get(conf, "decouple", decouple)))
        self.recalibrate_ln = bool(_cfg_get(conf, "armor_recalibrate_ln",
                                             _cfg_get(conf, "recalibrate_ln", recalibrate_ln)))
        self.lora_alpha = float(_cfg_get(conf, "armor_lora_alpha",
                                          _cfg_get(conf, "lora_alpha", lora_alpha)))
        self.max_error = float(_cfg_get(conf, "armor_max_error",
                                         _cfg_get(conf, "max_error", max_error)))
        self.fallback_to_original = bool(_cfg_get(conf, "armor_fallback_to_original",
                                                    _cfg_get(conf, "fallback_to_original", fallback_to_original)))
        self.target_attention_only = bool(_cfg_get(conf, "armor_target_attention_only",
                                                    _cfg_get(conf, "target_attention_only", target_attention_only)))
        self.skip_attention_qkv = bool(_cfg_get(conf, "armor_skip_attention_qkv",
                                                 _cfg_get(conf, "skip_attention_qkv", skip_attention_qkv)))
        self.target_modules = target_modules

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device) if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.comp_config = conf
        self._compressed_params: int = 0

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_actual_sparsities: Dict[str, float] = {}
        self.layer_effective_ratios: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in
                   ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, ARMORLinear):
            return False
        if any(name.endswith(sfx) for sfx in
               [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()

        weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return False

        # AR-FIX-05: Conv1D detection via class name.
        target_cls = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in target_cls
        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or is_conv1d
            or (Conv1D is not None and (isinstance(module, Conv1D)
                                        or isinstance(target_mod, Conv1D)))
            or "loralinear" in cls_name
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in
               ["lm_head", "classifier", "head", "output_layer", "score",
                "pooler", "pre_classifier"]):
            return False

        if any(nested in name_lower for nested in
               ["attention.self.output", "self.output.dense"]):
            return False

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only:
            ffn_patterns = ["ffn", "mlp", "lin1", "lin2", "intermediate",
                            "gate_proj", "up_proj", "down_proj", "c_fc"]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj",
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inp, outp):
                if inp and isinstance(inp[0], torch.Tensor):
                    if layer_name not in activations:
                        activations[layer_name] = []
                    activations[layer_name].append(inp[0].detach().cpu())
            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
                hooks.append(target_mod.register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad():
                if self.calibration_loader is not None:
                    for batch in self.calibration_loader:
                        try:
                            if isinstance(batch, dict):
                                b_dev = {k: v.to(self.target_device) if torch.is_tensor(v) else v
                                         for k, v in batch.items()}
                                self.model(**b_dev)
                            elif torch.is_tensor(batch):
                                self.model(batch.to(self.target_device))
                            elif isinstance(batch, (list, tuple)):
                                b_dev = [v.to(self.target_device) if torch.is_tensor(v) else v
                                         for v in batch]
                                self.model(*b_dev)
                        except Exception as b_err:
                            logger.debug("ARMOR calibration batch failed: %s", b_err)
                else:
                    calib_inputs = get_calibration_tensors(
                        self.model,
                        tokenizer=self.tokenizer,
                        calibration_loader=None,
                        calibration_texts=self.calibration_texts,
                        device=self.target_device,
                    )
                    try:
                        if isinstance(calib_inputs, dict):
                            self.model(**calib_inputs)
                        elif isinstance(calib_inputs, torch.Tensor):
                            self.model(calib_inputs)
                        elif isinstance(calib_inputs, (list, tuple)):
                            self.model(*calib_inputs)
                    except Exception as e:
                        logger.debug("ARMOR calibration pass exception: %s", e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(
                    [a.reshape(-1, a.shape[-1]) for a in act_list], dim=0
                )
        return aggregated

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        act_map = self._collect_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_actual_sparsities = {}
        self.layer_effective_ratios = {}
        total_correction_energy = 0.0
        seen_weights: set = set()

        total_base_sparse = 0
        total_correction = 0
        total_orig = 0

        targets = [
            (name, mod) for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("ARMORPruner: No compressible target modules found.")
            stats = self.get_compression_stats()
            self.stats = stats
            self.model.compression_stats = stats
            return self.model, stats

        for name, module in targets:
            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            target_cls = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in target_cls
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) \
                       or hasattr(module, "base_layer")

            # AR-FIX-07: use the actual weight tensor id.
            weight_tensor = getattr(target_mod, "weight", None)
            if weight_tensor is None:
                self.skipped_layers.append(name)
                continue
            w_id = id(weight_tensor)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = weight_tensor.data
            device = raw_w.device
            dtype = raw_w.dtype

            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
            out_features, in_features = W.shape
            grad = getattr(weight_tensor, "grad", None)
            act = act_map.get(name, None)

            try:
                saliency = compute_resilient_taylor_saliency(W, grad=grad, activations=act)

                # AR-FIX-04: robust mask handling.
                if self.structured_24:
                    W_sparse, raw_mask = apply_24_structured_mask(W, saliency=saliency, dim=1)
                    if raw_mask is None:
                        # Derive mask from zero pattern.
                        mask = (W_sparse.abs() > 1e-12).float()
                    else:
                        # Could be bool, int, or signed float.
                        if raw_mask.dtype == torch.bool:
                            mask = raw_mask.float()
                        else:
                            mask = (raw_mask > 0.5).float()
                else:
                    flat = saliency.reshape(-1)
                    k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    mask = (saliency >= thr).float()
                    W_sparse = W * mask

                # Residual low-rank.
                residual = W - W_sparse
                r = min(self.rank, residual.shape[0], residual.shape[1])
                try:
                    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    lora_B = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
                    lora_A = (sqrt_S.unsqueeze(1) * Vh[:r, :]).to(dtype=dtype, device=device)
                    correction = torch.matmul(lora_B, lora_A)
                    total_correction_energy += torch.norm(correction).item()
                except Exception as e:
                    logger.debug("ARMOR residual SVD failed on %s: %s", name, e)
                    lora_A = torch.zeros(r, in_features, dtype=dtype, device=device)
                    lora_B = torch.zeros(out_features, r, dtype=dtype, device=device)

                scaling = (self.lora_alpha / r) if r > 0 else 1.0
                W_recon = W_sparse + scaling * torch.matmul(lora_B.float(), lora_A.float())

                # AR-FIX-06: fast finite check.
                if not torch.isfinite(W_recon).all():
                    raise RuntimeError("non-finite reconstruction")

                w_norm = torch.norm(W, p="fro").clamp_min(1e-8)
                err = float(torch.norm(W - W_recon, p="fro").item() / w_norm)
                self.layer_errors[name] = err

                if err > self.max_error and self.fallback_to_original:
                    logger.warning(
                        "ARMOR reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err, name, self.max_error,
                    )
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

                base_sp_count = int((mask > 0).sum().item())
                corr_count = int(lora_A.numel() + lora_B.numel()) if self.decouple else 0
                orig_count = int(W.numel())
                total_base_sparse += base_sp_count
                total_correction += corr_count
                total_orig += orig_count

                actual_zero_frac = float((mask == 0).sum().item() / max(1, orig_count))
                eff_ratio = float((base_sp_count + corr_count) / max(1, orig_count))
                self.layer_actual_sparsities[name] = actual_zero_frac
                self.layer_effective_ratios[name] = eff_ratio

                # Dual-branch or in-place.
                if self.decouple:
                    armor_layer = ARMORLinear.from_linear(
                        target_mod,
                        mask=mask,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        lora_alpha=self.lora_alpha,
                    )
                    if has_lora:
                        module.base_layer = armor_layer
                        for p_name, p in module.named_parameters():
                            if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                                if not p_name.startswith("base_layer"):
                                    with torch.no_grad():
                                        p.data.zero_()
                                    p.requires_grad = False
                    elif name == "":
                        self.model = armor_layer
                    else:
                        replace_module(self.model, name, armor_layer)
                else:
                    # AR-FIX: mutation under no_grad.
                    with torch.no_grad():
                        if is_conv1d:
                            weight_tensor.data.copy_(W_recon.t().contiguous().to(dtype=dtype, device=device))
                        else:
                            weight_tensor.data.copy_(W_recon.to(dtype=dtype, device=device))

                self.compressed_layers.append(name)

            except Exception as e:
                logger.warning("ARMOR pruning failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # AR-FIX-08: LayerNorm recalibration only when data available.
        if self.recalibrate_ln and (
            self.calibration_loader is not None or self.calibration_texts is not None
        ):
            try:
                recalibrate_layernorms(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
            except Exception as e:
                logger.warning("ARMOR LayerNorm recalibration skipped: %s", e)

        # Forward sanity check.
        forward_check_success = True
        try:
            self.model.eval()
            with torch.no_grad():
                first_param = next(self.model.parameters(), torch.tensor(0.0))
                dev = first_param.device
                p_dtype = first_param.dtype if first_param.dtype.is_floating_point else torch.float32
                cfg = getattr(self.model, "config", None)
                hidden_size = getattr(cfg, "hidden_size", None)
                try:
                    dummy_ids = torch.zeros((1, 8), dtype=torch.long, device=dev)
                    self.model(input_ids=dummy_ids, attention_mask=torch.ones_like(dummy_ids))
                except Exception:
                    if hidden_size is not None:
                        dummy_embeds = torch.randn(1, 8, hidden_size, dtype=p_dtype, device=dev)
                    else:
                        first_linear = next(
                            (m for m in self.model.modules()
                             if isinstance(m, (nn.Linear, ARMORLinear))), None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        dummy_embeds = torch.randn(1, 8, in_dim, dtype=p_dtype, device=dev)
                    self.model(dummy_embeds)
        except Exception as e:
            logger.debug("ARMOR forward sanity check failed: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        avg_actual_sparsity = (
            sum(self.layer_actual_sparsities.values()) / max(1, len(self.layer_actual_sparsities))
            if self.layer_actual_sparsities
            else (0.5 if self.structured_24 else self.sparsity)
        )
        effective_param_ratio = (
            float((total_base_sparse + total_correction) / max(1, total_orig))
            if total_orig > 0 else 1.0
        )

        stats.update({
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "pruned_layers": len(self.compressed_layers),
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "actual_sparsity": avg_actual_sparsity,
            "layer_actual_sparsities": self.layer_actual_sparsities,
            "layer_effective_ratios": self.layer_effective_ratios,
            "base_sparse_params": total_base_sparse,
            "correction_params": total_correction,
            "effective_parameter_ratio": effective_param_ratio,
            "rank": self.rank,
            "structured_24": self.structured_24,
            "decouple": self.decouple,
            "lora_alpha": self.lora_alpha,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "total_correction_energy": total_correction_energy,
            "forward_check_success": forward_check_success,
            "method": "armor_sparse_plus_lowrank_decoupled",
        })
        self.stats = stats
        self.model.compression_stats = stats
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("armor_pruner_factory")
@COMPRESSOR_REGISTRY.register("armor_factory")
@COMPRESSOR_REGISTRY.register("armor_compressor_factory")
def armor_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ARMOR pruner."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    block_size = kwargs.pop("block_size",
                 _cfg_get(comp_config, "armor_block_size",
                 _cfg_get(comp_config, "block_size", 4)))
    rank = kwargs.pop("rank",
           _cfg_get(comp_config, "armor_rank",
           _cfg_get(comp_config, "rank", 16)))
    sparsity = kwargs.pop("sparsity",
               _cfg_get(comp_config, "armor_sparsity",
               _cfg_get(comp_config, "sparsity", 0.5)))
    structured_24 = kwargs.pop("structured_24",
                     _cfg_get(comp_config, "armor_structured_24",
                     _cfg_get(comp_config, "structured_24", True)))
    decouple = kwargs.pop("decouple",
               _cfg_get(comp_config, "armor_decouple",
               _cfg_get(comp_config, "decouple", True)))
    recalibrate_ln = kwargs.pop("recalibrate_ln",
                     _cfg_get(comp_config, "armor_recalibrate_ln",
                     _cfg_get(comp_config, "recalibrate_ln", True)))
    lora_alpha = kwargs.pop("lora_alpha",
                 _cfg_get(comp_config, "armor_lora_alpha",
                 _cfg_get(comp_config, "lora_alpha", 1.0)))
    max_error = kwargs.pop("max_error",
                _cfg_get(comp_config, "armor_max_error",
                _cfg_get(comp_config, "max_error", 0.35)))
    fallback_to_original = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "armor_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "armor_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv",
        _cfg_get(comp_config, "armor_skip_attention_qkv",
                 _cfg_get(comp_config, "skip_attention_qkv", False)),
    )

    compressor = ARMORPruner(
        model,
        block_size=block_size,
        rank=rank,
        sparsity=sparsity,
        structured_24=structured_24,
        decouple=decouple,
        recalibrate_ln=recalibrate_ln,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        target_attention_only=target_attention_only,
        skip_attention_qkv=skip_attention_qkv,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "compute_resilient_taylor_saliency",
    "ARMORLinear",
    "ARMORPruner",
    "armor_pruner_factory",
]