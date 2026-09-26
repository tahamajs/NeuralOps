"""
Optimal Subspace Projection: Activation-Aware Subspace Projection via Randomized SVD.

Registered in COMPRESSOR_REGISTRY as 'optimal_subspace_projection'.

Mathematical Formulation:
  1. Activation Covariance:      C = (1/N) X^T X + eps * I
  2. Cholesky Preconditioning:   C = L L^T,   M = W L
  3. Randomized Subspace Iter.:  Y = M Omega,  Y <- M (M^T Y),  Q = qr(Y)
  4. Low-Rank Projection:        B = Q^T M = U S V^T
     W_osp = (Q U_r S_r V_r^T) L^{-1}

Bug-fix map (v2):
  OSP-FIX-01  is_custom_attention = False on compressor (it is not attention).
  OSP-FIX-02  **Preserve LoRA wrapper**: if the target module has a
              `base_layer`, we set `module.base_layer = new_layer` instead
              of replacing the entire wrapper. This keeps any external LoRA
              adapters alive and consistent with the parent module's API.
  OSP-FIX-03  **Accurate parameter accounting**: count only the BASE
              parameters (not LoRA) for the `total_orig_params` baseline,
              and count the actual `new_layer.parameters()` for the
              compressed side.
  OSP-FIX-04  **Simplified, correct rank calculation**:
              rank = clamp(int(min(d_out, d_in) * rank_ratio), min_rank, max_rank).
  OSP-FIX-05  **Forward sanity check** no longer calls `dummy.float()`
              (which would corrupt int64 input_ids). Tries `input_ids=` first,
              then positional.
  OSP-FIX-06  **Deduplicated hooks**: register at most one hook per unique
              base weight id, to avoid double-counting covariance for
              shared/tied weights.
  OSP-FIX-07  **Skip already-compressed modules** (`OptimalSubspaceLoRALinear`).
  OSP-FIX-08  **Fallback layer accounting** now adds the original layer's
              parameter count only once.
  OSP-FIX-09  Omega is always on the same device as M.
  OSP-FIX-10  Empty-rank guard (r = 0) treated as a fallback.
  OSP-FIX-11  Output projection dtype preserved (uses original raw_w.dtype).
  OSP-FIX-12  Removed dead `power_iters` on already-converged subspace.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import get_calibration_tensors
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ===========================================================================
# Config helper
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "self_attn",
}


# ===========================================================================
# Compressed linear layer
# ===========================================================================
class OptimalSubspaceLoRALinear(nn.Module):
    """
    Frozen optimal-subspace-projected base weight + optional trainable LoRA.
    """
    is_custom_attention: bool = False   # OSP-FIX-01

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
        self.nf = out_features
        self.nx = in_features
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        # Normalize to (out, in)
        if (weight.shape[0] == in_features
                and weight.shape[1] == out_features
                and in_features != out_features):
            weight = weight.t().contiguous()

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
                torch.empty((self.lora_r, in_features),
                            dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r),
                            dtype=torch.float32, device=target_device)
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
    ) -> "OptimalSubspaceLoRALinear":
        base_layer = getattr(linear, "base_layer", linear)
        bias = (
            base_layer.bias.detach()
            if hasattr(base_layer, "bias") and base_layer.bias is not None
            else None
        )
        in_features = getattr(base_layer, "in_features",
                       getattr(base_layer, "nx", None))
        out_features = getattr(base_layer, "out_features",
                        getattr(base_layer, "nf", None))
        if in_features is None or out_features is None:
            out_features = out_features or weight.shape[0]
            in_features = in_features or weight.shape[1]

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha",
                          getattr(linear, "alpha", lora_alpha))

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

        # Preserve existing LoRA weights if present
        if (hasattr(linear, "lora_A") and hasattr(linear, "lora_B")
                and layer.lora_A is not None and layer.lora_B is not None):
            with torch.no_grad():
                src_A = (linear.lora_A.weight.data
                         if hasattr(linear.lora_A, "weight")
                         else linear.lora_A.data)
                src_B = (linear.lora_B.weight.data
                         if hasattr(linear.lora_B, "weight")
                         else linear.lora_B.data)
                if src_A.shape == layer.lora_A.shape and src_B.shape == layer.lora_B.shape:
                    layer.lora_A.copy_(src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype))
                    layer.lora_B.copy_(src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype))
            layer.lora_A.requires_grad = True
            layer.lora_B.requires_grad = True

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)
        return out


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("optimal_subspace_projection")
class OptimalSubspaceProjectionCompressor(BaseCompressor):
    """
    Optimal Subspace Projection Compressor via activation-aware randomized SVD.
    """
    # OSP-FIX-01: this is a compressor, not attention.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        rank_ratio: float = 0.5,
        power_iters: int = 2,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        min_rank: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="optimal_subspace_projection", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))

        if cfg is not None:
            rank_ratio = _cfg_get(cfg, "optimal_subspace_rank_ratio",
                                  _cfg_get(cfg, "rank_ratio", rank_ratio))
            power_iters = _cfg_get(cfg, "optimal_subspace_power_iters",
                                   _cfg_get(cfg, "power_iters", power_iters))
            max_error = _cfg_get(cfg, "optimal_subspace_max_error",
                                 _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "optimal_subspace_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.calibration_loader = calibration_loader
        self.rank_ratio = max(0.01, min(float(rank_ratio), 1.0))
        self.power_iters = max(0, int(power_iters))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_texts = calibration_texts
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.min_rank = max(1, int(min_rank))

        self.target_device = (
            torch.device(device) if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if (num_heads is not None and num_kv_heads is not None
                and num_kv_heads < num_heads):
            name_lower = name.lower()
            if any(p in name_lower for p in (
                "q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"
            )):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # OSP-FIX-07: skip already-compressed modules.
        if isinstance(module, OptimalSubspaceLoRALinear):
            return False
        # Skip nested lora params.
        if any(part in name.split(".") for part in ("lora_A", "lora_B")):
            return False

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
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only:
            if not any(k in name.lower() for k in _ATTN_NAMES):
                return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_rank(
        d_out: int, d_in: int, rank_ratio: float, min_rank: int
    ) -> int:
        """OSP-FIX-04: clean, correct rank formula."""
        max_rank = min(d_out, d_in)
        target = max(1, int(round(max_rank * rank_ratio)))
        rank = max(target, min_rank)
        rank = min(rank, max_rank)
        return rank

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        # --------------------------------------------------------------
        # 1. Collect activation covariances (dedup by base weight id).
        # --------------------------------------------------------------
        act_covs: Dict[int, torch.Tensor] = {}
        weight_id_to_scan_id: Dict[int, int] = {}   # base weight id → cov key
        hooks = []

        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            base = getattr(module, "base_layer", module)
            w = getattr(base, "weight", None)
            if w is None:
                return
            w_id = id(w)
            key = weight_id_to_scan_id.get(w_id, w_id)

            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).float()
            cov = torch.matmul(x_flat.T, x_flat) / max(x_flat.shape[0], 1)
            if key not in act_covs:
                act_covs[key] = cov.cpu()
            else:
                act_covs[key] = act_covs[key] + cov.cpu()

        # OSP-FIX-06: register one hook per unique base weight.
        registered_ids: Set[int] = set()
        for name, module in self.model.named_modules():
            if not self._is_target_module(name, module):
                continue
            base = getattr(module, "base_layer", module)
            w = getattr(base, "weight", None)
            if w is None:
                continue
            w_id = id(w)
            if w_id in registered_ids:
                continue
            registered_ids.add(w_id)
            weight_id_to_scan_id[w_id] = w_id
            hooks.append(module.register_forward_hook(hook_fn))

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
                logger.debug("OSP activation collection failed: %s", e)

        for h in hooks:
            h.remove()

        # --------------------------------------------------------------
        # 2. Compress each target layer.
        # --------------------------------------------------------------
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        seen_weights: Set[int] = set()
        total_orig_params = 0
        total_comp_params = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base_module = getattr(module, "base_layer", module)
            if not hasattr(base_module, "weight") or base_module.weight is None:
                continue

            w_id = id(base_module.weight)
            if w_id in seen_weights:
                logger.info("OSP: skipping %s (shared/tied weight)", name)
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            # OSP-FIX-03: count only BASE parameters.
            base_param_count = sum(
                p.numel() for p in base_module.parameters()
            )
            total_orig_params += base_param_count

            raw_w = base_module.weight.data
            cls_name = base_module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base_module, "nf")

            W = raw_w.t().contiguous().clone().float() if is_conv1d \
                else raw_w.clone().float()
            d_out, d_in = W.shape
            device = raw_w.device
            dtype = raw_w.dtype

            # ---------------- Cholesky preconditioner ----------------
            C = act_covs.get(w_id, None)
            L = torch.eye(d_in, device=device)
            L_inv = torch.eye(d_in, device=device)

            if C is not None and C.shape[-1] == d_in:
                try:
                    C = C.to(device).float()
                    diag_mean = torch.diagonal(C).mean().item()
                    reg = max(1e-4 * max(diag_mean, 0.0), 1e-5)
                    C_reg = C + reg * torch.eye(d_in, device=device)
                    L = torch.linalg.cholesky(C_reg)
                    L_inv = torch.linalg.inv(L)
                    if not torch.isfinite(L).all() or not torch.isfinite(L_inv).all():
                        raise ValueError("Cholesky produced non-finite values")
                except Exception as e:
                    logger.debug("OSP: Cholesky fallback for %s: %s", name, e)
                    L = torch.eye(d_in, device=device)
                    L_inv = torch.eye(d_in, device=device)

            # ---------------- Randomized subspace iteration --------
            M = torch.matmul(W, L)
            rank = self._compute_rank(d_out, d_in, self.rank_ratio, self.min_rank)

            # OSP-FIX-09: Omega on the same device as M.
            Omega = torch.randn(M.shape[1], rank, device=M.device, dtype=M.dtype)
            Y = torch.matmul(M, Omega)
            for _ in range(self.power_iters):
                Y = torch.matmul(M, torch.matmul(M.T, Y))

            try:
                Q, _ = torch.linalg.qr(Y)
                B = torch.matmul(Q.T, M)
                U, S, Vt = torch.linalg.svd(B, full_matrices=False)
                r = min(rank, len(S))

                # OSP-FIX-10: empty rank → fallback.
                if r < 1:
                    raise ValueError("Empty SVD rank")

                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                new_U = torch.matmul(Q, U_r)                       # (d_out, r)
                new_V = torch.matmul(Vt_r, L_inv)                  # (r, d_in)
                new_weight = torch.matmul(
                    new_U * S_r.unsqueeze(0), new_V
                )                                                   # (d_out, d_in)
                new_weight = torch.nan_to_num(new_weight, nan=0.0,
                                               posinf=0.0, neginf=0.0)

                # ---------------- Error check ----------------
                w_norm = torch.norm(W, p="fro").clamp_min(1e-12).item()
                err = (torch.norm(W - new_weight, p="fro") / w_norm).item()
                self.layer_errors[name] = float(err)
                self.layer_ranks[name] = r

                if not math.isfinite(err) or err > self.max_error:
                    logger.warning(
                        "OSP: reconstruction error %.4f on %s exceeded %.4f.",
                        err, name, self.max_error,
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        total_comp_params += base_param_count    # OSP-FIX-08
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                # ---------------- Commit ----------------
                W_target = new_weight.to(dtype=dtype, device=device)   # OSP-FIX-11
                compressed_layer = OptimalSubspaceLoRALinear.from_linear(
                    module,
                    weight=W_target,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )

                # OSP-FIX-02: preserve LoRA wrapper.
                if hasattr(module, "base_layer"):
                    module.base_layer = compressed_layer
                else:
                    replace_module(self.model, name, compressed_layer)

                self.compressed_layers.append(name)
                total_comp_params += sum(
                    p.numel() for p in compressed_layer.parameters()
                )

            except Exception as e:
                logger.warning("OSP: SVD/decomposition failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                total_comp_params += base_param_count
                for p in module.parameters():
                    p.requires_grad = False

        # --------------------------------------------------------------
        # 3. Forward sanity check.
        # --------------------------------------------------------------
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                # OSP-FIX-05: no `.float()` on int64 input ids.
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        # Try embedding-level call as last resort.
                        cfg = getattr(self.model, "config", None)
                        hidden = getattr(cfg, "hidden_size", None) if cfg else None
                        if hidden is None:
                            first_lin = next(
                                (m for m in self.model.modules()
                                 if isinstance(m, nn.Linear)), None
                            )
                            hidden = getattr(first_lin, "in_features", 8)
                        dummy_feat = torch.randn(
                            1, 8, hidden, device=dev,
                            dtype=next(self.model.parameters()).dtype,
                        )
                        self.model(dummy_feat)
        except Exception as e:
            logger.warning("OSP forward sanity check failed: %s", e)
            forward_check_success = False

        # --------------------------------------------------------------
        # 4. Stats.
        # --------------------------------------------------------------
        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )
        eff_ratio = (
            total_comp_params / max(1, total_orig_params)
            if total_orig_params > 0 else 1.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "rank_ratio": self.rank_ratio,
            "power_iters": self.power_iters,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "effective_parameter_ratio": round(eff_ratio, 4),
            "forward_check_success": forward_check_success,
            "method": "optimal_subspace_projection",
        })
        self.stats = stats
        return self.model, stats


# Alias
SubspaceProjectionCompressor = OptimalSubspaceProjectionCompressor


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("optimal_subspace_projection_factory")
def optimal_subspace_projection_factory(
    model: nn.Module, comp_config: Any = None, **kwargs: Any
) -> Any:
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    try:
        calibration_loader = kwargs.pop(
            "calibration_loader", _cfg_get(comp_config, "calibration_loader", None)
        )
        rank_ratio = kwargs.pop(
            "rank_ratio",
            _cfg_get(comp_config, "optimal_subspace_rank_ratio",
                     _cfg_get(comp_config, "rank_ratio", 0.5)),
        )
        power_iters = kwargs.pop(
            "power_iters",
            _cfg_get(comp_config, "optimal_subspace_power_iters",
                     _cfg_get(comp_config, "power_iters", 2)),
        )
        max_error = kwargs.pop(
            "max_error",
            _cfg_get(comp_config, "optimal_subspace_max_error",
                     _cfg_get(comp_config, "max_error", 0.50)),
        )
        fallback = kwargs.pop(
            "fallback_to_original",
            _cfg_get(comp_config, "optimal_subspace_fallback_to_original",
                     _cfg_get(comp_config, "fallback_to_original", True)),
        )
        target_attention_only = kwargs.pop(
            "target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        )
        tokenizer = kwargs.pop(
            "tokenizer",
            getattr(model, "tokenizer", None)
            or _cfg_get(comp_config, "tokenizer", None),
        )
        lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
        lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))

        compressor = OptimalSubspaceProjectionCompressor(
            model,
            calibration_loader=calibration_loader,
            rank_ratio=rank_ratio,
            power_iters=power_iters,
            tokenizer=tokenizer,
            max_error=max_error,
            fallback_to_original=fallback,
            target_attention_only=target_attention_only,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            comp_config=comp_config,
            **kwargs,
        )
        compressed_model, stats = compressor.compress()
        if return_stats:
            return compressed_model, stats
        return compressed_model
    except Exception as e:
        logger.error("optimal_subspace_projection_factory failed: %s", e)
        if return_stats:
            return model, {}
        return model


# ===========================================================================
# Injection alias
# ===========================================================================
@COMPRESSOR_REGISTRY.register("optimal_subspace_projection_inject")
@COMPRESSOR_REGISTRY.register("inject_optimal_subspace_projection")
def inject_optimal_subspace_projection(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    target = layer if layer is not None else model
    if target is None:
        return None
    try:
        return optimal_subspace_projection_factory(target, **kwargs)
    except Exception as e:
        logger.error("inject_optimal_subspace_projection failed: %s", e)
        return None


__all__ = [
    "OptimalSubspaceLoRALinear",
    "OptimalSubspaceProjectionCompressor",
    "SubspaceProjectionCompressor",
    "optimal_subspace_projection_factory",
    "inject_optimal_subspace_projection",
]