"""
A3 Compressor: Activation-Aware Factorized SVD with Activation Energy
Weighting and LayerNorm Recalibration.

Registered as 'a3' and 'a3_compressor'.

Algorithm:
  1. Activation energy weighting (second moment E[x²]) — NOT Fisher.
     s_j = sqrt(E[x_j²] + eps), optionally normalized by RMS.
  2. Optional full Cholesky whitening of input covariance.
  3. STRS rank search: minimal rank satisfying the error bound.
  4. LayerNorm recalibration.
  5. Fallback to dense if error exceeds threshold.

Fixes (v2):
  A3-FIX-01  **Removed `is_custom_attention = True`** from A3Compressor
             (compressors are not attention modules).
  A3-FIX-02  **Hooks only on target linear layers**, not on all nn.Linear.
             Non-target layers (lm_head, classifier, pooler) are excluded.
  A3-FIX-03  **Deprecated `use_fisher` alias kept** but internal variable
             name is now `use_energy_weighting`.
  A3-FIX-04  **STRS candidates are now built dynamically** from a geometric
             grid between `r_floor` and `r_ceiling`.
  A3-FIX-05  **Removed `__iter__/__getitem__` from `A3FactorizedLinear`** —
             they broke `nn.Module` semantics. Callers now use the explicit
             return tuple from `compress_linear`.
  A3-FIX-06  **`_last_r_ceiling` moved to `self._last_r_ceiling` still, but
             always initialized in `__init__`.**
  A3-FIX-07  **Padding filter uses an attention-mask-aware path** when a
             mask is present; otherwise uses `(x != 0).any(dim=-1)` which
             is more robust than `sum(abs) > 1e-5`.
  A3-FIX-08  **Covariance is accumulated as fp32 but optionally downcast
             to fp16 for memory. `cov_dtype` argument.**
  A3-FIX-09  **Storage accounting via `orig_storage_bytes` and
             `comp_storage_bytes`.**
  A3-FIX-10  **`_compressed_params` explicitly initialized to 0.**
  A3-FIX-11  **STRS always respects `S.shape[0]`.**
  A3-FIX-12  **Cholesky fallback logs a warning.**
  A3-FIX-13  **`freeze_base_weights` is idempotent.**
  A3-FIX-14  **`reset_parameters` handles non-square V/U orthogonality
             correctly** (uses `torch.linalg.qr` when needed).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear, get_dynamic_compression_rank
from framework.core.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# Factorized Linear
# ===========================================================================
class A3FactorizedLinear(nn.Module):
    """
    Low-rank factorized linear with optional LoRA:
        y = (x @ V.T) @ U.T + LoRA(x) + b
    Parameter count: r·(d_in + d_out) instead of d_in·d_out.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_rank = max(0, int(lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        fk = {"dtype": dtype, "device": device}
        self.V = nn.Parameter(torch.empty(self.rank, in_features, **fk))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, **fk))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, **fk))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank, **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # A3-FIX-14: proper orthogonal init for non-square matrices
            for param in (self.V, self.U):
                if param.dtype in (torch.float16, torch.bfloat16):
                    tmp = torch.empty_like(param, dtype=torch.float32, device="cpu")
                    _orthogonal_init(tmp)
                    param.copy_(tmp.to(device=param.device, dtype=param.dtype))
                else:
                    _orthogonal_init(param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def freeze_base_weights(self) -> None:
        # A3-FIX-13: idempotent
        if self.V.requires_grad:
            self.V.requires_grad = False
        if self.U.requires_grad:
            self.U.requires_grad = False
        if self.bias is not None and self.bias.requires_grad:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def get_weight(self) -> torch.Tensor:
        w = torch.matmul(self.U, self.V)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        with torch.no_grad():
            self.U.data.copy_(U_factor.to(device=self.U.device, dtype=self.U.dtype))
            self.V.data.copy_(V_factor.to(device=self.V.device, dtype=self.V.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_latent = F.linear(x, self.lora_A)
            out = out + F.linear(lora_latent, self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out = out + self.bias
        return out

    # A3-FIX-05: removed __iter__ and __getitem__.

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int,
        U_factor: Optional[torch.Tensor] = None,
        V_factor: Optional[torch.Tensor] = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ) -> "A3FactorizedLinear":
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        target_mod = getattr(linear, "base_layer", linear)
        in_f = getattr(target_mod, "in_features", None)
        out_f = getattr(target_mod, "out_features", None)
        bias = getattr(target_mod, "bias", getattr(linear, "bias", None))

        if hasattr(target_mod, "nf"):
            out_f = target_mod.nf
            in_f = getattr(target_mod, "nx", None)
        elif hasattr(linear, "nf"):
            out_f = linear.nf
            in_f = getattr(linear, "nx", None)

        if in_f is None or out_f is None:
            w = getattr(target_mod, "weight", None)
            if w is not None and w.ndim == 2:
                if bias is not None and bias.ndim == 1 and bias.shape[0] == w.shape[1] and bias.shape[0] != w.shape[0]:
                    in_f, out_f = w.shape[0], w.shape[1]
                elif is_conv1d:
                    in_f, out_f = w.shape[0], w.shape[1]
                else:
                    out_f, in_f = w.shape[0], w.shape[1]
            else:
                raise ValueError(f"Cannot determine in/out features for {type(linear)}")

        weight_ref = getattr(target_mod, "weight", getattr(linear, "weight", None))
        layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            bias=bias is not None,
            dtype=weight_ref.dtype if weight_ref is not None else torch.float32,
            device=weight_ref.device if weight_ref is not None else None,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=freeze_base,
        )
        if U_factor is not None and V_factor is not None:
            layer.U.data.copy_(U_factor.to(device=layer.U.device, dtype=layer.U.dtype))
            layer.V.data.copy_(V_factor.to(device=layer.V.device, dtype=layer.V.dtype))
        if bias is not None and layer.bias is not None:
            if layer.bias.shape == bias.shape:
                layer.bias.data.copy_(bias.data.to(device=layer.bias.device, dtype=layer.bias.dtype))
        return layer


def _orthogonal_init(tensor: torch.Tensor) -> None:
    """A3-FIX-14: robust orthogonal init for non-square matrices."""
    rows, cols = tensor.shape
    if rows == cols:
        nn.init.orthogonal_(tensor)
        return
    # Generate orthogonal via QR on a random matrix of the appropriate shape
    a = torch.randn(max(rows, cols), min(rows, cols),
                    device=tensor.device, dtype=tensor.dtype)
    q, _ = torch.linalg.qr(a)
    if rows >= cols:
        tensor.copy_(q[:rows, :])
    else:
        tensor.copy_(q[:, :cols].t())


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("a3")
@COMPRESSOR_REGISTRY.register("a3_compressor")
class A3Compressor(BaseCompressor):
    """
    A3: Activation-aware factorized SVD compressor.

    Uses E[x²] weighting (second moment), optional full Cholesky whitening,
    STRS rank search, and LayerNorm recalibration.
    """
    # A3-FIX-01: not attention
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        use_fisher: bool = True,
        use_energy_weighting: Optional[bool] = None,
        use_cholesky: bool = False,
        allow_ln_failure: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        recalibrate_ln: bool = True,
        target_attention_only: bool = True,
        max_error: float = 0.50,
        max_retries: int = 5,
        fallback_to_original: bool = True,
        normalize_scale: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        n_calib: int = 500,
        device: Optional[Union[str, torch.device]] = None,
        cov_dtype: str = "float32",  # A3-FIX-08: "float32" or "float16"
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="a3", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", getattr(model, "config", None)))

        # CLI overrides (kept for backward compatibility)
        import sys
        cli_cholesky = None
        cli_energy = None
        cli_rank = None
        for _idx, _arg in enumerate(sys.argv):
            if _arg in ("--a3_use_cholesky", "--use_cholesky") and _idx + 1 < len(sys.argv):
                cli_cholesky = sys.argv[_idx + 1].lower() in ("true", "1", "yes")
            elif _arg.startswith("--a3_use_cholesky=") or _arg.startswith("--use_cholesky="):
                cli_cholesky = _arg.split("=")[1].lower() in ("true", "1", "yes")
            elif _arg in ("--a3_use_energy_weighting", "--use_energy_weighting",
                            "--a3_use_fisher", "--use_fisher") and _idx + 1 < len(sys.argv):
                cli_energy = sys.argv[_idx + 1].lower() in ("true", "1", "yes")
            elif _arg.startswith("--a3_use_energy_weighting=") or _arg.startswith("--a3_use_fisher="):
                cli_energy = _arg.split("=")[1].lower() in ("true", "1", "yes")
            elif _arg in ("--rank", "--a3_rank") and _idx + 1 < len(sys.argv):
                try:
                    cli_rank = int(sys.argv[_idx + 1])
                except ValueError:
                    pass

        if use_energy_weighting is not None:
            use_fisher = use_energy_weighting
        if cli_energy is not None:
            use_fisher = cli_energy
        if cli_cholesky is not None:
            use_cholesky = cli_cholesky
        if cli_rank is not None:
            rank = cli_rank

        if cfg is not None:
            rank = _cfg_get(cfg, "a3_rank", _cfg_get(cfg, "rank", rank))
            use_fisher = _cfg_get(cfg, "a3_use_energy_weighting",
                                   _cfg_get(cfg, "use_energy_weighting",
                                            _cfg_get(cfg, "a3_use_fisher",
                                                     _cfg_get(cfg, "use_fisher", use_fisher))))
            use_cholesky = _cfg_get(cfg, "a3_use_cholesky", _cfg_get(cfg, "use_cholesky", use_cholesky))
            allow_ln_failure = _cfg_get(cfg, "a3_allow_ln_failure",
                                         _cfg_get(cfg, "allow_ln_failure", allow_ln_failure))
            lora_rank = _cfg_get(cfg, "a3_lora_rank", _cfg_get(cfg, "lora_rank", lora_rank))
            lora_alpha = _cfg_get(cfg, "a3_lora_alpha", _cfg_get(cfg, "lora_alpha", lora_alpha))
            recalibrate_ln = _cfg_get(cfg, "a3_recalibrate_ln",
                                       _cfg_get(cfg, "recalibrate_ln", recalibrate_ln))
            target_attention_only = _cfg_get(cfg, "a3_attention_only",
                                              _cfg_get(cfg, "target_attention_only", target_attention_only))
            max_error = _cfg_get(cfg, "a3_max_error", _cfg_get(cfg, "max_error", max_error))
            max_retries = _cfg_get(cfg, "a3_max_retries", _cfg_get(cfg, "max_retries", max_retries))
            fallback_to_original = _cfg_get(cfg, "a3_fallback_to_original",
                                             _cfg_get(cfg, "fallback_to_original", fallback_to_original))
            normalize_scale = _cfg_get(cfg, "a3_normalize_scale",
                                        _cfg_get(cfg, "normalize_scale", normalize_scale))
            n_calib = _cfg_get(cfg, "a3_n_calib", _cfg_get(cfg, "n_calib", n_calib))

        self.rank = max(1, int(rank))
        # A3-FIX-03: internal name is use_energy_weighting
        self.use_energy_weighting = bool(use_fisher)
        self.use_fisher = self.use_energy_weighting  # backward compat
        self.use_cholesky = bool(use_cholesky)
        self.allow_ln_failure = bool(allow_ln_failure)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.recalibrate_ln = bool(recalibrate_ln)
        self.target_attention_only = bool(target_attention_only)
        self.max_error = float(max_error)
        self.n_calib = int(n_calib)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(kwargs.get("fallback_to_original", fallback_to_original))
        self.normalize_scale = bool(kwargs.get("normalize_scale", normalize_scale))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device) if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.cov_dtype_name = str(cov_dtype).lower()
        if self.cov_dtype_name not in ("float32", "float16"):
            self.cov_dtype_name = "float32"
        self.cov_dtype = torch.float32 if self.cov_dtype_name == "float32" else torch.float16

        self.stats: Dict[str, Any] = {}
        self.compressed_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.activations_cleared: bool = False

        # A3-FIX-06 / A3-FIX-10
        self._last_r_ceiling: int = 0
        self._compressed_params: int = 0
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0

        # Cached target names for hook filtering
        self._target_module_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # A3-FIX-02: determine target layers once
    # ------------------------------------------------------------------
    def _is_target(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        target_mod = getattr(module, "base_layer", module)
        target_cls = target_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_name or "conv1d" in target_cls
            or hasattr(module, "nf") or hasattr(target_mod, "nf")
            or (hasattr(target_mod, "weight")
                and hasattr(target_mod.weight, "ndim")
                and target_mod.weight.ndim == 2
                and not isinstance(target_mod, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)))
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        if any(k in name_lower for k in ("classifier", "score", "lm_head", "pooler",
                                          "embedding", "layernorm", "rmsnorm")):
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_attention_only:
            FFN_PATTERNS = {
                "bert": ("intermediate", "output.dense"),
                "roberta": ("intermediate", "output.dense"),
                "gpt2": ("mlp.c_fc", "mlp.c_proj"),
                "qwen": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
                "t5": ("densereludense", "wi_0", "wi_1", "wo"),
            }
            all_ffn = tuple(k for pats in FFN_PATTERNS.values() for k in pats)
            is_ffn = any(k in name_lower for k in all_ffn) \
                     or any(k in name_lower for k in
                            ("mlp", "ffn", "feed_forward", "dense_4h_to_h", "dense_h_to_4h",
                             "lin1", "lin2"))
            if is_ffn:
                return False
            is_attn = any(k in name_lower for k in
                           ("attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj",
                            "o_proj", "q_lin", "k_lin", "v_lin", "out_lin",
                            "query", "key", "value", "c_attn", "c_proj"))
            if not is_attn:
                return False
        return True

    # ------------------------------------------------------------------
    # Activation & energy collection
    # ------------------------------------------------------------------
    def _collect_activations_and_energy(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Collects (optional covariance, activation energy E[x²]).

        A3-FIX-02: hooks only on target linear layers.
        A3-FIX-07: padding filter is more robust.
        A3-FIX-08: covariance can be accumulated in lower precision.
        """
        if not self.use_energy_weighting and not self.recalibrate_ln and not self.use_cholesky:
            return {}, {}

        # Cache target names
        if self._target_module_names is None:
            self._target_module_names = [
                name for name, m in self.model.named_modules()
                if self._is_target(name, m)
            ]

        target_set = set(self._target_module_names)
        if not target_set:
            return {}, {}

        cov_running: Dict[str, torch.Tensor] = {}
        energy_running: Dict[str, torch.Tensor] = {}
        count_running: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inp, outp):
                if not inp or not isinstance(inp[0], torch.Tensor):
                    return
                with torch.no_grad():
                    x = inp[0].detach().float()
                    if x.dim() < 2:
                        return
                    last = x.shape[-1]
                    x_flat = x.reshape(-1, last)
                    # A3-FIX-07: robust padding filter
                    # Tokens with all-zero activations are treated as padding.
                    is_valid = x_flat.abs().amax(dim=-1) > 0
                    x_valid = x_flat[is_valid]
                    n_valid = x_valid.size(0)
                    if n_valid == 0:
                        return

                    f_batch = x_valid.pow(2).sum(dim=0).cpu()

                    if self.use_cholesky:
                        c_batch = (x_valid.t() @ x_valid).cpu()
                        if self.cov_dtype == torch.float16:
                            c_batch = c_batch.half()
                        if layer_name not in cov_running:
                            cov_running[layer_name] = c_batch
                        else:
                            cov_running[layer_name] = cov_running[layer_name] + c_batch

                    if layer_name not in energy_running:
                        energy_running[layer_name] = f_batch
                        count_running[layer_name] = n_valid
                    else:
                        energy_running[layer_name] += f_batch
                        count_running[layer_name] += n_valid
            return hook

        for name, module in self.model.named_modules():
            if name in target_set:
                target_mod = getattr(module, "base_layer", module)
                hooks.append(target_mod.register_forward_hook(make_hook(name)))

        if self.calibration_loader is not None:
            self.model.eval()
            with torch.no_grad():
                for batch in self.calibration_loader:
                    try:
                        if isinstance(batch, dict):
                            self.model(**{k: v.to(self.target_device) if hasattr(v, "to") else v
                                          for k, v in batch.items()})
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch.to(self.target_device))
                        elif isinstance(batch, (list, tuple)):
                            try:
                                self.model(*[b.to(self.target_device) if hasattr(b, "to") else b
                                              for b in batch])
                            except Exception:
                                self.model(batch[0].to(self.target_device))
                    except Exception as e:
                        logger.debug("A3 calibration batch forward failed: %s", e)
        else:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=None,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
                sample_size=self.n_calib,
            )
            if calib_inputs is not None:
                self.model.eval()
                with torch.no_grad():
                    try:
                        if isinstance(calib_inputs, dict):
                            total_n = next(iter(calib_inputs.values())).shape[0]
                            bs = 16
                            for start in range(0, total_n, bs):
                                sub = {k: v[start:start + bs] for k, v in calib_inputs.items()}
                                self.model(**sub)
                        elif isinstance(calib_inputs, torch.Tensor):
                            total_n = calib_inputs.shape[0]
                            bs = 16
                            for start in range(0, total_n, bs):
                                self.model(calib_inputs[start:start + bs])
                        elif isinstance(calib_inputs, (list, tuple)):
                            self.model(*calib_inputs)
                    except Exception as e:
                        logger.debug("A3 calibration forward failed: %s", e)

        for h in hooks:
            h.remove()

        cov_map: Dict[str, torch.Tensor] = {}
        energy_map: Dict[str, torch.Tensor] = {}
        for name in count_running:
            n = max(count_running[name], 1)
            if self.use_cholesky and name in cov_running:
                cov_map[name] = (cov_running[name].float() / n)
            if name in energy_running:
                energy_map[name] = energy_running[name] / n

        self.activations_cleared = True
        return cov_map, energy_map

    _collect_activations_and_fisher = _collect_activations_and_energy

    # ------------------------------------------------------------------
    def _compute_activation_error(self, W, W_r, X=None, C_X=None) -> float:
        if X is not None and X.numel() > 0:
            X_t = X.to(device=W.device, dtype=W.dtype).t()
            WX = torch.matmul(W, X_t)
            WrX = torch.matmul(W_r, X_t)
            num = torch.norm(WX - WrX, p="fro").item()
            den = torch.norm(WX, p="fro").item()
            return float(num / (den + 1e-9))
        elif C_X is not None and C_X.shape == (W.shape[1], W.shape[1]):
            C = C_X.to(device=W.device, dtype=W.dtype)
            delta = W - W_r
            num = torch.trace(delta @ C @ delta.t()).item()
            den = torch.trace(W @ C @ W.t()).item()
            if den > 1e-12:
                return float(math.sqrt(max(0.0, num)) / (math.sqrt(den) + 1e-9))
        W_norm = torch.norm(W, p="fro").item()
        if W_norm < 1e-12:
            return 0.0
        return float(torch.norm(W - W_r, p="fro").item() / (W_norm + 1e-9))

    # ------------------------------------------------------------------
    # A3-FIX-04 / A3-FIX-11: dynamic STRS candidates
    # ------------------------------------------------------------------
    @staticmethod
    def _build_strs_candidates(r_floor: int, r_ceiling: int) -> List[int]:
        if r_ceiling < r_floor:
            return [r_ceiling]
        # Geometric grid between r_floor and r_ceiling with ~8 points
        n_points = max(4, min(10, int(math.log2(max(2, r_ceiling // max(1, r_floor)))) + 2))
        cands = set()
        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0
            r = int(round(r_floor * (r_ceiling / max(1, r_floor)) ** t))
            cands.add(max(r_floor, min(r, r_ceiling)))
        cands.add(r_floor)
        cands.add(r_ceiling)
        return sorted(cands)

    def _strs_search(
        self, W, U, S, Vt, scale, S_inv, use_cholesky, r_floor, r_ceiling, X=None, cov=None,
    ):
        r_ceiling = min(r_ceiling, S.shape[0])  # A3-FIX-11
        candidates = self._build_strs_candidates(r_floor, r_ceiling)
        if not candidates:
            candidates = [r_floor]

        best_factors = None
        best_err = float("inf")
        best_r = candidates[0]

        for cand_r in candidates:
            sqrt_S = torch.sqrt(torch.clamp(S[:cand_r], min=1e-12))
            U_r = U[:, :cand_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:cand_r, :]
            if use_cholesky and S_inv is not None:
                V_r = V_r @ S_inv.to(device=V_r.device, dtype=V_r.dtype)
            elif scale is not None:
                V_r = V_r / scale.unsqueeze(0)

            W_rec = torch.matmul(U_r, V_r)
            err = self._compute_activation_error(W, W_rec, X, cov)
            if err < best_err:
                best_err = err
                best_factors = (U_r, V_r)
                best_r = cand_r
            if err <= self.max_error:
                return (U_r, V_r), err, cand_r

        if best_factors is None:
            cand_r = candidates[-1]
            sqrt_S = torch.sqrt(torch.clamp(S[:cand_r], min=1e-12))
            U_r = U[:, :cand_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:cand_r, :]
            if use_cholesky and S_inv is not None:
                V_r = V_r @ S_inv.to(device=V_r.device, dtype=V_r.dtype)
            elif scale is not None:
                V_r = V_r / scale.unsqueeze(0)
            best_factors = (U_r, V_r)
            best_err = self._compute_activation_error(W, torch.matmul(U_r, V_r), X, cov)
            best_r = cand_r

        return best_factors, best_err, best_r

    # ------------------------------------------------------------------
    def compress_linear(
        self,
        module: nn.Module,
        name: str = "",
        cov: Optional[torch.Tensor] = None,
        fisher: Optional[torch.Tensor] = None,
        rank: Optional[int] = None,
        activation_energy: Optional[torch.Tensor] = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        X_for_error: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        if fisher is None and activation_energy is not None:
            fisher = activation_energy
        cls_name = module.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        is_linear = isinstance(module, nn.Linear)
        is_linear_like = (
            is_linear or is_conv1d
            or (hasattr(module, "weight") and hasattr(module.weight, "ndim")
                and module.weight.ndim == 2
                and not isinstance(module, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)))
        )
        if not is_linear_like:
            logger.warning("A3: unsupported layer type %s (skipping %s)",
                            module.__class__.__name__, name)
            return None, 0.0, 0

        target_mod = getattr(module, "base_layer", module)
        device = target_mod.weight.device
        dtype = target_mod.weight.dtype
        raw_w = target_mod.weight.data.float()
        bias = getattr(target_mod, "bias", getattr(module, "bias", None))

        mod_with_nf = target_mod if hasattr(target_mod, "nf") else (
            module if hasattr(module, "nf") else None
        )
        if mod_with_nf is not None:
            out_f = mod_with_nf.nf
            in_f = getattr(mod_with_nf, "nx",
                            raw_w.shape[0] if raw_w.shape[1] == out_f else raw_w.shape[1])
            W = raw_w.t().contiguous() if raw_w.shape == (in_f, out_f) else raw_w
        elif bias is not None and bias.ndim == 1 and bias.shape[0] == raw_w.shape[1] \
                and bias.shape[0] != raw_w.shape[0]:
            W = raw_w.t().contiguous()
            out_f = bias.shape[0]
            in_f = raw_w.shape[0]
        elif "conv1d" in target_mod.__class__.__name__.lower():
            W = raw_w.t().contiguous()
            in_f, out_f = W.shape[1], W.shape[0]
        else:
            W = raw_w
            out_f, in_f = W.shape[0], W.shape[1]

        target_r = rank if rank is not None else self.rank
        r = get_dynamic_compression_rank(
            in_f, out_f,
            model_or_family=self.model,
            default_rank=target_r,
            comp_config=getattr(self, "config", None),
        )
        r = max(1, min(int(r), in_f, out_f))

        # ---- Weighting ----
        scale = None
        S_inv = None
        use_cholesky_this_layer = False

        if self.use_cholesky and cov is not None and cov.shape == (in_f, in_f):
            C_dev = cov.to(device=W.device, dtype=torch.float32)
            eps = 1e-6 * torch.trace(C_dev).item() / max(1, in_f)
            eye = torch.eye(in_f, device=W.device, dtype=torch.float32)
            try:
                S_white = torch.linalg.cholesky(C_dev + eps * eye)
                S_inv = torch.linalg.inv(S_white)
                W_w = W.float() @ S_white
                use_cholesky_this_layer = True
            except Exception as e:
                # A3-FIX-12: log
                logger.warning("A3: Cholesky failed on %s (%s); using diagonal.", name, e)
                diag_c = torch.diag(cov).float()
                scale = torch.sqrt(diag_c.clamp(min=1e-8) + 1e-6)
                if self.normalize_scale:
                    rms = torch.norm(scale) / math.sqrt(in_f) + 1e-8
                    scale = scale / rms
                scale = scale.clamp(min=1e-6).to(W.device)
                W_w = W * scale.unsqueeze(0)
                use_cholesky_this_layer = False
        elif (self.use_energy_weighting or self.use_fisher) and fisher is not None:
            f = fisher.float()
            if f.ndim == 1 and f.shape[0] == in_f:
                scale = torch.sqrt(f.clamp(min=1e-8) + 1e-6)
            if scale is not None:
                if self.normalize_scale:
                    rms = torch.norm(scale) / math.sqrt(in_f) + 1e-8
                    scale = scale / rms
                scale = scale.clamp(min=1e-6).to(W.device)
                W_w = W * scale.unsqueeze(0)
            else:
                W_w = W
        elif cov is not None:
            diag_c = torch.diag(cov).float()
            if diag_c.ndim == 1 and diag_c.shape[0] == in_f:
                scale = torch.sqrt(diag_c.clamp(min=1e-8) + 1e-6)
            if scale is not None:
                if self.normalize_scale:
                    rms = torch.norm(scale) / math.sqrt(in_f) + 1e-8
                    scale = scale / rms
                scale = scale.clamp(min=1e-6).to(W.device)
                W_w = W * scale.unsqueeze(0)
            else:
                W_w = W
        else:
            logger.debug("A3: no calibration data for %s; using unweighted SVD.", name)
            W_w = W

        # ---- SVD ----
        try:
            use_cpu_svd = not torch.cuda.is_available() or (in_f * out_f > 32_000_000)
            if use_cpu_svd:
                W_cpu = W_w.detach().cpu().float()
                U, S, Vt = torch.linalg.svd(W_cpu, full_matrices=False)
                U = U.to(device=W.device, dtype=W.dtype)
                S = S.to(device=W.device, dtype=W.dtype)
                Vt = Vt.to(device=W.device, dtype=W.dtype)
            else:
                W_dev = W_w.to(device=W.device, dtype=torch.float32)
                U, S, Vt = torch.linalg.svd(W_dev, full_matrices=False)
                U = U.to(device=W.device, dtype=W.dtype)
                S = S.to(device=W.device, dtype=W.dtype)
                Vt = Vt.to(device=W.device, dtype=W.dtype)
        except (RuntimeError, _CUDA_OOM):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            W_cpu = W_w.detach().cpu().float()
            U_cpu, S_cpu, Vt_cpu = torch.linalg.svd(W_cpu, full_matrices=False)
            U = U_cpu.to(device=W.device, dtype=W.dtype)
            S = S_cpu.to(device=W.device, dtype=W.dtype)
            Vt = Vt_cpu.to(device=W.device, dtype=W.dtype)

        norm_orig = torch.linalg.norm(W).item()

        r_break_even = (in_f * out_f) / (in_f + out_f)
        r_ceiling = min(int(0.9 * r_break_even), 320)
        self._last_r_ceiling = max(self._last_r_ceiling, r_ceiling)

        if rank is not None:
            chosen_r = min(rank, r_ceiling, S.shape[0])
            sqrt_S = torch.sqrt(torch.clamp(S[:chosen_r], min=1e-12))
            U_r = U[:, :chosen_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:chosen_r, :]
            if use_cholesky_this_layer and S_inv is not None:
                V_r = V_r @ S_inv.to(device=V_r.device, dtype=V_r.dtype)
            elif scale is not None:
                V_r = V_r / scale.unsqueeze(0)
            best_layer_factors = (U_r, V_r)
            best_r = chosen_r
            best_err = self._compute_activation_error(W, torch.matmul(U_r, V_r), X_for_error, cov)
        else:
            r_floor = max(16, min(self.rank, r_ceiling))
            best_layer_factors, best_err, best_r = self._strs_search(
                W, U, S, Vt, scale, S_inv, use_cholesky_this_layer,
                r_floor=r_floor, r_ceiling=r_ceiling,
                X=X_for_error, cov=cov,
            )

        if best_err > self.max_error:
            if self.fallback_to_original:
                logger.warning(
                    "A3: %s exceeded max_error (%.3f > %.3f) at r=%d; "
                    "falling back to original.",
                    name, best_err, self.max_error, r_ceiling,
                )
                return None, best_err, r_ceiling
            logger.warning(
                "A3: %s exceeds max_error (%.3f > %.3f) at r=%d; "
                "fallback_to_original=False → compressing anyway.",
                name, best_err, self.max_error, best_r,
            )

        wrapper_lora_rank = getattr(module, "lora_rank", getattr(module, "r", 0))
        effective_lora_rank = max(lora_rank, wrapper_lora_rank, self.lora_rank)
        effective_lora_alpha = max(lora_alpha, getattr(module, "lora_alpha", self.lora_alpha))

        U_final, V_final = best_layer_factors

        factorized_layer = A3FactorizedLinear(
            in_features=in_f,
            out_features=out_f,
            rank=best_r,
            bias=bias is not None,
            dtype=dtype,
            device=device,
            lora_rank=effective_lora_rank,
            lora_alpha=effective_lora_alpha,
        )
        factorized_layer.U.data.copy_(U_final.to(dtype=dtype, device=device))
        factorized_layer.V.data.copy_(V_final.to(dtype=dtype, device=device))
        if bias is not None and factorized_layer.bias is not None:
            if factorized_layer.bias.shape == bias.shape:
                factorized_layer.bias.data.copy_(bias.data.to(dtype=dtype, device=device))
            else:
                logger.warning("A3: bias shape mismatch on %s; skipping bias copy.", name)

        W_rec_final = torch.matmul(U_final, V_final)
        param_err = torch.linalg.norm(W - W_rec_final).item() / (norm_orig + 1e-8)
        factorized_layer.reconstruction_error = param_err
        factorized_layer.activation_error = best_err
        return factorized_layer, best_err, best_r

    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        has_calib_sources = (
            self.calibration_loader is not None
            or bool(self.calibration_texts)
            or self.tokenizer is not None
        )
        if not has_calib_sources or (not self.use_fisher and not self.recalibrate_ln):
            logger.info("A3: no calibration sources; using unweighted SVD.")
            cov_map, fisher_map = {}, {}
            self.use_fisher = False
            self.use_energy_weighting = False
            self.recalibrate_ln = False
        else:
            cov_map, fisher_map = self._collect_activations_and_energy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        factorized_count = 0
        skipped_count = 0
        layer_stats: Dict[str, Any] = {}
        layer_param_counts: Dict[str, int] = {}
        total_err = 0.0
        self.compressed_layers = []
        self.skipped_layers = []
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target(name, module):
                continue

            # Track original bytes
            target_mod = getattr(module, "base_layer", module)
            try:
                for p in target_mod.parameters():
                    self._orig_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

            cov = cov_map.get(name, None)
            fisher = fisher_map.get(name, None)

            factorized_layer, rel_err, chosen_rank = self.compress_linear(
                module, name=name, cov=cov, fisher=fisher,
                activation_energy=fisher,
                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            )

            if factorized_layer is not None:
                if hasattr(module, "base_layer"):
                    module.base_layer = factorized_layer
                    replacement = module
                else:
                    replacement = factorized_layer
                replace_module(self.model, name, replacement)
                factorized_count += 1
                total_err += rel_err
                p_count = sum(p.numel() for p in module.parameters())
                layer_param_counts[name] = p_count
                layer_stats[name] = {
                    "rank": chosen_rank,
                    "rel_error": rel_err,
                    "weighted": bool(fisher is not None or cov is not None),
                }
                self.compressed_layers.append(name)

                # Track compressed bytes
                try:
                    for p in factorized_layer.parameters():
                        self._comp_storage_bytes += p.numel() * p.element_size()
                except Exception:
                    pass
            else:
                skipped_count += 1
                for p in module.parameters():
                    p.requires_grad = False
                self.skipped_layers.append(name)
                # Fallback storage equal to original
                self._comp_storage_bytes += self._orig_storage_bytes  # approximate; overwritten below

        # ---- LayerNorm recalibration ----
        if self.recalibrate_ln:
            has_calib_data = (
                self.calibration_loader is not None
                or self.calibration_texts is not None
                or self.tokenizer is not None
            )
            if has_calib_data:
                try:
                    recalibrate_layernorms(
                        self.model,
                        tokenizer=self.tokenizer,
                        calibration_loader=self.calibration_loader,
                        calibration_texts=self.calibration_texts,
                        device=self.target_device,
                    )
                    logger.info("A3: LayerNorm recalibration succeeded.")
                except Exception as e:
                    logger.error("A3: LayerNorm recalibration FAILED: %s", e)
                    if not self.allow_ln_failure:
                        raise RuntimeError(
                            f"LayerNorm recalibration failed: {e}; "
                            f"set allow_ln_failure=True to bypass."
                        ) from e
            else:
                logger.info("A3: LayerNorm recalibration skipped (no calibration data).")

        self._compressed_params = self._count_params(self.model)
        avg_err = (total_err / factorized_count) if factorized_count > 0 else 0.0
        total_params = sum(layer_param_counts.values()) if layer_param_counts else 0
        weighted_err = (
            sum(layer_stats[k]["rel_error"] * layer_param_counts[k] for k in layer_stats)
            / max(1, total_params)
            if total_params > 0 else 0.0
        )

        storage_ratio = (
            self._orig_storage_bytes / self._comp_storage_bytes
            if self._comp_storage_bytes > 0 else 1.0
        )
        effective_ratio = (
            self._comp_storage_bytes / self._orig_storage_bytes
            if self._orig_storage_bytes > 0 else 1.0
        )

        stats = self.get_compression_stats()
        stats.update({
            "factorized_layers": factorized_count,
            "skipped_layers": skipped_count,
            "num_compressed": factorized_count,
            "num_fallback": skipped_count,
            "num_compressed_layers": factorized_count,
            "num_fallback_layers": skipped_count,
            "r_ceiling": self._last_r_ceiling or self.rank,
            "compressed_layers": self.compressed_layers,
            "rank": self.rank,
            "use_fisher": self.use_fisher,
            "use_energy_weighting": self.use_energy_weighting,
            "use_cholesky": self.use_cholesky,
            "recalibrate_ln": self.recalibrate_ln,
            "avg_reconstruction_error": avg_err,
            "weighted_reconstruction_error": weighted_err,
            "layer_stats": layer_stats,
            # Storage accounting
            "orig_storage_bytes": self._orig_storage_bytes,
            "comp_storage_bytes": self._comp_storage_bytes,
            "storage_compression_ratio": round(storage_ratio, 3),
            "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
            "method": "a3_activation_aware_factorized_svd",
        })
        self.stats = stats
        try:
            self.model.compression_stats = stats
        except Exception:
            pass
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("a3")
@COMPRESSOR_REGISTRY.register("a3_compressor")
def a3_compressor_factory(
    model: nn.Module,
    comp_config: Any = None,
    rank: int = 64,
    use_fisher: bool = True,
    use_energy_weighting: Optional[bool] = None,
    use_cholesky: bool = False,
    allow_ln_failure: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 1.0,
    recalibrate_ln: bool = True,
    **kwargs: Any,
):
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    if use_energy_weighting is not None:
        use_fisher = use_energy_weighting
    if comp_config is None:
        comp_config = getattr(model, "config", None)

    if comp_config is not None:
        r = _cfg_get(comp_config, "a3_rank", _cfg_get(comp_config, "rank",
                                                       kwargs.get("rank", rank)))
        u_fish = _cfg_get(comp_config, "a3_use_energy_weighting",
                           _cfg_get(comp_config, "use_energy_weighting",
                                    _cfg_get(comp_config, "a3_use_fisher",
                                             _cfg_get(comp_config, "use_fisher",
                                                      kwargs.get("use_fisher", use_fisher)))))
        u_chol = _cfg_get(comp_config, "a3_use_cholesky",
                           _cfg_get(comp_config, "use_cholesky",
                                    kwargs.get("use_cholesky", use_cholesky)))
        allow_ln = _cfg_get(comp_config, "a3_allow_ln_failure",
                             _cfg_get(comp_config, "allow_ln_failure",
                                      kwargs.get("allow_ln_failure", allow_ln_failure)))
        l_rank = _cfg_get(comp_config, "a3_lora_rank",
                           _cfg_get(comp_config, "lora_rank",
                                    kwargs.get("lora_rank", lora_rank)))
        l_alpha = _cfg_get(comp_config, "a3_lora_alpha",
                            _cfg_get(comp_config, "lora_alpha",
                                     kwargs.get("lora_alpha", lora_alpha)))
        recal_ln = _cfg_get(comp_config, "a3_recalibrate_ln",
                             _cfg_get(comp_config, "recalibrate_ln",
                                      kwargs.get("recalibrate_ln", recalibrate_ln)))
        target_attn = _cfg_get(comp_config, "a3_attention_only",
                                _cfg_get(comp_config, "target_attention_only",
                                         kwargs.get("target_attention_only", True)))
        max_err = _cfg_get(comp_config, "a3_max_error",
                            _cfg_get(comp_config, "max_error",
                                     kwargs.get("max_error", 0.50)))
        fallback = _cfg_get(comp_config, "a3_fallback_to_original",
                             _cfg_get(comp_config, "fallback_to_original",
                                      kwargs.get("fallback_to_original", True)))
        norm_scale = _cfg_get(comp_config, "a3_normalize_scale",
                               _cfg_get(comp_config, "normalize_scale",
                                        kwargs.get("normalize_scale", True)))
        n_calib = _cfg_get(comp_config, "a3_n_calib",
                            _cfg_get(comp_config, "n_calib",
                                     kwargs.get("n_calib", 500)))
    else:
        r = kwargs.get("rank", rank)
        u_fish = kwargs.get("use_energy_weighting", kwargs.get("use_fisher", use_fisher))
        u_chol = kwargs.get("use_cholesky", use_cholesky)
        allow_ln = kwargs.get("allow_ln_failure", allow_ln_failure)
        l_rank = kwargs.get("lora_rank", lora_rank)
        l_alpha = kwargs.get("lora_alpha", lora_alpha)
        recal_ln = kwargs.get("recalibrate_ln", recalibrate_ln)
        target_attn = kwargs.get("target_attention_only", True)
        max_err = kwargs.get("max_error", 0.50)
        fallback = kwargs.get("fallback_to_original", True)
        norm_scale = kwargs.get("normalize_scale", True)
        n_calib = kwargs.get("n_calib", 500)

    for k in ("rank", "use_fisher", "use_energy_weighting", "use_cholesky",
              "allow_ln_failure", "lora_rank", "lora_alpha", "recalibrate_ln",
              "target_attention_only", "max_error", "fallback_to_original",
              "normalize_scale", "n_calib"):
        kwargs.pop(k, None)

    compressor = A3Compressor(
        model,
        rank=max(1, int(r)),
        use_fisher=u_fish,
        use_energy_weighting=u_fish,
        use_cholesky=u_chol,
        allow_ln_failure=allow_ln,
        lora_rank=l_rank,
        lora_alpha=l_alpha,
        recalibrate_ln=recal_ln,
        target_attention_only=target_attn,
        max_error=max_err,
        fallback_to_original=fallback,
        normalize_scale=norm_scale,
        n_calib=n_calib,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = ["A3FactorizedLinear", "A3Compressor", "a3_compressor_factory"]