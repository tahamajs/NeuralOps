"""
Swift-SVD Activation-Aware Compressor via cross-covariance / activation subspace projection.
Registered as 'swift_svd_act'.

Bug-fix map:
  SW-FIX-01  rank computation respects param_cap strictly.
  SW-FIX-02  Multiple calibration batches supported.
  SW-FIX-03  LoRA rank detection robust across r / lora_r / lora_A shape.
  SW-FIX-04  Online cross-covariance accumulation (no CPU memory blow-up).
  SW-FIX-05  Forward sanity check simplified (no .float() fallback).
  SW-FIX-06  Hooks always cleaned up via try/finally.
  SW-FIX-07  cross_cov computed in float64 for numerical stability.
  SW-FIX-08  weight-sharing detection robust (base_layer.weight fallback).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear
from framework.compression_engine.pruning_utils import get_calibration_tensors

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("swift_svd_act")
@COMPRESSOR_REGISTRY.register("swift_svd")
@COMPRESSOR_REGISTRY.register("swift_svd_compressor")
class SwiftSVDActivationCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Swift-SVD Activation-Aware Compressor.

    Captures empirical input-output cross-covariances (C_XY = Y^T X / N) across calibration data,
    derives activation-aligned singular subspaces, projects original weights onto the principal
    activation subspace, and replaces dense nn.Linear / Conv1D layers with factorized SVDLinear modules.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        rank: int = 64,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        min_rank: int = 4,
        target_modules: Optional[List[str]] = None,
        num_calibration_batches: int = 8,
        max_calibration_tokens: int = 32768,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="swift_svd_act", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "swift_svd_act_rank", _cfg_get(cfg, "rank", rank))
            tokenizer = _cfg_get(cfg, "tokenizer", tokenizer)
            calibration_loader = _cfg_get(cfg, "calibration_loader", calibration_loader)
            calibration_texts = _cfg_get(cfg, "calibration_texts", calibration_texts)
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)
            num_calibration_batches = _cfg_get(cfg, "num_calibration_batches", num_calibration_batches)
            max_calibration_tokens = _cfg_get(cfg, "max_calibration_tokens", max_calibration_tokens)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.tokenizer = tokenizer
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.rank = int(rank)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.min_rank = max(1, int(min_rank))
        self.target_modules = target_modules
        self.num_calibration_batches = max(1, int(num_calibration_batches))
        self.max_calibration_tokens = max(1, int(max_calibration_tokens))

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
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
        name_l = name.lower()
        if any(k in name_l for k in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
            return False

        if self.target_attention_only:
            attn_keywords = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'c_attn', 'c_proj', 'query', 'key', 'value']
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False
        if isinstance(base_mod, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    # Calibration batch normalization (SW-FIX-02)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_calibration_batches(calib_inputs: Any) -> List[Any]:
        """Convert whatever get_calibration_tensors returns into a list of batches."""
        if calib_inputs is None:
            return []
        if isinstance(calib_inputs, torch.Tensor):
            return [calib_inputs]
        if isinstance(calib_inputs, dict):
            return [calib_inputs]
        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []
            first = calib_inputs[0]
            # Heuristic: list of dicts / tensors / tuples ⇒ list of batches.
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            # tuple like (input_ids, attention_mask) ⇒ single batch
            if isinstance(first, (list, tuple)):
                return list(calib_inputs)
            # Fallback: treat whole thing as one batch
            return [calib_inputs]
        return [calib_inputs]

    # ------------------------------------------------------------------
    # Activation collection (SW-FIX-04, SW-FIX-07)
    # ------------------------------------------------------------------
    def _collect_activations(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Collect activations and accumulate cross-covariance C = E[Y^T X] per layer
        directly inside the hook (online). Returns dict {name: (cross_cov, n_tokens)}.

        cross_cov shape: [out_features, in_features] in fp32.
        """
        accum: Dict[str, Dict[str, Any]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[Any, ...], outp: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x_in = inp[0].detach()
                if isinstance(outp, torch.Tensor):
                    x_out = outp.detach()
                elif isinstance(outp, (tuple, list)) and len(outp) > 0 and isinstance(outp[0], torch.Tensor):
                    x_out = outp[0].detach()
                else:
                    return

                in_f = x_in.shape[-1]
                out_f = x_out.shape[-1]

                x_in_2d = x_in.reshape(-1, in_f).float()
                x_out_2d = x_out.reshape(-1, out_f).float()

                # Cap tokens per batch
                n = x_in_2d.shape[0]
                if n > self.max_calibration_tokens:
                    x_in_2d = x_in_2d[: self.max_calibration_tokens]
                    x_out_2d = x_out_2d[: self.max_calibration_tokens]
                    n = self.max_calibration_tokens

                if n == 0:
                    return

                if name not in accum:
                    accum[name] = {
                        "cov": torch.zeros(out_f, in_f, dtype=torch.float64),
                        "n": 0,
                        "in_f": in_f,
                        "out_f": out_f,
                    }

                # Cross-covariance in fp64 for numerical stability
                cov_local = (
                    x_out_2d.t().to(torch.float64) @ x_in_2d.to(torch.float64)
                ).cpu()
                accum[name]["cov"] += cov_local
                accum[name]["n"] += n
            return hook

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            return {}

        for name, mod in target_layers:
            hooks.append(mod.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters(), torch.tensor(0.0)).device
        self.model.eval()

        try:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=device,
            )
            batches = self._normalize_calibration_batches(calib_inputs)
            n_batches = min(len(batches), self.num_calibration_batches)

            if n_batches == 0:
                logger.warning("SwiftSVDActivation: no calibration batches available.")
                return {}

            with torch.no_grad():
                for i in range(n_batches):
                    batch = batches[i]
                    try:
                        if isinstance(batch, dict):
                            self.model(**batch)
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch)
                        elif isinstance(batch, (list, tuple)):
                            self.model(*batch)
                    except Exception as e:
                        logger.debug("Calibration batch %d forward failed: %s", i, e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, data in accum.items():
            n = max(1, data["n"])
            cov = (data["cov"] / n).float()
            aggregated[name] = (cov, torch.tensor(n))
        return aggregated

    # ------------------------------------------------------------------
    # LoRA rank detection (SW-FIX-03)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_lora_rank(module: nn.Module) -> int:
        """Return existing LoRA rank from a wrapper if present, else 0."""
        for attr in ("r", "lora_r", "rank"):
            val = getattr(module, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        la = getattr(module, "lora_A", None)
        if la is not None:
            if hasattr(la, "out_features"):
                try:
                    return int(la.out_features)
                except Exception:
                    pass
            if hasattr(la, "weight") and la.weight is not None and la.weight.dim() >= 1:
                return int(la.weight.shape[0])
            if isinstance(la, torch.Tensor) and la.dim() >= 1:
                return int(la.shape[0])
        return 0

    # ------------------------------------------------------------------
    # Per-layer compression
    # ------------------------------------------------------------------
    def compress_linear(
        self,
        layer: nn.Module,
        act_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[SVDLinear]:
        """Compress a single linear or Conv1D layer via Swift-SVD activation projection."""
        base_layer = getattr(layer, "base_layer", layer)
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
        out_f, in_f = W.shape
        max_r = min(out_f, in_f)

        # SW-FIX-01: strict param_cap clamping
        if (in_f * out_f) > 16:
            param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        else:
            param_cap = max_r
        param_cap = max(1, min(param_cap, max_r))

        r_upper = min(self.rank, max_r, param_cap)
        r_lower = min(self.min_rank, max_r, param_cap)
        r = max(r_upper, r_lower)
        r = max(1, min(r, param_cap))

        device = raw_w.device
        dtype = raw_w.dtype

        A, B = None, None

        # SW-FIX-04: act_pair is (cross_cov, n_tokens), not (x_in, x_out)
        if act_pair is not None:
            try:
                cov, _n = act_pair
                if cov.shape[0] == out_f and cov.shape[1] == in_f:
                    C = cov.to(device=device, dtype=torch.float32)

                    U_c, _, Vt_c = torch.linalg.svd(C, full_matrices=False)
                    r_act = min(r, U_c.shape[1], Vt_c.shape[0])
                    if r_act > 0:
                        U_act = U_c[:, :r_act]
                        V_act = Vt_c[:r_act, :].t().contiguous()

                        # Project W onto activation subspace: M = U_act^T @ W @ V_act
                        M = U_act.t().contiguous() @ W @ V_act
                        U_m, S_m, Vt_m = torch.linalg.svd(M, full_matrices=False)
                        r_act = min(r_act, len(S_m))
                        if r_act > 0:
                            sqrt_S_m = torch.sqrt(torch.clamp(S_m[:r_act], min=1e-12))
                            A = (U_act @ U_m[:, :r_act]) * sqrt_S_m.unsqueeze(0)
                            B = sqrt_S_m.unsqueeze(1) * (Vt_m[:r_act, :] @ V_act.t().contiguous())
                            r = r_act
            except Exception as e:
                logger.debug("Activation subspace projection failed (%s). Falling back to direct SVD.", e)
                A, B = None, None

        if A is None or B is None:
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(r, len(S))
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                A = U[:, :r] * sqrt_S.unsqueeze(0)
                B = sqrt_S.unsqueeze(1) * Vt[:r, :]
            except Exception as e:
                logger.warning("SwiftSVDActivation SVD failure: %s", e)
                return None

        # Frobenius reconstruction error
        W_rec = torch.matmul(A, B)
        norm_orig = torch.norm(W, p="fro").item()
        rel_err = (torch.norm(W - W_rec) / max(norm_orig, 1e-12)).item()

        if not math.isfinite(rel_err) or rel_err > self.max_error:
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r_direct = min(r, len(S))
                sqrt_S = torch.sqrt(torch.clamp(S[:r_direct], min=1e-12))
                A_direct = U[:, :r_direct] * sqrt_S.unsqueeze(0)
                B_direct = sqrt_S.unsqueeze(1) * Vt[:r_direct, :]
                W_rec_direct = torch.matmul(A_direct, B_direct)
                rel_err_direct = (torch.norm(W - W_rec_direct) / max(norm_orig, 1e-12)).item()
                if rel_err_direct <= self.max_error or rel_err_direct < rel_err:
                    A, B, r, rel_err = A_direct, B_direct, r_direct, rel_err_direct
            except Exception as e:
                logger.debug("Direct SVD fallback failed: %s", e)

        if not math.isfinite(rel_err) or (self.fallback_to_original and rel_err > self.max_error):
            logger.warning(
                "SwiftSVDActivation reconstruction error %.4f exceeded threshold %.4f. Keeping original.",
                rel_err, self.max_error,
            )
            return None

        effective_lora_r = self.lora_r
        detected = self._detect_lora_rank(layer)
        if detected > 0:
            effective_lora_r = detected

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=base_layer.bias is not None,
            dtype=dtype,
            device=device,
            lora_r=effective_lora_r,
            lora_alpha=self.lora_alpha,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        if base_layer.bias is not None and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(base_layer.bias.data)

        # Transfer existing LoRA if applicable
        if effective_lora_r > 0 and getattr(svd_layer, "use_lora", False):
            try:
                src_lA = getattr(layer, "lora_A", None)
                src_lB = getattr(layer, "lora_B", None)
                if src_lA is not None and src_lB is not None:
                    src_lA_w = getattr(src_lA, "weight", src_lA)
                    src_lB_w = getattr(src_lB, "weight", src_lB)
                    dst_lA_w = getattr(svd_layer.lora_A, "weight", svd_layer.lora_A)
                    dst_lB_w = getattr(svd_layer.lora_B, "weight", svd_layer.lora_B)
                    if src_lA_w.shape == dst_lA_w.shape:
                        dst_lA_w.data.copy_(src_lA_w.data)
                    if src_lB_w.shape == dst_lB_w.shape:
                        dst_lB_w.data.copy_(src_lB_w.data)
            except Exception as e:
                logger.debug("LoRA transfer failed: %s", e)

        svd_layer.reconstruction_error = rel_err
        return svd_layer

    # ------------------------------------------------------------------
    # Weight-sharing detection (SW-FIX-08)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_weight_tensor(module: nn.Module) -> Optional[torch.Tensor]:
        base = getattr(module, "base_layer", module)
        w = getattr(base, "weight", None)
        if w is None:
            return None
        return w

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.replaced_layers = []
        seen_weights: Set[int] = set()

        target_layers: List[Tuple[str, nn.Module]] = []
        for name, mod in list(self.model.named_modules()):
            if not self._is_target_module(name, mod):
                continue
            w = self._get_weight_tensor(mod)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)
            target_layers.append((name, mod))

        if not target_layers:
            logger.warning("SwiftSVDActivation: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        activations = self._collect_activations()

        for name, module in target_layers:
            act_pair = activations.get(name, None)
            svd_layer = self.compress_linear(module, act_pair=act_pair)
            if svd_layer is not None:
                err = getattr(svd_layer, "reconstruction_error", 0.0)
                self.layer_errors[name] = err
                self.replaced_layers.append({"name": name, "rank": svd_layer.rank, "error": err})
                if hasattr(module, "base_layer"):
                    module.base_layer = svd_layer
                    for attr in ("lora_A", "lora_B"):
                        sub = getattr(module, attr, None)
                        if sub is None:
                            continue
                        params = sub.parameters() if hasattr(sub, "parameters") else iter([sub])
                        for p in params:
                            p.requires_grad = True
                else:
                    replace_module(self.model, name, svd_layer)
                self.compressed_layers.append(name)
            else:
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # SW-FIX-05: simplified forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    try:
                        self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                    except Exception as e2:
                        logger.warning("Post-compression forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("Post-compression forward check warning: %s", e)
            forward_check_success = False

        # Param count (defensive)
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "method": "swift_svd_act",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("swift_svd_act_factory")
def swift_svd_act_factory(
    model: nn.Module, comp_config: Any = None, **kwargs: Any
) -> Any:
    """Swift-SVD activation-aware factory function."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "swift_svd_act_rank",
                                       _cfg_get(comp_config, "rank", 64)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop("fallback_to_original",
                          _cfg_get(comp_config, "fallback_to_original", True))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attn = kwargs.pop("target_attention_only",
                             _cfg_get(comp_config, "target_attention_only", False))
    num_calib = kwargs.pop("num_calibration_batches",
                           _cfg_get(comp_config, "num_calibration_batches", 8))
    max_tokens = kwargs.pop("max_calibration_tokens",
                            _cfg_get(comp_config, "max_calibration_tokens", 32768))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = SwiftSVDActivationCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attn,
        num_calibration_batches=num_calib,
        max_calibration_tokens=max_tokens,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SwiftSVDActivationCompressor",
    "swift_svd_act_factory",
]