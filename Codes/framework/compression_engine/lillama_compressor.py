"""
LiLLaMA: Local Distillation via Activation Cross-Covariance SVD.
Registered as 'lillama_class', 'lillama_compressor_class', 'lillama_factory'
and 'lillama' / 'lillama_compressor' as factory aliases.

Formulation:
  C_YX = (1/N) Y^T X                 (cross-covariance)
  C_YX ≈ U_r Σ_r V_r^T               (truncated SVD)
  W_distill = U_r Σ_r V_r^T          (distilled weight)

Bug-fix map:
  LL-FIX-01  is_custom_attention = False.
  LL-FIX-02  Registry collision resolved (class vs factory).
  LL-FIX-03  Safe device resolution.
  LL-FIX-04  r always bound; never rely on locals().
  LL-FIX-05  Recursion guard in factory.
  LL-FIX-06  Debug logging for no-target diagnostic.
  LL-FIX-07  Forward sanity check.
  LL-FIX-08  Idempotence flag on model.
  LL-FIX-09  skip_should_compress_filter option.
  LL-FIX-10  min_layer_dim option.
  LL-FIX-11  Tokenizer resolution from config.
  LL-FIX-12  Explicit note about parameter storage.
  LL-FIX-13  compression_stats written to model.
  LL-FIX-14  Depth-bounded traversal.
  LL-FIX-15  compressed_layers initialized in __init__.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
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


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "self_attn",
}


def _get_device(model: nn.Module) -> torch.device:
    """LL-FIX-03: safe device resolution."""
    p = next(iter(model.parameters()), None)
    return p.device if p is not None else torch.device("cpu")


def _is_conv1d(module: nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    return "conv1d" in cls or hasattr(module, "nf")


# ===========================================================================
# Compressor (LL-FIX-01, LL-FIX-02)
# ===========================================================================

@COMPRESSOR_REGISTRY.register("lillama_class")
@COMPRESSOR_REGISTRY.register("lillama_compressor_class")
@COMPRESSOR_REGISTRY.register("lillama_distiller_class")
class LiLLaMADistiller(BaseCompressor):
    # LL-FIX-01: this is a local distillation / weight-rewriter, not standard factorized attention.
    is_custom_attention: bool = False
    category: str = "weight_rewriting"

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
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        max_calibration_batches: int = 8,
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
            min_layer_dim = _cfg_get(cfg, "min_layer_dim", min_layer_dim)
            skip_should_compress_filter = _cfg_get(
                cfg, "skip_should_compress_filter", skip_should_compress_filter,
            )
            max_calibration_batches = _cfg_get(cfg, "max_calibration_batches", max_calibration_batches)

        # LL-FIX-11: tokenizer resolution.
        self.tokenizer = (
            tokenizer
            or getattr(model, "tokenizer", None)
            or getattr(getattr(model, "config", None), "tokenizer", None)
        )
        self.rank = max(1, int(rank))
        self.calibration_texts = calibration_texts
        self.calibration_loader = calibration_loader
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.skip_attention = bool(skip_attention)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.skip_should_compress_filter = bool(skip_should_compress_filter)
        self.max_calibration_batches = max(1, int(max_calibration_batches))

        # LL-FIX-03: safe device.
        if device is not None:
            self.target_device = torch.device(device)
        else:
            self.target_device = _get_device(model)

        # LL-FIX-15: initialize all lists in __init__.
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_details: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # Target check (LL-FIX-06)
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False

        # Skip already-compressed.
        cls_name = module.__class__.__name__
        if cls_name in ("LSCPrunedLoRALinear", "DynamicSVDLinear",
                        "SVDLinear", "FactorisedLinear", "LowRankPlusSparseLinear"):
            if _debug:
                logger.info(f"  [skip] {name}: already compressed ({cls_name})")
            return False

        if any(skip in name_lower for skip in
               ["classifier", "score", "lm_head", "pooler",
                "embedding", "router", "gate"]):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_lower = target_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_lower
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
            return False

        # LL-FIX-09: allow bypassing framework filter.
        if not self.skip_should_compress_filter:
            try:
                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        # LL-FIX-10: filter tiny layers.
        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if self.target_attention_only:
            is_attn = any(k in name_lower for k in _ATTN_NAMES)
            if not is_attn:
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
                return False

        if self.skip_attention:
            attn_keywords = {
                "query", "key", "value", "q_lin", "k_lin", "v_lin",
                "q_proj", "k_proj", "v_proj", "out_lin", "out_proj",
                "dense", "o_proj", "c_attn",
            }
            last = name.split(".")[-1].lower()
            if last in attn_keywords or any(
                k in name_lower for k in ("self_attn", "attn.c_attn")
            ):
                if _debug:
                    logger.info(f"  [skip] {name}: skip_attention")
                return False

        if self.target_modules is not None:
            ok = any(t in name for t in self.target_modules)
            if not ok and _debug:
                logger.info(f"  [skip] {name}: not in target_modules")
            return ok

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    # IO activation collection
    # ------------------------------------------------------------------
    def _collect_io_activations(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect input/output activations across target layers."""
        self.model.eval()
        activations: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
                inp = inputs[0].detach() if inputs and isinstance(inputs[0], torch.Tensor) else None
                if isinstance(output, torch.Tensor):
                    out = output.detach()
                elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                    out = output[0].detach()
                else:
                    out = None
                if inp is not None and out is not None:
                    if layer_name not in activations:
                        activations[layer_name] = {"in": [], "out": []}
                    activations[layer_name]["in"].append(inp.cpu())
                    activations[layer_name]["out"].append(out.cpu())
            return hook

        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]
        if not target_list:
            return {}

        for name, module in target_list:
            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )
            batches = self._normalize_batches(calib_inputs)
            n_batches = min(len(batches), self.max_calibration_batches)
            with torch.no_grad():
                for i in range(n_batches):
                    batch = batches[i]
                    try:
                        if isinstance(batch, dict):
                            batch = {k: v.to(self.target_device) if isinstance(v, torch.Tensor) else v
                                     for k, v in batch.items()}
                            self.model(**batch)
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch.to(self.target_device))
                        elif isinstance(batch, (list, tuple)):
                            batch = [v.to(self.target_device) if isinstance(v, torch.Tensor) else v
                                     for v in batch]
                            self.model(*batch)
                    except Exception as e:
                        logger.debug("Calibration batch %d failed: %s", i, e)
        except Exception as e:
            logger.warning("LiLLaMA: calibration forward raised: %s", e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        MAX_ACTIVATION_SAMPLES = 512
        for name, data in activations.items():
            if data["in"] and data["out"]:
                try:
                    cat_in = torch.cat(data["in"], dim=0)
                    cat_out = torch.cat(data["out"], dim=0)
                    if cat_in.size(0) > MAX_ACTIVATION_SAMPLES:
                        cat_in = cat_in[:MAX_ACTIVATION_SAMPLES]
                        cat_out = cat_out[:MAX_ACTIVATION_SAMPLES]
                    aggregated[name] = (cat_in, cat_out)
                except Exception as e:
                    logger.debug("Activation concat failed for %s: %s", name, e)
        return aggregated

    @staticmethod
    def _normalize_batches(calib_inputs: Any) -> List[Any]:
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
            if isinstance(first, (dict, torch.Tensor)):
                return list(calib_inputs)
            return [calib_inputs]
        return [calib_inputs]

    # ------------------------------------------------------------------
    # SVD factorization helper
    # ------------------------------------------------------------------
    def _factorize(
        self, W: torch.Tensor, rank: int,
    ) -> Tuple[torch.Tensor, int]:
        """Return (W_recon, rank_used)."""
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            total_energy = (S ** 2).sum().clamp(min=1e-12)
            cum_ratio = torch.cumsum(S ** 2, 0) / total_energy
            r = min(rank, len(S))
            cum_e = cum_ratio[r - 1].item() if r > 0 else 0.0
            if cum_e < 0.95:
                idx = torch.where(cum_ratio >= 0.95)[0]
                if len(idx) > 0:
                    r = min(int(idx[0].item()) + 1, len(S))
            W_recon = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
            return W_recon, r
        except Exception as e:
            logger.debug("LiLLaMA SVD factorization failed: %s", e)
            return W, 0

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        # LL-FIX-08: idempotence.
        if getattr(self.model, "_lillama_compressed", False):
            logger.warning("LiLLaMA: model already compressed; skipping.")
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_details = {}

        activations_map = self._collect_io_activations()
        logger.info("LiLLaMA: %d target activation sets collected.", len(activations_map))

        # LL-FIX-06: diagnostic if no targets.
        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]
        if not target_list:
            logger.warning("LiLLaMA: no compressible layers found. Running diagnostic...")
            n_scanned = 0
            for name, module in self.model.named_modules():
                n_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning("LiLLaMA diagnostic: %d modules scanned, 0 targets.", n_scanned)
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        logger.info("LiLLaMA: %d target layers identified.", len(target_list))

        total_orig_params = 0
        total_comp_params = 0

        # Deduplicated parameter accounting.
        seen_param_ids: set = set()
        count_param_ids: set = set()

        for p in self.model.parameters():
            if id(p) not in seen_param_ids:
                seen_param_ids.add(id(p))
                total_orig_params += p.numel()

        def _count_params_of(mod: nn.Module):
            nonlocal total_comp_params
            for p in mod.parameters(recurse=False):
                if id(p) not in count_param_ids:
                    count_param_ids.add(id(p))
                    total_comp_params += p.numel()

        for name, module in target_list:
            target_layer = getattr(module, "base_layer", module)
            raw_w = target_layer.weight.data
            is_conv1d = _is_conv1d(target_layer)
            device = raw_w.device
            dtype = target_layer.weight.dtype

            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            used_cross_cov = False
            r_used = self.rank

            # LL-FIX-04: initialize W_new to W (safe default).
            W_new = W

            if name in activations_map:
                inp, out = activations_map[name]
                try:
                    inp_flat = inp.reshape(-1, inp.shape[-1]).float()
                    out_flat = out.reshape(-1, out.shape[-1]).float()
                    if inp_flat.shape[-1] == d_in and out_flat.shape[-1] == d_out:
                        C = torch.matmul(out_flat.t().contiguous(), inp_flat) / max(len(inp_flat), 1)
                        W_new, r_used = self._factorize(C, self.rank)
                        used_cross_cov = True
                    else:
                        W_new, r_used = self._factorize(W, self.rank)
                except Exception as e:
                    logger.debug("Cross-covariance factorization failed for %s: %s", name, e)
                    W_new, r_used = self._factorize(W, self.rank)
            else:
                W_new, r_used = self._factorize(W, self.rank)

            norm_orig = torch.norm(W, p="fro")
            rel_err = (
                torch.norm(W - W_new, p="fro") / (norm_orig + 1e-8)
            ).item()
            self.layer_errors[name] = rel_err

            if rel_err > self.max_error:
                logger.warning(
                    "LiLLaMA: err %.4f on %s > %.4f. %s",
                    rel_err, name, self.max_error,
                    "Keeping original." if self.fallback_to_original
                    else "Proceeding with distilled weights.",
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False
                    _count_params_of(module)
                    continue

            # Shape integrity check.
            final_w = W_new.t().contiguous() if is_conv1d else W_new
            if final_w.shape != target_layer.weight.shape:
                logger.warning(
                    "LiLLaMA: shape mismatch for %s: expected %s, got %s; skipping.",
                    name, tuple(target_layer.weight.shape), tuple(final_w.shape),
                )
                _count_params_of(module)
                continue

            target_layer.weight.data.copy_(final_w.to(dtype=dtype, device=device))
            target_layer.weight.requires_grad = False
            if getattr(target_layer, "bias", None) is not None:
                target_layer.bias.requires_grad = False

            # Preserve LoRA adapter trainability.
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for attr in ("lora_A", "lora_B"):
                    p = getattr(module, attr, None)
                    if p is None:
                        continue
                    if hasattr(p, "weight"):
                        p.weight.requires_grad = True
                    elif isinstance(p, nn.Parameter):
                        p.requires_grad = True

            self.compressed_layers.append(name)
            _count_params_of(module)

            self.layer_details[name] = {
                "rank": int(r_used),
                "rel_error": float(rel_err),
                "used_cross_cov": bool(used_cross_cov),
                "shape": tuple(final_w.shape),
            }

        # LL-FIX-07: forward sanity check.
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy = torch.randint(0, 100, (1, 8), device=self.target_device, dtype=torch.long)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("LiLLaMA: forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("LiLLaMA: forward check raised: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        # LL-FIX-04: avg error only over compressed layers.
        compressed_errs = [self.layer_errors[n] for n in self.compressed_layers
                           if n in self.layer_errors]
        avg_err = (
            sum(compressed_errs) / len(compressed_errs)
            if compressed_errs else 0.0
        )
        eff_ratio = total_comp_params / max(1, total_orig_params)

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update({
            "distilled_layers": len(self.compressed_layers),
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "rank": self.rank,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "layer_errors": self.layer_errors,
            "layer_details": self.layer_details,
            "effective_parameter_ratio": eff_ratio,
            "forward_check_success": forward_check_success,
            "method": "lillama_cross_covariance_svd",
            # LL-FIX-12: honest note.
            "note": (
                "LiLLaMA rewrites weights in-place with the distilled reconstruction. "
                "The dense weight keeps its full (out, in) shape; framework-level "
                "compression_rate reads 1.0. Use effective_parameter_ratio for the "
                "theoretical storage metric."
            ),
        })
        self.stats = stats

        # LL-FIX-13: write stats onto model.
        try:
            if not hasattr(self.model, "compression_stats") or self.model.compression_stats is None:
                self.model.compression_stats = {}
            self.model.compression_stats.update({
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "compression_method": "lillama_cross_covariance_svd",
                "effective_parameter_ratio": eff_ratio,
                "avg_reconstruction_error": avg_err,
            })
        except Exception:
            pass

        self.model._lillama_compressed = True
        logger.info(
            "LiLLaMA: %d distilled, %d fallback. Avg err %.4f, eff_param_ratio %.2fx.",
            len(self.compressed_layers), len(self.fallback_layers), avg_err, eff_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory (LL-FIX-02, LL-FIX-05)
# ===========================================================================

@COMPRESSOR_REGISTRY.register("lillama")
@COMPRESSOR_REGISTRY.register("lillama_compressor")
@COMPRESSOR_REGISTRY.register("lillama_factory")
def lillama_factory(
    model: Any, comp_config: Any = None, _depth: int = 0, **kwargs: Any,
) -> Any:
    """Factory function for LiLLaMA compressor."""
    if _depth > 4:
        logger.warning("lillama_factory: recursion depth exceeded.")
        return model

    rank = _cfg_get(comp_config, "lillama_rank",
                    _cfg_get(comp_config, "rank", kwargs.pop("rank", 32)))
    max_error = _cfg_get(comp_config, "max_error", kwargs.pop("max_error", 0.20))
    fallback_to_original = _cfg_get(
        comp_config, "fallback_to_original",
        kwargs.pop("fallback_to_original", True),
    )
    skip_attention = _cfg_get(comp_config, "skip_attention",
                              kwargs.pop("skip_attention", False))
    target_attention_only = _cfg_get(
        comp_config, "target_attention_only",
        kwargs.pop("target_attention_only", False),
    )
    skip_filter = _cfg_get(
        comp_config, "skip_should_compress_filter",
        kwargs.pop("skip_should_compress_filter", False),
    )
    min_layer_dim = _cfg_get(comp_config, "min_layer_dim",
                             kwargs.pop("min_layer_dim", 8))
    tokenizer = (
        kwargs.pop("tokenizer", None)
        or getattr(model, "tokenizer", None)
        or _cfg_get(comp_config, "tokenizer", None)
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = LiLLaMACompressor(
        model,
        tokenizer=tokenizer,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        skip_attention=skip_attention,
        target_attention_only=target_attention_only,
        skip_should_compress_filter=skip_filter,
        min_layer_dim=min_layer_dim,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


LiLLaMACompressor = LiLLaMADistiller

__all__ = [
    "LiLLaMADistiller",
    "LiLLaMACompressor",
    "lillama_factory",
]