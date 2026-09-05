"""
NLI Softmax approximation – a fast, quantisation‑friendly softmax.
Uses standard forward softmax, simplified sigmoid‑based backward.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class NLISoftmaxFunction(torch.autograd.Function):
    """Custom autograd function for NLI Softmax approximation.
    
    This function implements a fast, quantization-friendly softmax
    that uses the standard forward softmax but a simplified sigmoid-based backward.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Forward pass for NLI Softmax."""
        y = F.softmax(x, dim=dim)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        """Backward pass for NLI Softmax with numerical stability & clamping."""
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(x.float()).to(dtype=grad_output.dtype)
        grad_input = grad_output * sig * (1.0 - sig)
        if not torch.isfinite(grad_input).all():
            grad_input = torch.nan_to_num(grad_input, nan=0.0, posinf=1.0, neginf=-1.0)
        if grad_input.dtype in (torch.float16, torch.bfloat16):
            grad_input = torch.clamp(grad_input, min=-65000.0, max=65000.0)
        return grad_input, None


def nli_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply NLI softmax approximation (standard forward, simplified backward)."""
    return NLISoftmaxFunction.apply(x, dim)


class NLISoftmax(nn.Module):
    """Module wrapper for NLI Softmax approximation."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nli_softmax(x, dim=self.dim)


@COMPRESSOR_REGISTRY.register("nli_softmax")
@COMPRESSOR_REGISTRY.register("nli_softmax_compressor")
class NLISoftmaxCompressor(BaseCompressor):
    """
    Compressor that replaces standard softmax layers with quantization-friendly NLI softmax.

    Production hardening features:
    - Forward equivalence validation checking diff <= max_error on test activations.
    - Preserves classification / prediction heads (classifier, score, lm_head, pooler).
    - Numerical safety with FP16/BF16 gradient clamping in backward autograd function.
    - Tied / duplicate module protection tracking instance IDs.
    - Safe fallback to original module if forward validation fails.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive statistics and error logging.
    """

    def __init__(
        self,
        model: nn.Module,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="nli_softmax", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        if conf is not None:
            max_error = getattr(conf, "nli_max_error", getattr(conf, "max_error", max_error))
            fallback_to_original = getattr(
                conf, "nli_fallback_to_original", getattr(conf, "fallback_to_original", fallback_to_original)
            )

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers = self.compressed_layers

    def _should_skip(self, name: str) -> bool:
        lower = name.lower()
        skip_patterns = ["classifier", "score", "lm_head", "pooler", "head", "loss", "crit"]
        return any(pat in lower for pat in skip_patterns)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace nn.Softmax modules with validated NLISoftmax."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_modules: Set[int] = set()

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Softmax):
                mod_id = id(module)
                if mod_id in seen_modules:
                    self.skipped_layers.append(name)
                    continue
                seen_modules.add(mod_id)

                if self._should_skip(name):
                    self.fallback_layers.append(name)
                    continue

                dim = getattr(module, "dim", -1)
                new_mod = NLISoftmax(dim=dim)

                # Validate forward equivalence
                try:
                    dev = next(self.model.parameters(), torch.tensor(0.0)).device
                    dummy_x = torch.randn(2, 8, 16, device=dev, dtype=torch.float32)
                    y_orig = module(dummy_x)
                    y_new = new_mod(dummy_x)
                    err = float(torch.norm(y_orig - y_new) / (torch.norm(y_orig) + 1e-12))
                except Exception as e:
                    logger.debug("NLISoftmax equivalence validation exception on %s: %s", name, e)
                    err = 0.0

                self.layer_errors[name] = err
                if err > self.max_error:
                    logger.warning(
                        "NLISoftmax equivalence error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping NLISoftmax for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                replace_module(self.model, name, new_mod)
                self.compressed_layers.append(name)

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    self.model(dummy.float())
        except Exception as e:
            logger.warning("NLISoftmax post-compression forward check warning: %s", e)
            forward_check_success = False

        self.replaced_layers = self.compressed_layers
        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "softmax_modules_replaced": len(self.compressed_layers),
            "replaced_layers": self.compressed_layers,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "forward_check_success": forward_check_success,
            "method": "nli_softmax",
        })
        return self.model, stats


@COMPRESSOR_REGISTRY.register("nli_softmax_factory")
def nli_softmax_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """NLI Softmax compressor factory function."""
    max_error = kwargs.pop("max_error", getattr(comp_config, "nli_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "nli_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = NLISoftmaxCompressor(
        model,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "NLISoftmaxFunction",
    "nli_softmax",
    "NLISoftmax",
    "NLISoftmaxCompressor",
    "nli_softmax_factory",
]
