"""
Swift-SVD Activation-Aware Compressor via cross-covariance / activation subspace projection.
Registered as 'swift_svd_act'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("swift_svd_act")
class SwiftSVDActivationCompressor(BaseCompressor):
    """
    Swift-SVD Activation-Aware Compressor.

    Captures empirical input-output cross-covariances (C_XY = Y^T X / N) across calibration data,
    derives activation-aligned singular subspaces, projects original weights onto the principal
    activation subspace, and replaces dense nn.Linear / Conv1D layers with factorized SVDLinear modules.

    Args:
        model: Model to compress.
        tokenizer: Optional tokenizer for text calibration.
        calibration_loader: Optional DataLoader for calibration.
        calibration_texts: Optional list of calibration text strings.
        rank: Target low-rank bottleneck dimension r.
        target_modules: Optional list of module name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        rank: int = 64,
        max_error: float = 0.5,
        fallback_to_original: bool = False,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="swift_svd_act", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "swift_svd_act_rank", getattr(cfg, "rank", rank))
            tokenizer = getattr(cfg, "tokenizer", tokenizer)
            calibration_loader = getattr(cfg, "calibration_loader", calibration_loader)
            calibration_texts = getattr(cfg, "calibration_texts", calibration_texts)
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.tokenizer = tokenizer
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.rank = int(rank)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _collect_activations(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect input and output activation tensors from calibration passes."""
        activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[Any, ...], outp: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x_in = inp[0].detach()
                x_out = outp.detach() if isinstance(outp, torch.Tensor) else outp[0].detach()
                activations[name] = (x_in.cpu(), x_out.cpu())

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

        device = next(self.model.parameters()).device
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
                        break
                elif self.tokenizer is not None and self.calibration_texts is not None:
                    inputs = self.tokenizer(
                        self.calibration_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    self.model(**inputs)
        except Exception as e:
            logger.warning(
                "SwiftSVDActivation: Calibration forward failed (%s). Falling back to standard SVD.",
                e,
            )
        finally:
            for h in hooks:
                h.remove()

        return activations

    def compress_linear(
        self,
        layer: nn.Module,
        act_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[SVDLinear]:
        """Compress a single linear or Conv1D layer using activation subspace projection."""
        raw_w = layer.weight.data
        cls_name = layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(layer, "in_features")
            and layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().float()
            out_f = W.shape[0]
            in_f = W.shape[1]
        else:
            W = raw_w.float()
            out_f = getattr(layer, "out_features", W.shape[0])
            in_f = getattr(layer, "in_features", W.shape[1])

        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        r = max(1, min(self.rank, max_r, param_cap))

        device = layer.weight.device
        dtype = layer.weight.dtype

        A: Optional[torch.Tensor] = None
        B: Optional[torch.Tensor] = None

        if act_pair is not None:
            try:
                inp, out = act_pair
                inp_flat = inp.reshape(-1, inp.size(-1)).float().to(device)
                out_flat = out.reshape(-1, out.size(-1)).float().to(device)

                if inp_flat.size(-1) == in_f and out_flat.size(-1) == out_f:
                    n_tokens = max(1, inp_flat.size(0))
                    C = (out_flat.t() @ inp_flat) / n_tokens  # [out_f, in_f]

                    U_c, _, Vt_c = torch.linalg.svd(C, full_matrices=False)
                    U_act = U_c[:, :r]            # [out_f, r]
                    V_act = Vt_c[:r, :].t()       # [in_f, r]

                    # Project W onto activation subspace: M = U_act^T @ W @ V_act
                    M = U_act.t() @ W @ V_act     # [r, r]
                    U_m, S_m, Vt_m = torch.linalg.svd(M, full_matrices=False)
                    sqrt_S_m = torch.sqrt(torch.clamp(S_m, min=1e-12))

                    A = (U_act @ U_m) * sqrt_S_m.unsqueeze(0)   # [out_f, r]
                    B = sqrt_S_m.unsqueeze(1) * (Vt_m @ V_act.t()) # [r, in_f]
            except Exception as e:
                logger.debug("Activation subspace projection failed (%s). Falling back to direct SVD.", e)
                A, B = None, None

        if A is None or B is None:
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                A = U[:, :r] * sqrt_S.unsqueeze(0)
                B = sqrt_S.unsqueeze(1) * Vt[:r, :]
            except Exception as e:
                logger.warning("SwiftSVDActivation SVD failure: %s", e)
                return None

        # Frobenius reconstruction error check
        W_rec = torch.matmul(A, B)
        norm_orig = torch.norm(W, p="fro")
        rel_err = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()

        if self.fallback_to_original and rel_err > self.max_error:
            logger.warning(
                "SwiftSVDActivation reconstruction error %.4f exceeded threshold %.4f. Keeping original.",
                rel_err,
                self.max_error,
            )
            return None

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        if layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data)

        svd_layer.reconstruction_error = rel_err
        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers in the model."""
        self.layer_errors = {}
        self.replaced_layers = []

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

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
                replace_module(self.model, name, svd_layer)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["layer_errors"] = self.layer_errors
        stats["compressed_layers"] = len(self.replaced_layers)
        return self.model, stats


def swift_svd_act_factory(
    model: nn.Module, comp_config: Any = None, **kwargs: Any
) -> nn.Module:
    """Swift-SVD activation-aware factory function."""
    compressor = SwiftSVDActivationCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "SwiftSVDActivationCompressor",
    "swift_svd_act_factory",
]
