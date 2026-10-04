"""
Fisher-Weighted Singular Value Decomposition (FWSVD) Compressor.

Computes diagonal Fisher information matrix F_i = E[x_i^2] from calibration activations,
weights the weight matrix W by diag(sqrt(F_i)), computes truncated SVD,
and unweights factors by diag(1 / sqrt(F_i)).
Uses the same structural SVDLinear replacement and fallback error logic as AASVD.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.compression_engine.aa_svd_compressor import SVDLinear
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import get_dynamic_compression_rank, should_compress_linear

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_CORPUS: list[str] = [
    (
        "Transformers rely heavily on multi-head self-attention mechanisms to capture complex syntactic "
        "and semantic dependencies across arbitrary sequence lengths. However, the standard quadratic computational "
        "complexity of softmax attention poses significant computational and memory bandwidth challenges during "
        "both pre-training and long-context autoregressive decoding."
    ),
    (
        "Singular Value Decomposition provides an optimal low-rank subspace approximation under the Frobenius norm. "
        "By preconditioning the weight matrices with empirical input activation covariances, activation-aware SVD "
        "preserves salient outlier directions that prevent catastrophic representation drift in deep neural networks."
    ),
    (
        "def compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:\n"
        "    scores = torch.matmul(q, k.transpose(-1, -2).contiguous()) * scale\n"
        "    probs = torch.softmax(scores, dim=-1)\n"
        "    return torch.matmul(probs, v)\n"
    ),
    (
        "In distributed multi-GPU environments, tensor parallelism and fully sharded data parallelism require "
        "efficient parameter sharding. Low-rank factorized layers reduce inter-node communication volume by communicating "
        "narrow bottleneck activations rather than high-dimensional hidden states across all-reduce collectives."
    ),
    (
        "Mathematical problem solving and chain-of-thought reasoning require high-fidelity preservation of intermediate "
        "residual stream representations. Let f(x) = Wx + b. If W is factorized into U and V with rank r << min(d_in, d_out), "
        "the total number of parameters scales as r*(d_in + d_out) compared to d_in*d_out."
    ),
]


@COMPRESSOR_REGISTRY.register("fwsvd")
@COMPRESSOR_REGISTRY.register("fwsvd_compressor")
class FWSVDCompressor(BaseCompressor):
    """
    Fisher-Weighted SVD Compressor.
    Projects weights using diagonal Fisher activation weights F_i = E[x_i^2].
    Replaces dense nn.Linear with SVDLinear.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        rank: int = 64,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        target_modules: list[str] | None = None,
        target_attention_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="fwsvd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            rank = getattr(cfg, "fwsvd_rank", getattr(cfg, "rank", rank))
            tokenizer = getattr(cfg, "tokenizer", tokenizer)
            calibration_texts = getattr(cfg, "calibration_texts", calibration_texts)
            max_error = getattr(cfg, "fwsvd_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg,
                "fwsvd_fallback_to_original",
                getattr(cfg, "fallback_to_original", fallback_to_original),
            )
            target_attention_only = getattr(
                cfg,
                "fwsvd_target_attention_only",
                getattr(cfg, "target_attention_only", target_attention_only),
            )

        target_attention_only = kwargs.get(
            "target_attention_only",
            kwargs.get("fwsvd_target_attention_only", target_attention_only),
        )
        self.rank = rank
        self.max_error = max_error
        self.fallback_to_original = fallback_to_original
        self.tokenizer = tokenizer
        self.calibration_texts = calibration_texts or DEFAULT_CALIBRATION_CORPUS
        self.target_modules = target_modules
        self.target_attention_only = target_attention_only
        self.device = (
            torch.device(device) if device is not None else next(model.parameters()).device
        )
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, SVDLinear):
            return False
        if any(
            name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout"]
        ):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
            or "loralinear" in cls_name
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False
        name_lower = name.lower()
        if any(
            skip in name_lower
            for skip in [
                "lm_head",
                "classifier",
                "head",
                "output_layer",
                "score",
                "pooler",
            ]
        ):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn",
                "mlp",
                "lin1",
                "lin2",
                "intermediate",
                "gate_proj",
                "up_proj",
                "down_proj",
                "c_fc",
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "out_lin",
                "q_lin",
                "k_lin",
                "v_lin",
                "c_attn",
                "c_proj",
                "query",
                "key",
                "value",
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int | None = None,
        fisher_diag: torch.Tensor | None = None,
    ) -> tuple[SVDLinear | None, float, int]:
        try:
            target_mod = getattr(layer, "base_layer", layer)
            cls_name = layer.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            is_conv1d = (
                "conv1d" in cls_name
                or "conv1d" in target_cls
                or hasattr(layer, "nf")
                or hasattr(target_mod, "nf")
            )
            weight_tensor = getattr(target_mod, "weight", getattr(layer, "weight", None))
            if weight_tensor is None:
                return None, float("inf"), 0

            raw_w = weight_tensor.data

            if is_conv1d:
                in_feat, out_feat = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()
            else:
                out_feat, in_feat = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            r = get_dynamic_compression_rank(
                in_feat,
                out_feat,
                model_or_family=self.model,
                default_rank=rank or self.rank,
                comp_config=getattr(self, "comp_config", None),
            )
            max_allowed_rank = max(
                1,
                min(
                    in_feat,
                    out_feat,
                    int(0.75 * (in_feat * out_feat) / (in_feat + out_feat)),
                ),
            )
            r = min(r, max_allowed_rank)
            W_norm = torch.norm(W, p="fro").item()

            sqrt_f: torch.Tensor | None = None
            inv_sqrt_f: torch.Tensor | None = None

            if fisher_diag is not None and fisher_diag.shape == (in_feat,):
                f_clamped = fisher_diag.to(device=layer.weight.device, dtype=torch.float32).clamp(
                    min=1e-8
                )
                sqrt_f = torch.sqrt(f_clamped)
                inv_sqrt_f = 1.0 / sqrt_f

            if sqrt_f is not None and inv_sqrt_f is not None:
                W_weighted = W * sqrt_f.unsqueeze(0)
            else:
                W_weighted = W

            W_weighted_dev = W_weighted.to(device=layer.weight.device, dtype=torch.float32)
            U, S, Vt = torch.linalg.svd(W_weighted_dev, full_matrices=False)

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = U[:, :r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]

            if inv_sqrt_f is not None:
                V_r = V_r * inv_sqrt_f.unsqueeze(0)

            W_recon = torch.matmul(U_r, V_r)
            err = float(torch.norm(W - W_recon, p="fro").item() / max(W_norm, 1e-12))

            dtype = weight_tensor.dtype
            device = weight_tensor.device
            bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))

            svd_layer = SVDLinear(
                in_features=in_feat,
                out_features=out_feat,
                rank=r,
                bias=bias_tensor is not None,
                dtype=dtype,
                device=device,
                lora_r=getattr(self, "lora_r", 0),
                lora_alpha=getattr(self, "lora_alpha", 1.0),
            )
            svd_layer.set_low_rank_factors(
                U_r.to(device=device, dtype=dtype),
                V_r.to(device=device, dtype=dtype),
            )
            if bias_tensor is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(bias_tensor.data)

            svd_layer.reconstruction_error = err
            svd_layer.rank = r
            return svd_layer, err, r
        except Exception as e:
            logger.warning("FWSVD factorization failed on linear: %s", e)
            return None, float("inf"), 0

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        self.model.eval()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        fishers: dict[str, torch.Tensor] = {}

        if self.tokenizer is None:
            logger.info("No tokenizer provided for FWSVD; proceeding with standard unweighted SVD.")
        else:
            hooks = []

            def make_hook(name: str):
                def hook(module: nn.Module, inp: tuple[torch.Tensor, ...], outp: Any):
                    if not inp or inp[0] is None:
                        return
                    x = inp[0].detach()
                    dev_type = x.device.type
                    autocast_ctx = (
                        torch.autocast(device_type=dev_type, enabled=False)
                        if hasattr(torch, "autocast") and dev_type in ("cuda", "cpu")
                        else contextlib.nullcontext()
                    )
                    with autocast_ctx:
                        target_mod = getattr(module, "base_layer", module)
                        cls_name = module.__class__.__name__.lower()
                        target_cls = target_mod.__class__.__name__.lower()
                        is_conv1d = (
                            "conv1d" in cls_name
                            or "conv1d" in target_cls
                            or hasattr(module, "nf")
                            or hasattr(target_mod, "nf")
                        )
                        weight_tensor = getattr(
                            target_mod, "weight", getattr(module, "weight", None)
                        )
                        if weight_tensor is None:
                            return
                        in_f = weight_tensor.shape[0] if is_conv1d else weight_tensor.shape[1]
                        x_fp32 = x.float()
                        x_flat = x_fp32.reshape(-1, in_f)
                        mask = x_flat.abs().sum(dim=-1) > 1e-5
                        if mask.sum() == 0:
                            return
                        x_active = x_flat[mask]
                        batch_fisher = torch.mean(x_active**2, dim=0).cpu()

                    if name not in fishers:
                        fishers[name] = batch_fisher
                    else:
                        fishers[name] = 0.5 * (fishers[name] + batch_fisher)

                return hook

            for name, module in self.model.named_modules():
                if self._is_target_module(name, module):
                    hooks.append(module.register_forward_hook(make_hook(name)))

            try:
                inputs = self.tokenizer(
                    self.calibration_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
            except Exception as e:
                logger.debug("FWSVD calibration forward pass exception: %s", e)

            for h in hooks:
                h.remove()

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                f_diag = fishers.get(name)
                svd_layer, err, r = self.compress_linear(module, rank=self.rank, fisher_diag=f_diag)
                self.layer_errors[name] = err
                self.layer_ranks[name] = r

                if svd_layer is None or err > self.max_error:
                    if self.fallback_to_original:
                        logger.warning(
                            "FWSVD reconstruction error %.4f on %s exceeded threshold %.4f. Skipping compression (fallback).",
                            err,
                            name,
                            self.max_error,
                        )
                        self.fallback_layers.append(name)
                        continue
                    else:
                        if svd_layer is None:
                            self.fallback_layers.append(name)
                            continue

                if hasattr(module, "base_layer"):
                    module.base_layer = svd_layer
                else:
                    replace_module(self.model, name, svd_layer)
                self.compressed_layers.append(name)

        stats = {
            "compressed_layers": len(self.compressed_layers),
            "fallback_layers": len(self.fallback_layers),
            "avg_reconstruction_error": (
                float(sum(self.layer_errors.values()) / max(1, len(self.layer_errors)))
                if self.layer_errors
                else 0.0
            ),
            "layer_ranks": self.layer_ranks,
        }
        self.model.compression_stats = stats
        return self.model, stats
