#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FWSVD: Fisher-Weighted Singular Value Decomposition

Weights weight parameters with empirical Fisher information or activation covariance
whitening pullback before computing low-rank singular factorizations.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.compression.aa_svd_compressor import SVDLinear
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


@COMPRESSOR_REGISTRY.register("fwsvd_compressor")
@COMPRESSOR_REGISTRY.register("fwsvd", is_alias=True)
@COMPRESSOR_REGISTRY.register("fisher_weighted_svd", is_alias=True)
class FWSVDCompressor(BaseCompressor):
    """
    Fisher-Weighted SVD Compressor.

    Preconditions weight matrices using true empirical Fisher Information
    (gradient outer-product F = (1/M) sum g_i g_i^T) or activation covariance
    whitening pullback (C_X = (1/M) X^T X + lambda * I) before SVD factorization.
    Replaces dense nn.Linear with SVDLinear.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        rank: int = 64,
        max_error: float = 1.0,
        fallback_to_original: bool = False,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        target_modules: list[str] | None = None,
        target_attention_only: bool = True,
        damping: float = 1e-4,
        use_gradients: bool = True,
        loss_fn: Any | None = None,
        gradients: dict[str, torch.Tensor] | None = None,
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
            damping = getattr(cfg, "fwsvd_damping", getattr(cfg, "damping", damping))
            use_gradients = getattr(
                cfg, "fwsvd_use_gradients", getattr(cfg, "use_gradients", use_gradients)
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
        self.damping = max(1e-7, float(damping))
        self.use_gradients = bool(use_gradients)
        self.loss_fn = loss_fn or kwargs.get("loss_fn", None)
        self.gradients = gradients or kwargs.get("gradients", None)
        self.device = (
            torch.device(device) if device is not None else next(model.parameters()).device
        )
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.fisher_matrices: dict[str, torch.Tensor] = {}
        self.fishers: dict[str, torch.Tensor] = {}

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
        fisher_matrix: torch.Tensor | None = None,
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

            F_mat = fisher_matrix
            if F_mat is None and fisher_diag is not None and fisher_diag.ndim == 2:
                F_mat = fisher_diag

            if F_mat is not None and F_mat.shape == (in_feat, in_feat):
                F_dev = F_mat.to(device=layer.weight.device, dtype=torch.float32)
                F_sym = 0.5 * (F_dev + F_dev.t())
                tr_F = torch.trace(F_sym).item()
                damp_val = max(1e-7, float(self.damping * tr_F / max(1, in_feat)))
                F_damped = F_sym + damp_val * torch.eye(
                    in_feat, device=F_dev.device, dtype=torch.float32
                )

                try:
                    eigvals, eigvecs = torch.linalg.eigh(F_damped)
                    eigvals_clamped = torch.clamp(eigvals, min=damp_val)
                    sqrt_eig = torch.sqrt(eigvals_clamped)
                    inv_sqrt_eig = 1.0 / sqrt_eig
                    F_sqrt = eigvecs @ torch.diag(sqrt_eig) @ eigvecs.t()
                    F_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eig) @ eigvecs.t()

                    W_weighted = W @ F_sqrt
                    U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]) @ F_inv_sqrt
                except Exception as e:
                    logger.debug(
                        "Full Fisher whitening SVD failed (%s); falling back to unweighted SVD", e
                    )
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                    V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
            elif fisher_diag is not None and fisher_diag.shape == (in_feat,):
                f_clamped = fisher_diag.to(device=layer.weight.device, dtype=torch.float32).clamp(
                    min=1e-8
                )
                sqrt_f = torch.sqrt(f_clamped)
                inv_sqrt_f = 1.0 / sqrt_f
                W_weighted = W * sqrt_f.unsqueeze(0)
                U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]) * inv_sqrt_f.unsqueeze(0)
            else:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]

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

    def compress(
        self,
        calibration_data: Any | None = None,
        gradients: dict[str, torch.Tensor] | None = None,
        loss_fn: Any | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.fisher_matrices = {}
        self.fishers = {}

        user_grads = gradients or self.gradients
        target_layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]

        if not target_layers:
            logger.warning("FWSVD: no compressible linear layers found.")
            return self.model, {"compressed_layers": 0, "fallback_layers": 0}

        # Step 1: Ingest pre-supplied gradients if provided
        remaining_layers = []
        for name, module in target_layers:
            if user_grads is not None and name in user_grads:
                g_val = user_grads[name].detach().float()
                target_mod = getattr(module, "base_layer", module)
                cls_name = module.__class__.__name__.lower()
                target_cls = target_mod.__class__.__name__.lower()
                is_conv1d = (
                    "conv1d" in cls_name
                    or "conv1d" in target_cls
                    or hasattr(module, "nf")
                    or hasattr(target_mod, "nf")
                )
                weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
                if weight_tensor is None:
                    continue
                in_f = weight_tensor.shape[0] if is_conv1d else weight_tensor.shape[1]

                if g_val.ndim == 2 and g_val.shape == (in_f, in_f):
                    F_mat = g_val
                elif g_val.ndim == 2 and g_val.shape[1] == in_f:
                    # g_val is (M, in_f) gradient matrix => F = (1/M) * G^T G
                    M_samples = max(1, g_val.shape[0])
                    F_mat = (g_val.t() @ g_val) / float(M_samples)
                elif g_val.ndim == 1 and g_val.shape[0] == in_f:
                    F_mat = torch.diag(g_val)
                else:
                    logger.warning(
                        "Supplied gradient shape %s invalid for layer %s (in_feat=%d); skipping",
                        tuple(g_val.shape),
                        name,
                        in_f,
                    )
                    remaining_layers.append((name, module))
                    continue

                tr_F = torch.trace(F_mat).item()
                damp_val = max(1e-7, float(self.damping * tr_F / max(1, in_f)))
                F_damped = 0.5 * (F_mat + F_mat.t()) + damp_val * torch.eye(
                    in_f, device=F_mat.device, dtype=torch.float32
                )
                self.fisher_matrices[name] = F_damped.cpu()
                self.fishers[name] = torch.diag(F_damped).cpu()
            else:
                remaining_layers.append((name, module))

        # Step 2: Compute empirical Fisher / activation covariance for remaining layers
        if remaining_layers:
            act_cov: dict[str, torch.Tensor] = {}
            act_counts: dict[str, int] = {}
            grad_cov: dict[str, torch.Tensor] = {}
            grad_counts: dict[str, int] = {}
            captured_inputs: dict[str, list[torch.Tensor]] = {}

            hooks = []

            def make_hook(layer_name: str):
                def hook(module: nn.Module, inp: tuple[torch.Tensor, ...], outp: Any):
                    if not inp or inp[0] is None:
                        return
                    x = inp[0]
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

                    x_det = x.detach()
                    x_fp32 = x_det.float().reshape(-1, in_f)
                    mask = x_fp32.abs().sum(dim=-1) > 1e-5
                    if mask.sum() == 0:
                        return
                    x_active = x_fp32[mask]

                    batch_cov = (x_active.t() @ x_active).cpu()
                    n_samples = x_active.shape[0]
                    if layer_name not in act_cov:
                        act_cov[layer_name] = batch_cov
                        act_counts[layer_name] = n_samples
                    else:
                        act_cov[layer_name] += batch_cov
                        act_counts[layer_name] += n_samples

                    if self.use_gradients and torch.is_grad_enabled() and x.requires_grad:
                        x.retain_grad()
                        captured_inputs.setdefault(layer_name, []).append(x)

                return hook

            for name, module in remaining_layers:
                hooks.append(module.register_forward_hook(make_hook(name)))

            # Execute forward calibration (and backward if gradient computation is possible)
            try:
                active_loss_fn = loss_fn or self.loss_fn
                if calibration_data is not None:
                    if isinstance(calibration_data, torch.Tensor):
                        batches = [calibration_data]
                    elif isinstance(calibration_data, (list, tuple)):
                        batches = list(calibration_data)
                    else:
                        batches = [calibration_data]

                    for batch in batches:
                        captured_inputs.clear()
                        if isinstance(batch, torch.Tensor):
                            b_dev = batch.to(self.device)
                            if self.use_gradients:
                                with torch.enable_grad():
                                    out = self.model(b_dev)
                                    loss = (
                                        active_loss_fn(out)
                                        if active_loss_fn is not None
                                        else getattr(out, "loss", None)
                                    )
                                    if loss is not None and loss.requires_grad:
                                        loss.backward()
                            else:
                                with torch.no_grad():
                                    self.model(b_dev)
                        elif isinstance(batch, dict):
                            b_dev = {
                                k: v.to(self.device)
                                for k, v in batch.items()
                                if isinstance(v, torch.Tensor)
                            }
                            if self.use_gradients:
                                with torch.enable_grad():
                                    out = self.model(**b_dev)
                                    loss = (
                                        active_loss_fn(out)
                                        if active_loss_fn is not None
                                        else getattr(out, "loss", None)
                                    )
                                    if loss is not None and loss.requires_grad:
                                        loss.backward()
                            else:
                                with torch.no_grad():
                                    self.model(**b_dev)

                        # Accumulate gradients if backward succeeded
                        for l_name, tensor_list in captured_inputs.items():
                            for x_t in tensor_list:
                                if x_t.grad is not None:
                                    g_flat = (
                                        x_t.grad.detach()
                                        .float()
                                        .reshape(-1, x_t.grad.shape[-1])
                                    )
                                    g_cov = (g_flat.t() @ g_flat).cpu()
                                    n_g = g_flat.shape[0]
                                    if l_name not in grad_cov:
                                        grad_cov[l_name] = g_cov
                                        grad_counts[l_name] = n_g
                                    else:
                                        grad_cov[l_name] += g_cov
                                        grad_counts[l_name] += n_g
                        self.model.zero_grad(set_to_none=True)

                elif self.tokenizer is not None and self.calibration_texts:
                    inputs = self.tokenizer(
                        self.calibration_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    if self.use_gradients:
                        with torch.enable_grad():
                            inputs_forward = dict(inputs)
                            if (
                                "input_ids" in inputs_forward
                                and "labels" not in inputs_forward
                            ):
                                inputs_forward["labels"] = inputs_forward[
                                    "input_ids"
                                ].clone()
                            out = self.model(**inputs_forward)
                            loss = (
                                active_loss_fn(out)
                                if active_loss_fn is not None
                                else getattr(out, "loss", None)
                            )
                            if loss is not None and loss.requires_grad:
                                loss.backward()
                    else:
                        with torch.no_grad():
                            self.model(**inputs)

                    for l_name, tensor_list in captured_inputs.items():
                        for x_t in tensor_list:
                            if x_t.grad is not None:
                                g_flat = (
                                    x_t.grad.detach()
                                    .float()
                                    .reshape(-1, x_t.grad.shape[-1])
                                )
                                g_cov = (g_flat.t() @ g_flat).cpu()
                                n_g = g_flat.shape[0]
                                if l_name not in grad_cov:
                                    grad_cov[l_name] = g_cov
                                    grad_counts[l_name] = n_g
                                else:
                                    grad_cov[l_name] += g_cov
                                    grad_counts[l_name] += n_g
                    self.model.zero_grad(set_to_none=True)
            except Exception as e:
                logger.debug("FWSVD calibration execution notice: %s", e)
            finally:
                for h in hooks:
                    h.remove()

            # Process collected statistics: gradients outer-product if available, else activation covariance
            for name, module in remaining_layers:
                target_mod = getattr(module, "base_layer", module)
                cls_name = module.__class__.__name__.lower()
                target_cls = target_mod.__class__.__name__.lower()
                is_conv1d = (
                    "conv1d" in cls_name
                    or "conv1d" in target_cls
                    or hasattr(module, "nf")
                    or hasattr(target_mod, "nf")
                )
                weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
                if weight_tensor is None:
                    continue
                in_f = weight_tensor.shape[0] if is_conv1d else weight_tensor.shape[1]

                F_raw: torch.Tensor | None = None
                if name in grad_cov and grad_counts.get(name, 0) > 0:
                    F_raw = grad_cov[name] / float(grad_counts[name])
                elif name in act_cov and act_counts.get(name, 0) > 0:
                    F_raw = act_cov[name] / float(act_counts[name])

                if F_raw is not None and F_raw.shape == (in_f, in_f):
                    F_sym = 0.5 * (F_raw + F_raw.t())
                    tr_F = torch.trace(F_sym).item()
                    damp_val = max(1e-7, float(self.damping * tr_F / max(1, in_f)))
                    F_damped = F_sym + damp_val * torch.eye(
                        in_f, device=F_sym.device, dtype=torch.float32
                    )
                    self.fisher_matrices[name] = F_damped
                    self.fishers[name] = torch.diag(F_damped)

        # Step 3: Compress each target module with preconditioned SVD
        self.model.eval()
        for name, module in target_layers:
            f_mat = self.fisher_matrices.get(name)
            f_diag = self.fishers.get(name)
            svd_layer, err, r = self.compress_linear(
                module, rank=self.rank, fisher_matrix=f_mat, fisher_diag=f_diag
            )
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            if svd_layer is None:
                self.fallback_layers.append(name)
                continue

            if err > self.max_error:
                logger.warning(
                    "FWSVD reconstruction error %.4f on %s exceeded threshold %.4f. Compressing anyway.",
                    err,
                    name,
                    self.max_error,
                )

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
