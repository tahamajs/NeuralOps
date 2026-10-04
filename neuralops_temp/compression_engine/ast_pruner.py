r"""
AST: Adaptive Semi-Structured 2:4 and N:M Pruning with Empirical Fisher
Calibration and PCA Coordinate Rotation (Corrected).

Registered in COMPRESSOR_REGISTRY as 'ast_pruner' and 'ast'.

Mathematical Formulation (CORRECTED):
  1. Activation Covariance & PCA:
       Σ_X = (1/(N-1)) (X-μ)^T (X-μ) = Q Λ Q^T,  Q^T Q = I
  2. Empirical Jacobian-weighted saliency:
       g = ∂L/∂W,  L = 0.5·||logits||²,  F = E[g ⊙ g]
  3. PCA-rotated Fisher projected to original basis:
       F̃ = F · (Q ⊙ Q)             [in rotated basis]
       F_proj = F̃ · Q^T            [projected back to original basis]
  4. Saliency in ORIGINAL basis:
       S_ij = |W_ij| · sqrt(F_proj_ij + ε)
  5. 2:4 structured mask on ORIGINAL weights:
       W_pruned = Π_{2:4}(W, S)

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.pruning_utils import (
    adaptive_max_error,
    apply_24_structured_mask,
    get_calibration_tensors,
    recalibrate_layernorms,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ============================================================================
# ASTPrunedLoRALinear
# ============================================================================


class ASTPrunedLoRALinear(nn.Module):
    """
    Linear layer with frozen AST-pruned base weights and optional trainable
    LoRA adapters.

    The base weight is stored in original coordinate basis with a true 2:4
    (or N:M) sparsity pattern. Framework may pack it as sparse if desired.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compat
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

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
                torch.empty(
                    (self.lora_r, in_features),
                    dtype=torch.float32,
                    device=target_device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (out_features, self.lora_r),
                    dtype=torch.float32,
                    device=target_device,
                )
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
    ) -> ASTPrunedLoRALinear:
        bias = None
        if hasattr(linear, "bias") and linear.bias is not None:
            bias = linear.bias.detach()
        in_features = getattr(linear, "in_features", getattr(linear, "nf", weight.shape[1]))
        out_features = getattr(linear, "out_features", getattr(linear, "nf", weight.shape[0]))
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

    # AST-FIX-12: framework merge() compatibility
    def merge(self) -> None:
        """Merge LoRA into base weight in place."""
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            with torch.no_grad():
                delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
                self.weight.data = (self.weight.data.float() + delta).to(self.weight.dtype)
                self.lora_A = None
                self.lora_B = None
                self.lora_r = 0

    def merge_and_unload(self) -> None:
        self.merge()

    def unmerge(self) -> None:
        # Not supported after merge
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(x_flat, self.weight, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            # AST-FIX: single dtype conversion
            x_f = x_flat.float()
            lora_out = F.linear(F.linear(x_f, self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


# ============================================================================
# ASTPruner
# ============================================================================


@COMPRESSOR_REGISTRY.register("ast_pruner")
@COMPRESSOR_REGISTRY.register("ast")
class ASTPruner(BaseCompressor):
    """
    AST Pruner: vectorized semi-structured 2:4 / N:M pruning with empirical
    Jacobian calibration and PCA coordinate rotation (saliency only).

    Design notes:
      * PCA rotation is used **for saliency estimation only**. The actual
        2:4 mask is applied to the ORIGINAL weights so the sparsity pattern
        is preserved and remains exploitable by sparse backends.
      * The Jacobian proxy (0.5·||logits||²) is a heuristic inspired by
        SparseGPT; it is NOT the true Fisher Information Matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        n_keep: int = 2,
        dim: int = 1,
        use_pca: bool = True,
        use_fisher: bool = True,
        recalibrate_ln: bool = True,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        target_attention_only: bool = False,
        target_modules: list[str] | None = None,
        tokenizer: Any | None = None,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="ast_pruner", **kwargs)
        conf = cfg or kwargs.get("comp_config")
        if conf is not None:
            block_size = _cfg_get(conf, "ast_block_size", _cfg_get(conf, "block_size", block_size))
            n_keep = _cfg_get(conf, "ast_n_keep", _cfg_get(conf, "n_keep", n_keep))
            dim = _cfg_get(conf, "ast_dim", _cfg_get(conf, "dim", dim))
            use_pca = _cfg_get(conf, "ast_use_pca", _cfg_get(conf, "use_pca", use_pca))
            use_fisher = _cfg_get(conf, "ast_use_fisher", _cfg_get(conf, "use_fisher", use_fisher))
            recalibrate_ln = _cfg_get(
                conf,
                "ast_recalibrate_ln",
                _cfg_get(conf, "recalibrate_ln", recalibrate_ln),
            )
            max_error = _cfg_get(conf, "ast_max_error", _cfg_get(conf, "max_error", max_error))
            fallback_to_original = _cfg_get(
                conf,
                "ast_fallback_to_original",
                _cfg_get(conf, "fallback_to_original", fallback_to_original),
            )
            protect_critical = _cfg_get(conf, "protect_critical", protect_critical)
            lora_r = _cfg_get(conf, "lora_r", _cfg_get(conf, "lora_rank", lora_r))
            lora_alpha = _cfg_get(conf, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(conf, "skip_attention_qkv", False)

        self.block_size = max(2, int(block_size))
        self.n_keep = max(1, min(int(n_keep), self.block_size - 1))
        self.dim = dim
        self.use_pca = bool(use_pca)
        self.use_fisher = bool(use_fisher)
        self.recalibrate_ln = bool(recalibrate_ln)
        if max_error is None or max_error == 0.20:
            max_error = adaptive_max_error(
                int(768 * (self.n_keep / self.block_size)), 768, 768, "distilbert"
            )

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = target_attention_only
        self.target_modules = target_modules
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}

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
            if any(
                p in name_lower
                for p in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                ]
            ):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        )
        if not is_linear_like or isinstance(module, ASTPrunedLoRALinear):
            return False

        if self.protect_critical:
            name_lower = name.lower()
            if any(
                skip in name_lower
                for skip in [
                    "classifier",
                    "score",
                    "lm_head",
                    "pooler",
                    "embed",
                    "logits_processor",
                ]
            ):
                return False

        from neuralops.utils.compat import should_compress_linear

        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only and not any(
            k in name.lower()
            for k in [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "o_proj",
                "q_lin",
                "k_lin",
                "v_lin",
                "out_lin",
                "query",
                "key",
                "value",
                "c_attn",
                "c_proj",
            ]
        ):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    # Calibration: collect activations and Jacobian-proxy squared gradients
    # ------------------------------------------------------------------
    def _collect_jacobian_and_activations(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Returns (activation_covariances, jacobian_proxy_squared).
        AST-FIX-06: uses eval() + enable_grad() (not train()).
        AST-FIX-07: always averages over batches.
        AST-FIX-13: passes attention_mask when present.
        """
        activations: dict[str, list[torch.Tensor]] = {}
        fisher_info: dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inp, outp):
                if inp and isinstance(inp[0], torch.Tensor):
                    activations.setdefault(name, []).append(inp[0].detach().cpu())

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Build calibration batches
        batches: list[Any] = []
        if self.calibration_loader is not None:
            for b in self.calibration_loader:
                batches.append(b)
                if len(batches) >= 4:
                    break
        if not batches:
            batches = [
                get_calibration_tensors(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
            ]

        # AST-FIX-06: eval mode (no dropout) + enable_grad for backward
        self.model.eval()
        n_batches_run = 0
        try:
            with torch.enable_grad():
                for batch_input in batches:
                    out = None
                    if isinstance(batch_input, dict):
                        moved = {
                            k: v.to(self.target_device) if hasattr(v, "to") else v
                            for k, v in batch_input.items()
                        }
                        out = self.model(**moved)
                    elif isinstance(batch_input, torch.Tensor):
                        out = self.model(batch_input.to(self.target_device))
                    elif isinstance(batch_input, (list, tuple)):
                        out = self.model(
                            *[
                                b.to(self.target_device) if hasattr(b, "to") else b
                                for b in batch_input
                            ]
                        )

                    if self.use_fisher and out is not None:
                        # AST-FIX-14: explicit naming - this is NOT true Fisher
                        # 0.5·||logits||² is a Jacobian-weighted norm proxy.
                        logits = None
                        if hasattr(out, "logits"):
                            logits = out.logits
                        elif isinstance(out, torch.Tensor):
                            logits = out
                        elif (
                            isinstance(out, (tuple, list))
                            and out
                            and isinstance(out[0], torch.Tensor)
                        ):
                            logits = out[0]

                        if logits is not None and logits.requires_grad:
                            loss = 0.5 * torch.sum(logits.pow(2))
                            self.model.zero_grad(set_to_none=True)
                            loss.backward()
                            for name, module in self.model.named_modules():
                                if not self._is_target_module(name, module):
                                    continue
                                # Resolve weight
                                target_w = getattr(module, "weight", None)
                                base = getattr(module, "base_layer", module)
                                if target_w is None:
                                    target_w = getattr(base, "weight", None)
                                if target_w is None or target_w.grad is None:
                                    continue
                                g = target_w.grad.detach()
                                g_sq = g.pow(2).cpu()
                                if name in fisher_info:
                                    fisher_info[name] = fisher_info[name] + g_sq
                                else:
                                    fisher_info[name] = g_sq
                            self.model.zero_grad(set_to_none=True)
                            n_batches_run += 1
        except Exception as e:
            logger.warning("AST Jacobian calibration failed: %s", e)
        finally:
            for h in hooks:
                h.remove()

        # AST-FIX-07: always divide (even for n_batches_run == 1)
        if n_batches_run >= 1:
            inv = 1.0 / float(n_batches_run)
            for k in list(fisher_info.keys()):
                fisher_info[k] = fisher_info[k] * inv
        else:
            logger.warning(
                "AST: no successful backward passes; saliency will fall back to |W| only."
            )

        # Build covariance map, then discard raw activations (AST-FIX-11)
        cov_map: dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if not act_list:
                continue
            act_concat = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0).float()
            N, D = act_concat.shape
            if N < 2:
                continue
            mean = torch.mean(act_concat, dim=0, keepdim=True)
            centered = act_concat - mean
            cov = torch.matmul(centered.T, centered) / float(N - 1)
            # AST-FIX-09: explicit symmetrize
            cov = (cov + cov.T) * 0.5
            cov_map[name] = cov
            # Free memory
            del act_concat, centered, mean
            if name not in fisher_info:
                fisher_info[name] = torch.var(torch.zeros(1, D), dim=0, keepdim=True).expand(1, D)
        activations.clear()

        return cov_map, fisher_info

    # ------------------------------------------------------------------
    # PCA basis extraction (per-layer, no covariance retained)
    # ------------------------------------------------------------------
    @staticmethod
    def _pca_basis(cov: torch.Tensor, d_in: int, device: torch.device) -> torch.Tensor:
        """
        AST-FIX-08: scaled eigenvalue regularization.
        AST-FIX-09: symmetrize before eigh.
        Returns orthonormal Q (d_in, d_in) in original column basis.
        """
        cov = cov.to(device).float()
        if cov.shape != (d_in, d_in):
            return torch.eye(d_in, device=device)

        # Eigenvalue regularization proportional to mean variance
        diag_mean = float(cov.diagonal().mean().item())
        eps = max(1e-8, 1e-6 * diag_mean)
        cov = cov + eps * torch.eye(d_in, device=device)

        try:
            eigvals, Q = torch.linalg.eigh(cov)
            idx = torch.argsort(eigvals, descending=True)
            Q = Q[:, idx]
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                logger.warning("AST: PCA basis contained NaN/Inf; using identity.")
                return torch.eye(d_in, device=device)
            return Q.contiguous()
        except Exception as e:
            logger.warning("AST: eigh failed (%s); using identity.", e)
            return torch.eye(d_in, device=device)

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        from neuralops.core.model_utils import replace_module

        cov_map, fisher_map = self._collect_jacobian_and_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set = set()

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue

            w_id = id(module.weight)
            if w_id in seen_weights:
                for p in module.parameters():
                    p.requires_grad = False
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            target_mod = getattr(module, "base_layer", module)
            raw_w = target_mod.weight.data
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

            # W: (d_out, d_in)
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            device = W.device
            dtype = target_mod.weight.dtype

            # ------------------------------------------------------------------
            # 1. Compute saliency in ORIGINAL basis (AST-FIX-01)
            # ------------------------------------------------------------------
            # Saliency is computed as:
            #   1a. Rotate Fisher into PCA basis: F_rot = F @ (Q ⊙ Q)
            #   1b. Apply 2:4-aware importance per (row, column) in the rotated
            #       basis, then project the rotation-weighted per-column score
            #       back to the original column basis using Q.
            #
            # In practice, we use the following simplified and provably
            # structure-preserving form:
            #   S_ij = |W_ij| · sqrt(F_proj_ij + ε)
            # where F_proj = F @ (Q ⊙ Q)  (a per-element projection back).
            # This yields a dense saliency map in the ORIGINAL basis, but the
            # actual mask is applied to W (not W_rot), preserving 2:4 sparsity.
            # ------------------------------------------------------------------
            fisher = fisher_map.get(name, None)
            if fisher is not None and self.use_fisher:
                F_mat = fisher.to(device).float()
                # AST-FIX-10: NaN/Inf guard
                F_mat = torch.nan_to_num(F_mat, nan=0.0, posinf=0.0, neginf=0.0)

                if is_conv1d and F_mat.dim() == 2 and F_mat.shape == raw_w.shape:
                    F_mat = F_mat.t().contiguous()

                if F_mat.dim() == 2 and F_mat.shape == W.shape:
                    if self.use_pca and name in cov_map:
                        Q = self._pca_basis(cov_map[name], d_in, device)
                        # Project Fisher into rotated basis, then back
                        F_rot = F_mat @ (Q * Q)  # (d_out, d_in) × (d_in, d_in)
                        F_proj = F_rot @ Q.T  # (d_out, d_in) × (d_in, d_in) → (d_out, d_in)
                    else:
                        F_proj = F_mat
                elif F_mat.shape[-1] == d_in and F_mat.dim() == 1:
                    F_proj = F_mat.reshape(1, d_in).expand(d_out, d_in)
                else:
                    F_proj = torch.ones_like(W)
                F_proj = torch.clamp(F_proj, min=1e-8)
                saliency = torch.abs(W) * torch.sqrt(F_proj)
            else:
                saliency = torch.abs(W)

            saliency = torch.nan_to_num(saliency, nan=0.0, posinf=1e4, neginf=0.0)

            # ------------------------------------------------------------------
            # 2. Apply 2:4 structured mask on ORIGINAL weights
            # ------------------------------------------------------------------
            W_pruned, _mask = apply_24_structured_mask(W, saliency=saliency, dim=self.dim)
            W_final = torch.nan_to_num(W_pruned, nan=0.0, posinf=0.0, neginf=0.0)

            # ------------------------------------------------------------------
            # 3. Error check
            # ------------------------------------------------------------------
            w_norm = torch.linalg.norm(W).item()
            err = float(torch.linalg.norm(W - W_final).item() / max(w_norm, 1e-12))
            self.layer_errors[name] = err

            if err > self.max_error:
                logger.warning(
                    "AST: reconstruction error %.4f on %s exceeds %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("AST: falling back to original for %s.", name)
                    # AST-FIX-02: only freeze the base weight, not LoRA
                    for pn, p in module.named_parameters(recurse=False):
                        if "lora_" in pn.lower():
                            continue  # keep LoRA trainable
                        p.requires_grad = False
                    self.fallback_layers.append(name)
                    continue

            # ------------------------------------------------------------------
            # 4. Replace / in-place update
            # ------------------------------------------------------------------
            if self.lora_r > 0:
                new_layer = ASTPrunedLoRALinear.from_linear(
                    target_mod,
                    weight=W_final.contiguous(),
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )

                # AST-FIX-04: type-safe LoRA transfer
                if hasattr(module, "base_layer"):
                    if hasattr(module, "lora_A") and module.lora_A is not None:
                        src_a = getattr(module.lora_A, "weight", module.lora_A)
                        if (
                            isinstance(src_a, torch.Tensor)
                            and src_a.shape == new_layer.lora_A.shape
                        ):
                            new_layer.lora_A.data.copy_(src_a.to(new_layer.lora_A.dtype))
                    if hasattr(module, "lora_B") and module.lora_B is not None:
                        src_b = getattr(module.lora_B, "weight", module.lora_B)
                        if (
                            isinstance(src_b, torch.Tensor)
                            and src_b.shape == new_layer.lora_B.shape
                        ):
                            new_layer.lora_B.data.copy_(src_b.to(new_layer.lora_B.dtype))

                    # Swap the base layer in place
                    if hasattr(module, "base_layer"):
                        module.base_layer = new_layer
                    # AST-FIX-03: removed dead `if name == "":` branch
                    # Don't call replace_module for the wrapped module
                    # because we already replaced its inner base_layer
                else:
                    if name != "":
                        replace_module(self.model, name, new_layer)
                    else:
                        self.model = new_layer
            else:
                # AST-FIX-05: no_grad around mutation
                with torch.no_grad():
                    new_w = W_final.to(dtype=dtype, device=device)
                    if is_conv1d:
                        new_w = new_w.t().contiguous()
                    target_mod.weight.data.copy_(new_w)

            self.compressed_layers.append(name)

        # ------------------------------------------------------------------
        # LayerNorm recalibration
        # ------------------------------------------------------------------
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
                logger.warning("AST: LayerNorm recalibration skipped: %s", e)

        # ------------------------------------------------------------------
        # Forward sanity check (AST-FIX-13: passes attention_mask)
        # ------------------------------------------------------------------
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                mask = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=mask)
                except TypeError:
                    self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.warning("AST: post-compression forward check failed: %s", e)

        # AST-FIX-15: log skipped layer count
        if self.skipped_layers:
            logger.info(
                "AST: skipped %d layers (dup weights / target rules).",
                len(self.skipped_layers),
            )

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update(
            {
                "pruned_layers": len(self.compressed_layers),
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "block_size": self.block_size,
                "n_keep": self.n_keep,
                "pattern": f"{self.n_keep}:{self.block_size}",
                "use_pca": self.use_pca,
                "use_fisher": self.use_fisher,
                "recalibrate_ln": self.recalibrate_ln,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
                "method": "ast_jacobian_pca_2_4_pruning",
            }
        )
        self.stats = stats
        logger.info(
            "Compressor ast finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ============================================================================
# Factory
# ============================================================================


@COMPRESSOR_REGISTRY.register("ast_pruner_factory")
@COMPRESSOR_REGISTRY.register("ast_factory")
def ast_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    block_size = kwargs.pop(
        "block_size",
        _cfg_get(comp_config, "ast_block_size", _cfg_get(comp_config, "block_size", 4)),
    )
    n_keep = kwargs.pop(
        "n_keep",
        _cfg_get(comp_config, "ast_n_keep", _cfg_get(comp_config, "n_keep", 2)),
    )
    dim = kwargs.pop("dim", _cfg_get(comp_config, "ast_dim", _cfg_get(comp_config, "dim", 1)))
    use_pca = kwargs.pop("use_pca", _cfg_get(comp_config, "ast_use_pca", True))
    use_fisher = kwargs.pop("use_fisher", _cfg_get(comp_config, "ast_use_fisher", True))
    recalibrate_ln = kwargs.pop("recalibrate_ln", _cfg_get(comp_config, "ast_recalibrate_ln", True))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "ast_max_error", _cfg_get(comp_config, "max_error", 0.20)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "ast_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    lora_r = kwargs.pop(
        "lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 0))
    )
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", True)
    )

    compressor = ASTPruner(
        model,
        block_size=block_size,
        n_keep=n_keep,
        dim=dim,
        use_pca=use_pca,
        use_fisher=use_fisher,
        recalibrate_ln=recalibrate_ln,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "ASTPruner",
    "ASTPrunedLoRALinear",
    "ast_pruner_factory",
]
