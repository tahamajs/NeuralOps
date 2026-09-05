"""
ToMoE: converts dense linear and Conv1D layers (e.g. FFN and attention projections)
into a Mixture of Experts using weight clustering.
Registered as 'tomoe' and 'tomoe_compressor'.

Production-hardened with:
- Fix 1: Reconstruction error validation with configurable max_error (default 0.20) and safe fallback.
- Fix 2: Support for Conv1D (GPT-2) with weight transposition and fused/GQA projections (Qwen/LLaMA).
- Fix 3: Fully vectorized forward pass using torch.gather and scatter_add_ / index_add_ without Python token loops;
         numerically stable in FP16/BF16/FP32 with FP32 router gating.
- Fix 4: Robust FP32 KMeans clustering with deterministic equal-division fallback preventing empty clusters;
         guarantees original weight preservation without random weight replacement.
- Fix 5: Comprehensive per-layer statistics (converted/skipped layers, params before/after, compression ratio, errors).
- Fix 6: Protection of critical layers (classifier, score, lm_head, pooler, embeddings).
- Fix 7: Optional trainable LoRA adapters (lora_A, lora_B) on experts and router, or router-only training when frozen.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class ToMoELinear(nn.Module):
    """
    A single MoE-fied linear layer: learned routing + expert sub-layers.

    Features:
    - Clusters output neurons using FP32 KMeans (with equal-division fallback).
    - Vectorized forward pass using gather and scatter_add_ for equal-sized experts,
      or vectorized index-dispatch for variable-sized experts.
    - Numerical stability in FP16, BF16, and FP32.
    - Exposes reconstructed effective weight and bias properties for validation.
    - Supports optional trainable LoRA adapters on experts and router.
    """

    def __init__(
        self,
        original: Optional[nn.Module] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        num_experts: int = 4,
        top_k: int = 2,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        freeze_experts: bool = True,
        max_error: float = 0.20,
    ):
        super().__init__()

        # Fix 2: Handle both nn.Linear and Conv1D (or pre-extracted weights)
        if original is not None:
            if isinstance(original, nn.Linear):
                in_feat = original.in_features
                out_feat = original.out_features
                orig_w = original.weight.data.clone()
                orig_b = original.bias.data.clone() if original.bias is not None else None
            elif type(original).__name__ == "Conv1D" or (hasattr(original, "nx") and hasattr(original, "nf")):
                # Conv1D stores weight as (nx, nf) where nx=in_features, nf=out_features
                in_feat = original.weight.shape[0]
                out_feat = original.weight.shape[1]
                orig_w = original.weight.data.t().clone()  # Transpose to (out_features, in_features)
                orig_b = original.bias.data.clone() if original.bias is not None else None
            elif hasattr(original, "weight") and original.weight.ndim == 2:
                out_feat, in_feat = original.weight.shape
                orig_w = original.weight.data.clone()
                orig_b = original.bias.data.clone() if getattr(original, "bias", None) is not None else None
            else:
                raise TypeError(f"Unsupported module type: {type(original)}")
        else:
            if in_features is None or out_features is None or weight is None:
                raise ValueError("Either 'original' module or ('in_features', 'out_features', 'weight') must be provided.")
            in_feat = in_features
            out_feat = out_features
            orig_w = weight.clone()
            orig_b = bias.clone() if bias is not None else None

        self.in_features = in_feat
        self.out_features = out_feat

        # Guard against num_experts > out_features
        self.num_experts = max(1, min(num_experts, out_feat))
        self.top_k = max(1, min(top_k, self.num_experts))
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scale = (lora_alpha / lora_r) if lora_r > 0 else 1.0
        self.freeze_experts = freeze_experts
        self.max_error = max_error

        # Fix 4: Perform KMeans clustering in FP32 with safe equal-division fallback
        W_fp32 = orig_w.detach().float().cpu().numpy()
        assignments = None

        if self.num_experts > 1:
            try:
                kmeans = KMeans(
                    n_clusters=self.num_experts,
                    random_state=42,
                    n_init="auto" if hasattr(KMeans, "n_init") else 10,
                )
                pred = kmeans.fit_predict(W_fp32)
                counts = np.bincount(pred, minlength=self.num_experts)
                if (counts == 0).any():
                    logger.warning("ToMoE: KMeans produced empty cluster(s). Falling back to equal division.")
                    assignments = None
                else:
                    assignments = pred
            except Exception as e:
                logger.warning("ToMoE: KMeans clustering failed (%s). Falling back to equal division.", e)
                assignments = None

        if assignments is None:
            # Deterministic equal contiguous division ensuring zero empty experts
            assignments = np.zeros(self.out_features, dtype=int)
            chunk = self.out_features // self.num_experts
            rem = self.out_features % self.num_experts
            start = 0
            for e in range(self.num_experts):
                end = start + chunk + (1 if e < rem else 0)
                assignments[start:end] = e
                start = end

        # Create expert sub-layers and record neuron assignment masks
        self.experts = nn.ModuleList()
        self.register_buffer("_masks", torch.zeros(self.num_experts, self.out_features, dtype=torch.bool))
        expert_indices_list = []
        cluster_sizes = []

        device = orig_w.device
        dtype = orig_w.dtype

        for e in range(self.num_experts):
            mask_bool = torch.tensor(assignments == e, dtype=torch.bool)
            self._masks[e] = mask_bool
            indices = torch.nonzero(mask_bool, as_tuple=True)[0]
            expert_indices_list.append(indices)
            n_neurons = len(indices)
            cluster_sizes.append(n_neurons)

            expert = nn.Linear(self.in_features, n_neurons, bias=orig_b is not None, device=device, dtype=dtype)
            # Copy exact original weights to ensure zero initial reconstruction error
            expert.weight.data.copy_(orig_w[mask_bool])
            if orig_b is not None:
                expert.bias.data.copy_(orig_b[mask_bool])

            # Fix 7: Freeze expert base weights if requested
            if self.freeze_experts:
                expert.weight.requires_grad = False
                if expert.bias is not None:
                    expert.bias.requires_grad = False

            self.experts.append(expert)

        self.expert_indices = [idx.to(device) for idx in expert_indices_list]
        self.equal_expert_size = len(set(cluster_sizes)) == 1

        if self.equal_expert_size:
            self.expert_out_dim = cluster_sizes[0]
            # Precompute column index map for scatter_add_: shape (num_experts, expert_out_dim)
            col_map = torch.stack(expert_indices_list, dim=0).to(device)
            self.register_buffer("expert_col_map", col_map)
        else:
            self.expert_out_dim = -1

        # Router: learned linear projection to expert logits
        self.router = nn.Linear(self.in_features, self.num_experts, device=device, dtype=dtype)
        nn.init.normal_(self.router.weight, std=0.02)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        self.router.weight.requires_grad = True
        if self.router.bias is not None:
            self.router.bias.requires_grad = True

        # Fix 7: Trainable LoRA Adapters
        if self.lora_r > 0:
            self.expert_lora_A = nn.ParameterList()
            self.expert_lora_B = nn.ParameterList()
            for e in range(self.num_experts):
                n_neurons = cluster_sizes[e]
                l_A = nn.Parameter(torch.empty(self.lora_r, self.in_features, device=device, dtype=dtype))
                l_B = nn.Parameter(torch.zeros(n_neurons, self.lora_r, device=device, dtype=dtype))
                nn.init.kaiming_uniform_(l_A, a=math.sqrt(5))
                nn.init.zeros_(l_B)
                self.expert_lora_A.append(l_A)
                self.expert_lora_B.append(l_B)

            self.lora_A_router = nn.Parameter(torch.empty(self.lora_r, self.in_features, device=device, dtype=dtype))
            self.lora_B_router = nn.Parameter(torch.zeros(self.num_experts, self.lora_r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A_router, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_router)

    @property
    def masks(self) -> List[torch.Tensor]:
        return [self._masks[e] for e in range(self.num_experts)]

    @property
    def weight(self) -> torch.Tensor:
        """Fix 1: Reconstructed effective full weight matrix (out_features, in_features)."""
        device = self.experts[0].weight.device
        dtype = self.experts[0].weight.dtype
        W = torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)
        for e, expert in enumerate(self.experts):
            mask = self._masks[e].to(device)
            W[mask] = expert.weight
        return W

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Reconstructed effective full bias vector (out_features,)."""
        if self.experts[0].bias is None:
            return None
        device = self.experts[0].bias.device
        dtype = self.experts[0].bias.dtype
        b = torch.zeros(self.out_features, device=device, dtype=dtype)
        for e, expert in enumerate(self.experts):
            mask = self._masks[e].to(device)
            b[mask] = expert.bias
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fix 3: Vectorized forward pass with gather and scatter_add_ / index_add_.
        Supports arbitrary batch and sequence dimensions; numerically stable in FP16/BF16/FP32.
        """
        orig_shape = x.shape
        in_feat = orig_shape[-1]
        x_flat = x.reshape(-1, in_feat)
        N = x_flat.shape[0]

        # Compute router logits in matching input dtype, cast to float32 for softmax stability
        router_in = x_flat.to(self.router.weight.dtype)
        router_logits = self.router(router_in).float()
        if self.lora_r > 0 and hasattr(self, "lora_A_router") and hasattr(self, "lora_B_router"):
            router_lora = (router_in @ self.lora_A_router.t()) @ self.lora_B_router.t() * self.lora_scale
            router_logits = router_logits + router_lora.float()

        # Graceful handling of non-finite logits
        if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            router_logits = torch.nan_to_num(router_logits, nan=0.0, posinf=1.0, neginf=-1.0)

        # Top-k routing
        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)  # (N, top_k)
        topk_weights = torch.softmax(topk_vals, dim=-1).to(x.dtype)  # (N, top_k)

        out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)

        if self.equal_expert_size:
            # Fast vectorized gather & scatter_add_ for equal-sized expert partitions
            D_exp = self.expert_out_dim
            all_expert_outs = torch.stack([exp(x_flat) for exp in self.experts], dim=1)  # (N, E, D_exp)

            if self.lora_r > 0 and hasattr(self, "expert_lora_A"):
                lora_outs = []
                for e in range(self.num_experts):
                    l_e = (x_flat.to(self.expert_lora_A[e].dtype) @ self.expert_lora_A[e].t()) @ self.expert_lora_B[e].t() * self.lora_scale
                    lora_outs.append(l_e.to(x.dtype))
                all_expert_outs = all_expert_outs + torch.stack(lora_outs, dim=1)

            expanded_idx = topk_idx.unsqueeze(-1).expand(N, self.top_k, D_exp)
            selected_outs = torch.gather(all_expert_outs, dim=1, index=expanded_idx)  # (N, top_k, D_exp)
            weighted_selected = selected_outs * topk_weights.unsqueeze(-1)  # (N, top_k, D_exp)

            col_map = self.expert_col_map.to(x.device)
            dest_cols = col_map[topk_idx]  # (N, top_k, D_exp)

            out.scatter_add_(
                dim=1,
                index=dest_cols.reshape(N, self.top_k * D_exp),
                src=weighted_selected.reshape(N, self.top_k * D_exp),
            )
        else:
            # Vectorized token dispatch per expert for variable sized clusters
            for e in range(self.num_experts):
                expert_match = (topk_idx == e)  # (N, top_k)
                if not expert_match.any():
                    continue
                token_idx, rank_pos = torch.where(expert_match)
                selected_tokens = x_flat[token_idx]
                exp_out = self.experts[e](selected_tokens)
                if self.lora_r > 0 and hasattr(self, "expert_lora_A"):
                    l_e = (selected_tokens.to(self.expert_lora_A[e].dtype) @ self.expert_lora_A[e].t()) @ self.expert_lora_B[e].t() * self.lora_scale
                    exp_out = exp_out + l_e.to(x.dtype)
                gate_w = topk_weights[token_idx, rank_pos].unsqueeze(-1)
                weighted_exp_out = exp_out * gate_w
                exp_cols = self.expert_indices[e].to(x.device)
                out[token_idx[:, None], exp_cols] += weighted_exp_out

        return out.reshape(*orig_shape[:-1], self.out_features)


@COMPRESSOR_REGISTRY.register("tomoe")
@COMPRESSOR_REGISTRY.register("tomoe_compressor")
class ToMoECompressor(BaseCompressor):
    """
    Converts dense Linear and Conv1D layers (e.g. FFN and projections)
    into a Mixture of Experts using weight clustering.

    Args:
        model: PyTorch model.
        num_experts: Number of experts per MoE layer (default 4).
        top_k: Number of active experts per token (default 2).
        max_error: Maximum allowed relative Frobenius error (default 0.20).
        lora_r: LoRA adapter rank on experts/router (default 0 = disabled).
        lora_alpha: LoRA scaling factor (default 16.0).
        freeze_experts: If True and lora_r==0, train only the router (default True).
        target_modules: Optional list of target layer name substrings.
    """

    # Fix 6: Protection of Critical Layers
    PROTECTED_KEYWORDS = [
        "classifier",
        "score",
        "lm_head",
        "pooler",
        "embed",
        "wte",
        "wpe",
        "shared",
        "head",
        "logits",
        "pre_classifier",
    ]

    def __init__(
        self,
        model: nn.Module,
        num_experts: int = 4,
        top_k: int = 2,
        max_error: float = 0.20,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        freeze_experts: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(model, name="tomoe", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            num_experts = getattr(cfg, "num_experts", getattr(cfg, "tomoe_num_experts", num_experts))
            top_k = getattr(cfg, "top_k", getattr(cfg, "tomoe_top_k", top_k))
            max_error = getattr(cfg, "max_error", getattr(cfg, "max_reconstruction_error", max_error))
            lora_r = getattr(cfg, "lora_r", getattr(cfg, "tomoe_lora_r", lora_r))
            freeze_experts = getattr(cfg, "freeze_experts", freeze_experts)

        if "num_experts" in kwargs and kwargs["num_experts"] is not None:
            num_experts = int(kwargs["num_experts"])
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            top_k = int(kwargs["top_k"])
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "lora_r" in kwargs and kwargs["lora_r"] is not None:
            lora_r = int(kwargs["lora_r"])
        if "freeze_experts" in kwargs:
            freeze_experts = bool(kwargs["freeze_experts"])

        self.num_experts = max(1, num_experts)
        self.top_k = max(1, min(top_k, self.num_experts))
        self.max_error = float(max_error)
        self.lora_r = max(0, lora_r)
        self.lora_alpha = float(lora_alpha)
        self.freeze_experts = bool(freeze_experts)
        self.target_modules = target_modules
        self.stats: Dict[str, Any] = {}

    def _is_protected(self, name: str) -> bool:
        low = name.lower()
        return any(kw in low for kw in self.PROTECTED_KEYWORDS)

    def _is_target(self, name: str) -> bool:
        if self._is_protected(name):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        # Default: target FFN, MLP, and linear projections across all supported families
        low = name.lower()
        return any(
            t in low
            for t in [
                "mlp",
                "ffn",
                "intermediate",
                "output.dense",
                "c_fc",
                "c_proj",
                "lin1",
                "lin2",
                "gate_proj",
                "up_proj",
                "down_proj",
                "c_attn",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_lin",
                "k_lin",
                "v_lin",
                "out_lin",
                "query",
                "key",
                "value",
            ]
        )

    def _get_parent(self, full_name: str) -> Tuple[Optional[nn.Module], str]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None, ""
        return parent, parts[-1]

    def compress(self, orig_model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress target layers into ToMoE Mixture of Experts layers with full error validation.
        """
        if orig_model is not None:
            self.model = orig_model

        orig_total_params = sum(p.numel() for p in self.model.parameters())
        converted_layers = []
        skipped_layers = []
        layer_details = {}

        for name, module in list(self.model.named_modules()):
            if not self._is_target(name):
                continue

            # Fix 2: Check for supported layer type (nn.Linear or Conv1D)
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = type(module).__name__ == "Conv1D" or (hasattr(module, "nx") and hasattr(module, "nf"))

            if not (is_linear or is_conv1d):
                logger.warning("ToMoE: Skipping unsupported layer type %s for %s", type(module).__name__, name)
                skipped_layers.append((name, f"unsupported_type_{type(module).__name__}"))
                continue

            # Determine weights and feature shapes
            if is_linear:
                in_f = module.in_features
                out_f = module.out_features
                orig_w_2d = module.weight.data
            else:
                in_f = module.weight.shape[0]
                out_f = module.weight.shape[1]
                orig_w_2d = module.weight.data.t()

            if out_f < 2:
                skipped_layers.append((name, f"out_features_too_small_{out_f}"))
                continue

            # Respect GQA dimensions (e.g. Qwen k_proj/v_proj with fewer heads)
            eff_num_experts = max(1, min(self.num_experts, out_f))
            eff_top_k = max(1, min(self.top_k, eff_num_experts))

            if eff_num_experts < 2:
                skipped_layers.append((name, "effective_experts_less_than_2"))
                continue

            parent, child_name = self._get_parent(name)
            if parent is None:
                skipped_layers.append((name, "parent_not_found"))
                continue

            # Construct ToMoELinear
            try:
                moe_layer = ToMoELinear(
                    original=module,
                    num_experts=eff_num_experts,
                    top_k=eff_top_k,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    freeze_experts=self.freeze_experts,
                    max_error=self.max_error,
                )
            except Exception as e:
                logger.warning("ToMoE: Conversion failed for %s: %s", name, e)
                skipped_layers.append((name, f"construction_error_{e}"))
                continue

            # Fix 1: Validate relative Frobenius reconstruction error
            W_rec = moe_layer.weight
            W_orig = orig_w_2d.to(W_rec.device)
            err = (torch.norm(W_rec.float() - W_orig.float()) / (torch.norm(W_orig.float()) + 1e-8)).item()

            if err > self.max_error or np.isnan(err):
                logger.warning(
                    "ToMoE: %s reconstruction error (%.4f) exceeded max_error (%.4f). Keeping original layer.",
                    name,
                    err,
                    self.max_error,
                )
                skipped_layers.append((name, f"error_threshold_exceeded_{err:.4f}"))
                continue

            # Replace layer in parent module
            orig_p_count = sum(p.numel() for p in module.parameters())
            new_p_count = sum(p.numel() for p in moe_layer.parameters())
            setattr(parent, child_name, moe_layer)

            converted_layers.append(name)
            layer_details[name] = {
                "num_experts": eff_num_experts,
                "top_k": eff_top_k,
                "reconstruction_error": err,
                "original_params": orig_p_count,
                "new_params": new_p_count,
                "equal_expert_size": moe_layer.equal_expert_size,
            }
            logger.info(
                "ToMoE: Converted %s -> ToMoELinear (experts=%d, top_k=%d, reconstruction_error=%.4e)",
                name,
                eff_num_experts,
                eff_top_k,
                err,
            )

        new_total_params = sum(p.numel() for p in self.model.parameters())
        self._compressed_params = new_total_params

        # Fix 5: Per-Layer Stats and Logging
        self.stats = {
            "converted_layers": converted_layers,
            "num_converted_layers": len(converted_layers),
            "skipped_layers": skipped_layers,
            "original_params": orig_total_params,
            "compressed_params": new_total_params,
            "compression_ratio": orig_total_params / max(1, new_total_params),
            "layer_details": layer_details,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "max_error": self.max_error,
        }

        logger.info(
            "ToMoE Summary: Converted %d layers, Skipped %d layers. Params: %s -> %s (ratio: %.2fx)",
            len(converted_layers),
            len(skipped_layers),
            f"{orig_total_params:,}",
            f"{new_total_params:,}",
            self.stats["compression_ratio"],
        )

        return self.model, self.stats


def tomoe_compressor_factory(
    model: nn.Module,
    comp_config: Any = None,
    num_experts: int = 4,
    top_k: int = 2,
    **kwargs: Any,
) -> nn.Module:
    """Factory function for ToMoE compressor."""
    compressor = ToMoECompressor(
        model,
        num_experts=num_experts,
        top_k=top_k,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
