"""
ToMoE: converts dense linear and Conv1D layers (e.g. FFN and attention projections)
into a Mixture of Experts using weight clustering.
Registered as 'tomoe' and 'tomoe_compressor'.

"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear
from sklearn.cluster import KMeans, MiniBatchKMeans

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
        original: nn.Module | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        freeze_experts: bool = True,
        max_error: float = 0.20,
    ):
        super().__init__()

        # Fix 2: Handle both nn.Linear, Conv1D, and LoRALinear wrappers
        if original is not None:
            base_layer = getattr(original, "base_layer", original)
            cls_name = base_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or (
                hasattr(base_layer, "nx") and hasattr(base_layer, "nf")
            )
            if is_conv1d:
                # Conv1D stores weight as (nx, nf) where nx=in_features, nf=out_features
                in_feat = base_layer.weight.shape[0]
                out_feat = base_layer.weight.shape[1]
                orig_w = (
                    base_layer.weight.data.t().contiguous().clone()
                )  # Transpose to (out_features, in_features)
                orig_b = (
                    base_layer.bias.data.clone()
                    if base_layer.bias is not None
                    else None
                )
            elif isinstance(base_layer, nn.Linear):
                in_feat = base_layer.in_features
                out_feat = base_layer.out_features
                orig_w = base_layer.weight.data.contiguous().clone()
                orig_b = (
                    base_layer.bias.data.clone()
                    if base_layer.bias is not None
                    else None
                )
            elif hasattr(base_layer, "weight") and base_layer.weight.ndim == 2:
                in_feat = getattr(base_layer, "in_features", base_layer.weight.shape[1])
                out_feat = getattr(
                    base_layer, "out_features", base_layer.weight.shape[0]
                )
                orig_w = base_layer.weight.data.contiguous().clone()
                orig_b = (
                    base_layer.bias.data.clone()
                    if getattr(base_layer, "bias", None) is not None
                    else None
                )
            else:
                raise TypeError(f"Unsupported module type: {type(base_layer)}")
        else:
            base_layer = None
            if in_features is None or out_features is None or weight is None:
                raise ValueError(
                    "Either 'original' module or ('in_features', 'out_features', 'weight') must be provided."
                )
            in_feat = in_features
            out_feat = out_features
            orig_w = weight.clone()
            orig_b = bias.clone() if bias is not None else None

        self.in_features = in_feat
        self.out_features = out_feat

        # Guard against num_experts > out_features
        self.num_experts = max(1, min(num_experts, out_feat))
        self.top_k = max(1, min(top_k, self.num_experts))

        # Check for existing LoRA in original
        has_orig_lora = (
            original is not None
            and hasattr(original, "lora_A")
            and hasattr(original, "lora_B")
        )
        if has_orig_lora and getattr(original, "r", 0) > 0:
            lora_r = getattr(original, "r", lora_r)

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scale = (lora_alpha / lora_r) if lora_r > 0 else 1.0
        self.freeze_experts = freeze_experts
        self.max_error = max_error

        # Perform clustering with MiniBatchKMeans for large layers, standard KMeans otherwise
        W_fp32 = orig_w.detach().float().cpu().numpy()
        assignments = None

        if self.num_experts > 1:
            try:
                sig = inspect.signature(KMeans.__init__).parameters
                n_init_val = "auto" if "n_init" in sig else 10
                if self.out_features > 2048:
                    km = MiniBatchKMeans(
                        n_clusters=self.num_experts,
                        random_state=42,
                        batch_size=min(1024, self.out_features),
                        n_init=n_init_val,
                    )
                else:
                    km = KMeans(
                        n_clusters=self.num_experts,
                        random_state=42,
                        n_init=n_init_val,
                    )
                pred = km.fit_predict(W_fp32)
                counts = np.bincount(pred, minlength=self.num_experts)
                if (counts == 0).any():
                    logger.warning(
                        "ToMoE: KMeans produced empty cluster(s). Falling back to equal division."
                    )
                    assignments = None
                else:
                    # Balance KMeans clusters to equal size for fast vectorized path
                    target_size = self.out_features // self.num_experts
                    rem = self.out_features % self.num_experts
                    if rem == 0 and not (counts == target_size).all():
                        # Distances to cluster centers for balanced reassignment
                        dists = km.transform(W_fp32)  # [out_features, num_experts]
                        # Sort neurons by distance margin between assigned cluster and others
                        order = np.argsort(np.min(dists, axis=1))
                        balanced_assign = np.full(self.out_features, -1, dtype=int)
                        exp_occupancy = np.zeros(self.num_experts, dtype=int)
                        for idx in order:
                            pref_order = np.argsort(dists[idx])
                            for p in pref_order:
                                if exp_occupancy[p] < target_size:
                                    balanced_assign[idx] = p
                                    exp_occupancy[p] += 1
                                    break
                        assignments = balanced_assign
                    else:
                        assignments = pred
            except Exception as e:
                logger.warning(
                    "ToMoE: KMeans clustering failed (%s). Falling back to equal division.",
                    e,
                )
                assignments = None

        if assignments is None or (assignments < 0).any():
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
        self.register_buffer(
            "_masks", torch.zeros(self.num_experts, self.out_features, dtype=torch.bool)
        )
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

            expert = nn.Linear(
                self.in_features,
                n_neurons,
                bias=orig_b is not None,
                device=device,
                dtype=dtype,
            )
            expert.weight.data.copy_(orig_w[mask_bool])
            if orig_b is not None:
                expert.bias.data.copy_(orig_b[mask_bool])

            if self.freeze_experts:
                expert.weight.requires_grad = False
                if expert.bias is not None:
                    expert.bias.requires_grad = False

            self.experts.append(expert)

        # Register padded expert indices as a proper buffer so it moves with .to(device)
        max_size = max(len(idx) for idx in expert_indices_list)
        padded_indices = torch.zeros(self.num_experts, max_size, dtype=torch.long)
        for e, idx in enumerate(expert_indices_list):
            padded_indices[e, : len(idx)] = idx
        self.register_buffer("expert_indices_padded", padded_indices)
        self.register_buffer(
            "expert_lens", torch.tensor(cluster_sizes, dtype=torch.long)
        )

        self.equal_expert_size = len(set(cluster_sizes)) == 1

        if self.equal_expert_size:
            self.expert_out_dim = cluster_sizes[0]
            col_map = torch.stack(expert_indices_list, dim=0).to(device)
            self.register_buffer("expert_col_map", col_map)
        else:
            self.expert_out_dim = -1

        # Router in FP32 for numerical stability & speed
        self.router = nn.Linear(
            self.in_features, self.num_experts, device=device, dtype=torch.float32
        )
        nn.init.normal_(self.router.weight, std=0.02)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        self.router.weight.requires_grad = True
        if self.router.bias is not None:
            self.router.bias.requires_grad = True

        # Trainable LoRA Adapters
        if self.lora_r > 0:
            self.expert_lora_A = nn.ParameterList()
            self.expert_lora_B = nn.ParameterList()
            for e in range(self.num_experts):
                n_neurons = cluster_sizes[e]
                l_A = nn.Parameter(
                    torch.empty(
                        self.lora_r, self.in_features, device=device, dtype=dtype
                    )
                )
                l_B = nn.Parameter(
                    torch.zeros(n_neurons, self.lora_r, device=device, dtype=dtype)
                )
                nn.init.kaiming_uniform_(l_A, a=math.sqrt(5))
                nn.init.zeros_(l_B)
                self.expert_lora_A.append(l_A)
                self.expert_lora_B.append(l_B)

            self.lora_A_router = nn.Parameter(
                torch.empty(
                    self.lora_r, self.in_features, device=device, dtype=torch.float32
                )
            )
            self.lora_B_router = nn.Parameter(
                torch.zeros(
                    self.num_experts, self.lora_r, device=device, dtype=torch.float32
                )
            )
            nn.init.kaiming_uniform_(self.lora_A_router, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_router)

    @property
    def masks(self) -> list[torch.Tensor]:
        return [self._masks[e] for e in range(self.num_experts)]

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective full weight matrix (out_features, in_features)."""
        device = self.experts[0].weight.device
        dtype = self.experts[0].weight.dtype
        W = torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)
        for e, expert in enumerate(self.experts):
            mask = self._masks[e]
            W[mask] = expert.weight
        return W

    @property
    def bias(self) -> torch.Tensor | None:
        """Reconstructed effective full bias vector (out_features,)."""
        if self.experts[0].bias is None:
            return None
        device = self.experts[0].bias.device
        dtype = self.experts[0].bias.dtype
        b = torch.zeros(self.out_features, device=device, dtype=dtype)
        for e, expert in enumerate(self.experts):
            mask = self._masks[e]
            b[mask] = expert.bias
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        in_feat = orig_shape[-1]
        x_flat = x.reshape(-1, in_feat)
        N = x_flat.shape[0]

        # Compute router logits in FP32
        router_logits = self.router(x_flat.float())
        if (
            self.lora_r > 0
            and hasattr(self, "lora_A_router")
            and hasattr(self, "lora_B_router")
        ):
            router_lora = (
                (x_flat.float() @ self.lora_A_router.t())
                @ self.lora_B_router.t()
                * self.lora_scale
            )
            router_logits = router_logits + router_lora

        if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            router_logits = torch.nan_to_num(
                router_logits, nan=0.0, posinf=1.0, neginf=-1.0
            )

        # Top-k routing
        topk_vals, topk_idx = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # (N, top_k)
        topk_weights = torch.softmax(topk_vals, dim=-1).to(x.dtype)  # (N, top_k)

        out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)

        if self.equal_expert_size:
            # Fast vectorized gather & scatter_add_ for equal-sized expert partitions
            D_exp = self.expert_out_dim
            all_expert_outs = torch.stack(
                [exp(x_flat) for exp in self.experts], dim=1
            )  # (N, E, D_exp)

            if self.lora_r > 0 and hasattr(self, "expert_lora_A"):
                lora_outs = []
                for e in range(self.num_experts):
                    l_e = (
                        (
                            x_flat.to(self.expert_lora_A[e].dtype)
                            @ self.expert_lora_A[e].t()
                        )
                        @ self.expert_lora_B[e].t()
                        * self.lora_scale
                    )
                    lora_outs.append(l_e.to(x.dtype))
                all_expert_outs = all_expert_outs + torch.stack(lora_outs, dim=1)

            expanded_idx = topk_idx.unsqueeze(-1).expand(N, self.top_k, D_exp)
            selected_outs = torch.gather(
                all_expert_outs, dim=1, index=expanded_idx
            )  # (N, top_k, D_exp)
            weighted_selected = selected_outs * topk_weights.unsqueeze(
                -1
            )  # (N, top_k, D_exp)

            col_map = self.expert_col_map
            dest_cols = col_map[topk_idx]  # (N, top_k, D_exp)

            out.scatter_add_(
                dim=1,
                index=dest_cols.reshape(N, self.top_k * D_exp),
                src=weighted_selected.reshape(N, self.top_k * D_exp),
            )
        else:
            # Vectorized token dispatch per expert with scatter_add_
            for e in range(self.num_experts):
                expert_match = topk_idx == e  # (N, top_k)
                if not expert_match.any():
                    continue
                token_idx, rank_pos = torch.where(expert_match)
                selected_tokens = x_flat[token_idx]
                exp_out = self.experts[e](selected_tokens)
                if self.lora_r > 0 and hasattr(self, "expert_lora_A"):
                    l_e = (
                        (
                            selected_tokens.to(self.expert_lora_A[e].dtype)
                            @ self.expert_lora_A[e].t()
                        )
                        @ self.expert_lora_B[e].t()
                        * self.lora_scale
                    )
                    exp_out = exp_out + l_e.to(x.dtype)
                gate_w = topk_weights[token_idx, rank_pos].unsqueeze(-1)
                weighted_exp_out = exp_out * gate_w

                exp_cols = self.expert_indices_padded[e, : self.expert_lens[e]]
                flat_idx = token_idx[:, None] * self.out_features + exp_cols
                out.view(-1).scatter_add_(
                    0, flat_idx.reshape(-1), weighted_exp_out.reshape(-1)
                )

        return out.reshape(*orig_shape[:-1], self.out_features)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("tomoe_class")
@COMPRESSOR_REGISTRY.register("tomoe_compressor_class")
class ToMoECompressor(BaseCompressor):
    """
    ToMoE Compressor: dynamically replaces dense feed-forward and projection layers
    with an MoE-fied equivalent.

    Args:
        model: PyTorch model.
        num_experts: Number of expert sub-layers to form (default: 4).
        top_k: Number of experts to route each token to (default: 2).
        max_error: Maximum allowed relative Frobenius error on reconstruction (default: 0.20).
        lora_r: LoRA adapter rank on experts and router (0 = disabled).
        lora_alpha: LoRA scaling factor (default: 16.0).
        freeze_experts: Whether to freeze expert base weights (default: True).
        target_attention_only: Whether to restrict to attention layers only.
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
        target_attention_only: bool = False,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model, name="tomoe", **kwargs)
        cfg = kwargs.pop("comp_config", kwargs.pop("cfg", None))
        if cfg is not None:
            num_experts = _cfg_get(
                cfg, "num_experts", _cfg_get(cfg, "tomoe_num_experts", num_experts)
            )
            top_k = _cfg_get(cfg, "top_k", _cfg_get(cfg, "tomoe_top_k", top_k))
            max_error = _cfg_get(
                cfg, "max_error", _cfg_get(cfg, "max_reconstruction_error", max_error)
            )
            lora_r = _cfg_get(cfg, "lora_r", _cfg_get(cfg, "tomoe_lora_r", lora_r))
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            freeze_experts = _cfg_get(cfg, "freeze_experts", freeze_experts)
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )

        num_experts = int(kwargs.pop("num_experts", num_experts))
        top_k = int(kwargs.pop("top_k", top_k))
        max_error = float(kwargs.pop("max_error", max_error))
        lora_r = int(kwargs.pop("lora_r", lora_r))
        lora_alpha = float(kwargs.pop("lora_alpha", lora_alpha))
        freeze_experts = bool(kwargs.pop("freeze_experts", freeze_experts))
        target_attention_only = bool(
            kwargs.pop("target_attention_only", target_attention_only)
        )

        self.num_experts = max(1, num_experts)
        self.top_k = max(1, min(top_k, self.num_experts))
        self.max_error = float(max_error)
        self.lora_r = max(0, lora_r)
        self.lora_alpha = float(lora_alpha)
        self.freeze_experts = bool(freeze_experts)
        self.target_attention_only = target_attention_only
        self.target_modules = target_modules
        self.stats: dict[str, Any] = {}

    def _is_protected(self, name: str) -> bool:
        low = name.lower()
        return any(kw in low for kw in self.PROTECTED_KEYWORDS)

    def _is_target(self, name: str) -> bool:
        if self._is_protected(name):
            return False
        if not should_compress_linear(self.model, name):
            return False
        # Exclude fused QKV (c_attn, in_proj_qkv) to prevent mixing semantic groups across heads in clustering
        if "c_attn" in name.lower() or "in_proj" in name.lower():
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        low = name.lower()
        if self.target_attention_only:
            attn_keywords = [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "c_proj",
                "query",
                "key",
                "value",
            ]
            return any(k in low for k in attn_keywords)
        # Default: target FFN, MLP, and linear projections across all supported families
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

    def _get_parent(self, full_name: str) -> tuple[nn.Module | None, str]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None, ""
        return parent, parts[-1]

    def compress(
        self, orig_model: nn.Module | None = None
    ) -> tuple[nn.Module, dict[str, Any]]:
        """
        Compress target layers into ToMoE Mixture of Experts layers with full error validation.
        """
        if orig_model is not None:
            self.model = orig_model

        orig_total_params = sum(p.numel() for p in self.model.parameters())
        converted_layers = []
        skipped_layers = []
        layer_details = {}

        seen_weight_ptrs: set[int] = set()

        for name, module in list(self.model.named_modules()):
            if not self._is_target(name):
                continue

            # Fix 2: Check for supported layer type (nn.Linear or Conv1D) on base module
            base_mod = getattr(module, "base_layer", module)
            cls_name = base_mod.__class__.__name__.lower()
            is_linear = isinstance(base_mod, nn.Linear)
            is_conv1d = "conv1d" in cls_name or (
                hasattr(base_mod, "nx") and hasattr(base_mod, "nf")
            )

            if not (
                is_linear
                or is_conv1d
                or (hasattr(base_mod, "weight") and base_mod.weight.ndim == 2)
            ):
                logger.warning(
                    "ToMoE: Skipping unsupported layer type %s for %s",
                    type(base_mod).__name__,
                    name,
                )
                skipped_layers.append(
                    (name, f"unsupported_type_{type(base_mod).__name__}")
                )
                continue

            # H20: Detect and protect tied weights (e.g. wte and lm_head sharing same storage)
            if hasattr(base_mod, "weight") and base_mod.weight is not None:
                w_ptr = base_mod.weight.data_ptr()
                if w_ptr in seen_weight_ptrs:
                    logger.warning(
                        "ToMoE: Skipping tied module %s with shared weight storage (H20)",
                        name,
                    )
                    skipped_layers.append((name, "tied_weight_shared_storage"))
                    continue
                seen_weight_ptrs.add(w_ptr)

            # Determine weights and feature shapes ensuring orientation is (out_features, in_features)
            if is_conv1d:
                in_f = base_mod.weight.shape[0]
                out_f = base_mod.weight.shape[1]
                orig_w_2d = base_mod.weight.data.t().contiguous()
            else:
                in_f = getattr(base_mod, "in_features", base_mod.weight.shape[1])
                out_f = getattr(base_mod, "out_features", base_mod.weight.shape[0])
                orig_w_2d = base_mod.weight.data.contiguous()

            if torch.linalg.norm(orig_w_2d.float()) < 1e-4:
                skipped_layers.append((name, "frobenius_norm_too_small"))
                continue

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

            # Meaningful fidelity check: activation reconstruction error on a calibration batch
            with torch.no_grad():
                device = orig_w_2d.device
                dtype = orig_w_2d.dtype
                calib_x = torch.randn(4, 16, in_f, device=device, dtype=dtype)
                orig_out = module(calib_x)
                moe_out = moe_layer(calib_x)
                err = (
                    torch.norm(moe_out.float() - orig_out.float())
                    / (torch.norm(orig_out.float()) + 1e-8)
                ).item()

            if err > self.max_error or np.isnan(err):
                logger.warning(
                    "ToMoE: %s activation reconstruction error (%.4f) exceeded max_error (%.4f). Keeping original layer.",
                    name,
                    err,
                    self.max_error,
                )
                skipped_layers.append((name, f"error_threshold_exceeded_{err:.4f}"))
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # Replace layer in parent module preserving outer LoRALinear wrapper if present
            orig_p_count = sum(p.numel() for p in module.parameters())
            new_p_count = sum(p.numel() for p in moe_layer.parameters())
            if hasattr(module, "base_layer"):
                module.base_layer = moe_layer
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    for p in (
                        module.lora_A.parameters()
                        if hasattr(module.lora_A, "parameters")
                        else [module.lora_A]
                    ):
                        p.requires_grad = True
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    for p in (
                        module.lora_B.parameters()
                        if hasattr(module.lora_B, "parameters")
                        else [module.lora_B]
                    ):
                        p.requires_grad = True
            else:
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


@COMPRESSOR_REGISTRY.register("tomoe")
@COMPRESSOR_REGISTRY.register("tomoe_compressor")
@COMPRESSOR_REGISTRY.register("tomoe_factory")
@COMPRESSOR_REGISTRY.register("tomoe_compressor_factory")
def tomoe_compressor_factory(
    model: nn.Module,
    comp_config: Any = None,
    num_experts: int = 4,
    top_k: int = 2,
    **kwargs: Any,
) -> Any:
    """Factory function for ToMoE compressor."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    target_attn = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    compressor = ToMoECompressor(
        model,
        num_experts=num_experts,
        top_k=top_k,
        comp_config=comp_config,
        target_attention_only=target_attn,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


tomoe_factory = tomoe_compressor_factory


@COMPRESSOR_REGISTRY.register("inject_tomoe")
@COMPRESSOR_REGISTRY.register("tomoe_inject")
def inject_tomoe(model: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook for ToMoE."""
    if model is None:
        return None
    try:
        return tomoe_compressor_factory(model, **kwargs)
    except Exception as e:
        logger.error("inject_tomoe failed: %s", e)
        return None
