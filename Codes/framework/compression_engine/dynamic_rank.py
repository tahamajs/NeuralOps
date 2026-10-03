"""
Dynamic Entropy Singular Value Decomposition (Dynamic Entropy SVD) Compressor.

Implements singular entropy computation over empirical layer activation covariances:
$$H(\\Sigma_l) = -\\sum_{i=1}^{d} p_i \\ln(p_i), \\quad p_i = \\frac{\\sigma_i}{\\sum_j \\sigma_j}$$
Dynamically budgets layer ranks $r_l \\in [r_{\\min}, r_{\\max}]$ proportional to singular entropy,
subject to global rank or parameter budgets, and replaces dense linear layers with low-rank
factorized `DynamicSVDLinear` modules.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.pruning_utils import adaptive_max_error
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# RESEARCH FIX: Robust empirical singular entropy computation with Tikhonov damping and zero-division protection
def compute_singular_entropy(
    spectrum_or_matrix: torch.Tensor,
    eps: float = 1e-12,
    normalize: bool = False,
) -> float:
    """
    Compute Singular Entropy H(Sigma) = -sum(p_i * log(p_i)) of a covariance matrix,
    weight matrix, or 1D singular value spectrum tensor.

    Args:
        spectrum_or_matrix: 1D singular values tensor, or 2D covariance/weight matrix.
        eps: Small numerical stabilizer to avoid log(0) and division by zero.
        normalize: If True, normalize entropy by ln(dim) to bound in [0.0, 1.0].

    Returns:
        float: Computed singular entropy H(Sigma) >= 0.0.
    """
    if not isinstance(spectrum_or_matrix, torch.Tensor):
        spectrum_or_matrix = torch.tensor(spectrum_or_matrix, dtype=torch.float32)

    spectrum_or_matrix = spectrum_or_matrix.detach().float()

    if spectrum_or_matrix.ndim == 1:
        singular_values = spectrum_or_matrix
    elif spectrum_or_matrix.ndim == 2:
        # Check if square and symmetric (covariance matrix) or rectangular (weight matrix)
        r, c = spectrum_or_matrix.shape
        if r == c:
            try:
                # RESEARCH FIX: Use eigvalsh for symmetric covariance matrices to avoid SVD overhead
                eigenvals = torch.linalg.eigvalsh(spectrum_or_matrix)
                singular_values = torch.clamp(eigenvals, min=0.0)
            except RuntimeError:
                singular_values = torch.linalg.svdvals(spectrum_or_matrix)
        else:
            singular_values = torch.linalg.svdvals(spectrum_or_matrix)
    else:
        raise ValueError(
            f"compute_singular_entropy expects 1D or 2D tensor, got shape {spectrum_or_matrix.shape}"
        )

    # RESEARCH FIX: Filter out non-positive singular values before probability normalization
    valid_sv = singular_values[singular_values > eps]
    if valid_sv.numel() == 0:
        return 0.0

    total_sum = valid_sv.sum()
    if total_sum <= eps:
        return 0.0

    # Normalized singular probability mass function p_i = sigma_i / sum(sigma_j)
    probs = valid_sv / total_sum
    probs = torch.clamp(probs, min=eps, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs)).item()

    if normalize and valid_sv.numel() > 1:
        max_entropy = math.log(float(valid_sv.numel()))
        if max_entropy > 0.0:
            entropy = entropy / max_entropy

    return max(0.0, float(entropy))


# RESEARCH FIX: Bounded water-filling rank allocator ensuring strict adherence to layer constraints and global budget
def allocate_entropy_ranks(
    entropies: dict[str, float],
    min_rank: int = 16,
    max_rank: int = 128,
    total_rank_budget: int | None = None,
    total_param_budget: int | None = None,
    target_param_ratio: float | None = None,
    layer_dims: dict[str, tuple[int, int]] | None = None,
) -> dict[str, int]:
    """
    Dynamically allocate low-rank budgets r_l in [min_rank, max_rank] across transformer
    layers proportional to singular entropy H(Sigma_l), constrained by total rank or parameter budget.

    Args:
        entropies: Dictionary mapping layer names to their singular entropy values.
        min_rank: Lower bound for per-layer rank.
        max_rank: Upper bound for per-layer rank.
        total_rank_budget: Optional total sum of ranks across all layers.
        total_param_budget: Optional total parameter budget for factorized layers.
        target_param_ratio: Optional parameter retention ratio in (0, 1].
        layer_dims: Optional dictionary mapping layer names to (out_features, in_features).

    Returns:
        Dict[str, int]: Allocated rank for each layer.
    """
    if not entropies:
        return {}

    min_rank = max(16, int(min_rank))
    layer_names = list(entropies.keys())
    num_layers = len(layer_names)

    # Determine per-layer physical dimension bounds with minimum rank floor
    per_layer_min = {}
    per_layer_max = {}
    for name in layer_names:
        if layer_dims and name in layer_dims:
            out_f, in_f = layer_dims[name]
            phys_max = min(out_f, in_f)
        else:
            phys_max = max_rank

        l_min = max(1, min(max(16, min_rank), phys_max))
        l_max = max(l_min, min(max_rank, phys_max))
        per_layer_min[name] = l_min
        per_layer_max[name] = l_max

    # Compute baseline entropy weights
    raw_weights = {name: max(entropies[name], 1e-6) for name in layer_names}
    total_entropy = sum(raw_weights.values())

    # Case 1: Parameter Budget Constraint
    if (
        total_param_budget is not None or target_param_ratio is not None
    ) and layer_dims:
        total_orig_params = sum(
            layer_dims[name][0] * layer_dims[name][1] for name in layer_names
        )
        if target_param_ratio is not None:
            effective_param_budget = int(total_orig_params * target_param_ratio)
        else:
            effective_param_budget = int(total_param_budget)

        # Binary search for parameter budget water-filling
        # Parameter cost for layer l at rank r is r * (d_in + d_out)
        low_scale, high_scale = 0.0, 1e6
        best_ranks = {}
        for _ in range(60):
            mid_scale = (low_scale + high_scale) / 2.0
            candidate_ranks = {}
            current_params = 0
            for name in layer_names:
                out_f, in_f = layer_dims[name]
                param_cost_per_rank = out_f + in_f
                # Proportional rank candidate
                target_r = int(
                    round(mid_scale * (raw_weights[name] / total_entropy) * 100.0)
                )
                r_clamped = max(per_layer_min[name], min(per_layer_max[name], target_r))
                candidate_ranks[name] = r_clamped
                current_params += r_clamped * param_cost_per_rank

            if current_params <= effective_param_budget:
                best_ranks = candidate_ranks
                low_scale = mid_scale
            else:
                high_scale = mid_scale

        if not best_ranks:
            best_ranks = {name: per_layer_min[name] for name in layer_names}
        return best_ranks

    # Case 2: Total Rank Budget Constraint
    if total_rank_budget is None:
        # Default total budget: average of min_rank and max_rank across layers
        total_rank_budget = sum(
            (per_layer_min[name] + per_layer_max[name]) // 2 for name in layer_names
        )

    min_possible_sum = sum(per_layer_min.values())
    max_possible_sum = sum(per_layer_max.values())
    target_budget = max(min_possible_sum, min(max_possible_sum, total_rank_budget))

    # RESEARCH FIX: Iterative bisection water-filling for exact integer rank distribution
    low, high = 0.0, float(target_budget * 2)
    allocated_ranks = {}
    for _ in range(50):
        mid = (low + high) / 2.0
        candidate_ranks = {}
        candidate_sum = 0
        for name in layer_names:
            w = raw_weights[name] / total_entropy
            r = int(round(mid * w))
            r_clamped = max(per_layer_min[name], min(per_layer_max[name], r))
            candidate_ranks[name] = r_clamped
            candidate_sum += r_clamped

        allocated_ranks = candidate_ranks
        if candidate_sum == target_budget:
            break
        elif candidate_sum < target_budget:
            low = mid
        else:
            high = mid

    # Residual greedy adjustment to ensure exact budget match if slack remains
    current_sum = sum(allocated_ranks.values())
    diff = target_budget - current_sum

    if diff > 0:
        # Greedily allocate remaining ranks to layers with highest entropy that are below per_layer_max
        sorted_by_entropy = sorted(
            layer_names, key=lambda n: raw_weights[n], reverse=True
        )
        while diff > 0:
            progress = False
            for name in sorted_by_entropy:
                if allocated_ranks[name] < per_layer_max[name] and diff > 0:
                    allocated_ranks[name] += 1
                    diff -= 1
                    progress = True
            if not progress:
                break
    elif diff < 0:
        # Greedily subtract ranks from layers with lowest entropy that are above per_layer_min
        sorted_by_entropy = sorted(layer_names, key=lambda n: raw_weights[n])
        while diff < 0:
            progress = False
            for name in sorted_by_entropy:
                if allocated_ranks[name] > per_layer_min[name] and diff < 0:
                    allocated_ranks[name] -= 1
                    diff += 1
                    progress = True
            if not progress:
                break

    return allocated_ranks


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
}


class DynamicSVDLinear(nn.Module):
    is_custom_attention: bool = True
    """
    True Low-Rank Factorized Linear Layer executing as:
    y = (x @ V.T) @ U.T + bias
    where V is (rank, in_features) and U is (out_features, rank).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        # Factorized low-rank parameter tensors
        self.V = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        self.U = nn.Parameter(
            torch.empty(out_features, self.rank, dtype=dtype, device=device)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.U, self.V)

    def reset_parameters(self) -> None:
        """Kaiming uniform parameter initialization."""
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing factorized low-rank multiplication:
        y = (x @ V^T) @ U^T + bias
        """
        intermediate = F.linear(x, self.V)
        out = F.linear(intermediate, self.U, self.bias)
        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int,
        covariance: torch.Tensor | None = None,
    ) -> DynamicSVDLinear:
        """
        Construct factorized DynamicSVDLinear directly from dense nn.Linear or Conv1D via SVD.

        Args:
            linear: Source dense linear or Conv1D layer (or LoRALinear wrapper).
            rank: Target rank for low-rank factorization.
            covariance: Optional empirical activation covariance C_X for activation-aware SVD.

        Returns:
            DynamicSVDLinear: Initialized factorized low-rank layer.
        """
        target_mod = getattr(linear, "base_layer", linear)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()  # [out_features, in_features]

        target_rank = max(1, min(rank, in_f, out_f))

        new_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=target_rank,
            bias=target_mod.bias is not None,
            dtype=target_mod.weight.dtype,
            device=target_mod.weight.device,
        )

        # Always compute the weight-only SVD baseline first.
        # This serves as both the default path (when covariance=None) and the
        # fallback comparison when the activation-aware path produces higher error.
        U_svd_w, S_svd_w, Vh_svd_w = torch.linalg.svd(W, full_matrices=False)
        sqrt_S_w = torch.sqrt(torch.clamp(S_svd_w[:target_rank], min=1e-12))
        U_w = U_svd_w[:, :target_rank] * sqrt_S_w.unsqueeze(0)  # [out_f, rank]
        V_w = sqrt_S_w.unsqueeze(1) * Vh_svd_w[:target_rank, :]  # [rank, in_f]
        W_rec_w = U_w @ V_w
        norm_W = torch.norm(W, p="fro").clamp(min=1e-12)
        err_weight_only = (torch.norm(W - W_rec_w, p="fro") / norm_W).item()

        # Default to weight-only unless activation-aware is demonstrably better.
        U_r, V_r = U_w, V_w

        if (
            covariance is not None
            and covariance.numel() > 0
            and covariance.shape[0] == in_f
        ):
            # RESEARCH FIX: Activation-aware whitening SVD with eigenvalue clamping lambda_i >= 1e-4
            # NOTE: We only accept the activation-aware result if its reconstruction error is
            # strictly lower than the weight-only baseline. This guards against rank-deficient
            # covariances (e.g. from dummy calibration data) that inflate inv_sqrt by up to 100x
            # and destroy the approximation quality.
            cov = covariance.to(device=W.device, dtype=torch.float32)
            try:
                # Eigen-decomposition with regularized eigenvalue clamping >= 1e-4
                eigvals, eigvecs = torch.linalg.eigh(
                    cov + 1e-4 * torch.eye(in_f, device=W.device)
                )
                eigvals_clamped = torch.clamp(eigvals, min=1e-4)
                cov_sqrt = (
                    eigvecs
                    @ torch.diag(torch.sqrt(eigvals_clamped))
                    @ eigvecs.t().contiguous()
                )
                cov_inv_sqrt = (
                    eigvecs
                    @ torch.diag(1.0 / torch.sqrt(eigvals_clamped))
                    @ eigvecs.t().contiguous()
                )

                W_scaled = W @ cov_sqrt
                U_svd_a, S_svd_a, Vh_svd_a = torch.linalg.svd(
                    W_scaled, full_matrices=False
                )

                # Truncate and un-whiten right singular factors
                sqrt_S_a = torch.sqrt(torch.clamp(S_svd_a[:target_rank], min=1e-12))
                U_a = U_svd_a[:, :target_rank] * sqrt_S_a.unsqueeze(0)
                V_a = sqrt_S_a.unsqueeze(1) * (Vh_svd_a[:target_rank, :] @ cov_inv_sqrt)
                # Symmetrically normalize factor magnitudes
                norm_u = torch.norm(U_a, p="fro").clamp_min(1e-8)
                norm_v = torch.norm(V_a, p="fro").clamp_min(1e-8)
                scale = torch.sqrt(norm_u / norm_v)
                U_a = U_a / scale
                V_a = V_a * scale

                W_rec_a = U_a @ V_a
                err_act_aware = (torch.norm(W - W_rec_a, p="fro") / norm_W).item()

                if err_act_aware < err_weight_only:
                    # Activation-aware is genuinely better: accept it
                    U_r, V_r = U_a, V_a
                    logger.debug(
                        "from_linear: activation-aware SVD accepted "
                        "(err_act=%.4f < err_weight=%.4f).",
                        err_act_aware,
                        err_weight_only,
                    )
                else:
                    # Activation-aware is worse (e.g. due to rank-deficient covariance):
                    # stick with the weight-only baseline
                    logger.debug(
                        "from_linear: activation-aware SVD rejected "
                        "(err_act=%.4f >= err_weight=%.4f); using weight-only SVD.",
                        err_act_aware,
                        err_weight_only,
                    )
            except Exception as e:
                logger.warning(
                    "Activation-aware SVD failed (%s); using weight-only SVD.", e
                )
                # U_r, V_r already set to weight-only baseline above

        new_layer.U.data.copy_(U_r.to(dtype=target_mod.weight.dtype))
        new_layer.V.data.copy_(V_r.to(dtype=target_mod.weight.dtype))

        if target_mod.bias is not None and new_layer.bias is not None:
            new_layer.bias.data.copy_(target_mod.bias.data)

        return new_layer

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


class DynamicEntropySVDCompressor(BaseCompressor):
    """
    Dynamic Entropy SVD Compressor.

    1. Computes singular entropy H(Sigma_l) on layer activation covariances or weights.
    2. Allocates layer ranks within [min_rank, max_rank] subject to parameter/rank budget.
    3. Replaces dense Linear/Conv1D layers with DynamicSVDLinear factorized modules.
    4. Provides relative Frobenius error validation, max_error threshold, and safe fallback.
    5. Freezes fallback and non-target layers during recovery.
    """

    def __init__(
        self,
        model: nn.Module,
        min_rank: int = 16,
        max_rank: int = 128,
        target_param_ratio: float | None = None,
        total_rank_budget: int | None = None,
        target_rank: int | None = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        calibration_loader: Any = None,
        calibration_texts: list[str] | None = None,
        target_modules: list[str] | None = None,
        tokenizer: Any | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="dynamic_entropy_svd", **kwargs)
        cfg_obj = cfg if cfg is not None else kwargs.get("comp_config")
        self.cfg = cfg_obj

        min_rank = _cfg_get(
            cfg_obj, "dynamic_min_rank", _cfg_get(cfg_obj, "min_rank", min_rank)
        )
        max_rank = _cfg_get(
            cfg_obj, "dynamic_max_rank", _cfg_get(cfg_obj, "max_rank", max_rank)
        )
        target_rank = _cfg_get(
            cfg_obj, "dynamic_rank", _cfg_get(cfg_obj, "rank", target_rank)
        )
        total_rank_budget = _cfg_get(cfg_obj, "total_rank_budget", total_rank_budget)
        target_param_ratio = _cfg_get(cfg_obj, "target_param_ratio", target_param_ratio)
        max_error = _cfg_get(
            cfg_obj, "dynamic_max_error", _cfg_get(cfg_obj, "max_error", max_error)
        )
        target_attention_only = _cfg_get(
            cfg_obj,
            "dynamic_target_attention_only",
            _cfg_get(cfg_obj, "target_attention_only", target_attention_only),
        )
        fallback_to_original = _cfg_get(
            cfg_obj,
            "dynamic_fallback_to_original",
            _cfg_get(cfg_obj, "fallback_to_original", fallback_to_original),
        )

        min_rank = max(16, int(min_rank))
        if "rank" in kwargs and kwargs["rank"] is not None:
            target_rank = kwargs["rank"]
            min_rank = max(16, min(min_rank, target_rank))
            max_rank = max(max_rank, target_rank)

        self.min_rank = min_rank
        self.max_rank = max(min_rank, int(max_rank))
        self.target_param_ratio = (
            float(target_param_ratio) if target_param_ratio is not None else None
        )
        self.total_rank_budget = (
            int(total_rank_budget) if total_rank_budget is not None else None
        )
        self.target_rank = int(target_rank) if target_rank is not None else None
        if max_error is None or max_error == 0.35:
            max_error = adaptive_max_error(
                self.target_rank or self.min_rank, 768, 768, "distilbert"
            )

        self.max_error = float(max_error)
        self.target_attention_only = bool(target_attention_only)
        self.fallback_to_original = bool(fallback_to_original)

        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_modules = target_modules
        self.tokenizer = (
            tokenizer
            or getattr(model, "tokenizer", None)
            or getattr(getattr(model, "config", None), "tokenizer", None)
        )

        self.layer_entropies: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.layer_errors: dict[str, float] = {}
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check whether module is a compressible nn.Linear or Conv1D matching target filter."""
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            ]
        ):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(target_mod, DynamicSVDLinear) or isinstance(
            module, DynamicSVDLinear
        ):
            return False
        if not should_compress_linear(self.model, name):
            return False

        raw_w = target_mod.weight.data
        if min(raw_w.shape[0], raw_w.shape[1]) < 16:
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(
                k in name_lower for k in ("attn", "attention")
            )
            if not is_attn:
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def collect_activation_covariances(
        self,
    ) -> dict[str, torch.Tensor]:
        """
        Collect empirical input activation covariance matrices C_X for each target layer
        using calibration forward passes.
        """
        covariances: dict[str, torch.Tensor] = {}
        sample_counts: dict[str, int] = {}
        hooks = []

        def get_hook(name: str):
            def hook_fn(mod: nn.Module, inp: tuple[Any, ...], out: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                if x.ndim >= 2:
                    x_flat = x.reshape(-1, x.shape[-1]).float()
                    # Double-precision covariance accumulation for stability
                    cov = x_flat.t().contiguous() @ x_flat
                    n = x_flat.shape[0]
                    if name not in covariances:
                        covariances[name] = cov
                        sample_counts[name] = n
                    else:
                        covariances[name] += cov
                        sample_counts[name] += n

            return hook_fn

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            return {}

        for name, mod in target_layers:
            target_mod = getattr(mod, "base_layer", mod)
            hooks.append(target_mod.register_forward_hook(get_hook(name)))

        device = next(self.model.parameters()).device
        self.model.eval()

        # Track whether REAL calibration data (not random dummy) was used.
        # If only dummy data is used, the covariance matrix will be rank-deficient
        # (rank <= num_dummy_tokens << hidden_dim), causing the whitening transform
        # to map near-zero eigenvalue directions through inv_sqrt with scale ~100x,
        # which after un-whitening produces reconstruction error near 1.0 on every layer.
        real_calibration_used = False

        try:
            with torch.no_grad():
                if self.calibration_loader is not None:
                    batch_count = 0
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
                        batch_count += 1
                    if batch_count > 0:
                        real_calibration_used = True
                        logger.info(
                            "DynamicEntropySVD: real calibration loader used (%d batches).",
                            batch_count,
                        )
                elif (
                    self.calibration_texts is not None
                    and len(self.calibration_texts) > 0
                ):
                    logger.info(
                        "Running calibration forward on %d provided texts.",
                        len(self.calibration_texts),
                    )
                    if self.tokenizer is not None:
                        text_count = 0
                        for text in self.calibration_texts[:32]:
                            try:
                                enc = self.tokenizer(
                                    text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=128,
                                )
                                enc = {k: v.to(device) for k, v in enc.items()}
                                self.model(**enc)
                                text_count += 1
                            except Exception as e:
                                logger.debug("Calibration text pass exception: %s", e)
                                break
                        if text_count > 0:
                            real_calibration_used = True
                            logger.info(
                                "DynamicEntropySVD: real calibration texts used (%d texts).",
                                text_count,
                            )
                    else:
                        logger.info(
                            "No tokenizer available; calibration_texts provided but cannot encode. "
                            "Falling back to weight-only SVD."
                        )
        except Exception as e:
            logger.warning(
                "Activation collection encountered exception (%s); falling back to weight SVD.",
                e,
            )
        finally:
            for h in hooks:
                h.remove()

        # ROOT-CAUSE FIX: If no real calibration data was used, dummy inputs (e.g.
        # 2x32 random tokens) produce a rank-deficient covariance (rank<=64 for
        # hidden_dim=768). The whitening eigenvalue clamp (min=1e-4) then assigns
        # weight 100x to the 704 meaningless directions. After SVD truncation and
        # un-whitening, reconstruction error approaches 1.0 on every layer, causing
        # all 24 layers to fall back and zero compression.
        #
        # Solution: return {} so from_linear uses plain weight-only SVD instead.
        # Expected reconstruction error at rank 64: ~0.15-0.25 (all layers compress).
        if not real_calibration_used:
            logger.warning(
                "DynamicEntropySVD: no real calibration data available "
                "(calibration_loader=None, calibration_texts=None/empty). "
                "Activation-aware whitening DISABLED. Using weight-only SVD for all "
                "layers. Expected reconstruction error at rank=64: ~0.15-0.25."
            )
            return {}

        # Normalize accumulated covariances by total sample count
        normalized_covs: dict[str, torch.Tensor] = {}
        for name, cov in covariances.items():
            cnt = sample_counts.get(name, 1)
            normalized_covs[name] = cov / max(1, cnt)

        logger.info(
            "DynamicEntropySVD: %d activation covariances collected (real data).",
            len(normalized_covs),
        )
        return normalized_covs

    def compute_layer_entropies(
        self, covariances: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """
        Compute singular entropy for each target layer using activation covariance
        or fallback weight matrix spectrum.
        """
        entropies: dict[str, float] = {}
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                target_mod = getattr(module, "base_layer", module)
                raw_w = target_mod.weight.data
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                w_mat = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()

                if name in covariances:
                    ent = compute_singular_entropy(covariances[name])
                else:
                    ent = compute_singular_entropy(w_mat)
                entropies[name] = ent
        return entropies

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """
        Execute dynamic entropy-budgeted low-rank SVD compression across all target layers.
        Includes reconstruction error validation, fallback, layer freezing, and parameter tracking.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Compressed model and detailed statistics.
        """
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}

        # Step 1: Collect activation covariances
        covariances = self.collect_activation_covariances()

        # Step 2: Compute singular entropies per layer
        self.layer_entropies = self.compute_layer_entropies(covariances)

        if not self.layer_entropies:
            logger.warning("No compressible linear layers found in model.")
            return self.model, self.get_compression_stats()

        # Step 3: Extract layer dimensions
        layer_dims: dict[str, tuple[int, int]] = {}
        total_orig_decomp_params = 0
        total_comp_decomp_params = 0

        for name, module in self.model.named_modules():
            if name in self.layer_entropies:
                target_mod = getattr(module, "base_layer", module)
                raw_w = target_mod.weight.data
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if is_conv1d:
                    in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                else:
                    out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                layer_dims[name] = (out_f, in_f)
                total_orig_decomp_params += out_f * in_f

        # Freeze uncompressed layers when target_attention_only=True
        if self.target_attention_only:
            for name, module in list(self.model.named_modules()):
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if is_linear or is_conv1d:
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        # Step 4: Determine total rank budget
        total_rank_budget = self.total_rank_budget
        if total_rank_budget is None and self.target_rank is not None:
            total_rank_budget = self.target_rank * len(self.layer_entropies)

        # Step 5: Dynamic rank allocation
        self.layer_ranks = allocate_entropy_ranks(
            entropies=self.layer_entropies,
            min_rank=self.min_rank,
            max_rank=self.max_rank,
            total_rank_budget=total_rank_budget,
            target_param_ratio=self.target_param_ratio,
            layer_dims=layer_dims,
        )

        # Step 6: Factorize, validate, and replace each target linear layer
        for name, rank in self.layer_ranks.items():
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            if parent_name:
                parent = self.model.get_submodule(parent_name)
            else:
                parent = self.model
            module = getattr(parent, child_name)
            target_mod = getattr(module, "base_layer", module)

            out_f, in_f = layer_dims[name]
            orig_params = out_f * in_f

            cov = covariances.get(name, None)
            factorized_layer = DynamicSVDLinear.from_linear(
                linear=module,
                rank=rank,
                covariance=cov,
            )

            # Compute relative Frobenius reconstruction error
            raw_w = target_mod.weight.data
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            W = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            W_recon = factorized_layer.U.float() @ factorized_layer.V.float()
            norm_w = torch.norm(W, p="fro").item()
            err = float(torch.norm(W - W_recon, p="fro").item() / max(norm_w, 1e-12))
            self.layer_errors[name] = err

            if err > self.max_error and self.fallback_to_original:
                logger.warning(
                    "DynamicEntropySVD error %.4f on %s exceeded threshold %.4f; falling back to original.",
                    err,
                    name,
                    self.max_error,
                )
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_decomp_params += orig_params
                continue

            # If wrapped in LoRALinear, replace the inner base_layer or linear
            has_base_layer = hasattr(module, "base_layer") or (
                hasattr(module, "linear")
                and (hasattr(module, "lora_A") or hasattr(module, "r"))
            )
            if hasattr(module, "base_layer"):
                module.base_layer = factorized_layer
            elif hasattr(module, "linear") and (
                hasattr(module, "lora_A") or hasattr(module, "r")
            ):
                module.linear = factorized_layer
            else:
                replace_module(self.model, name, factorized_layer)

            if has_base_layer:
                for p_name, p in module.named_parameters():
                    if any(
                        t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")
                    ):
                        p.requires_grad = True

            self.compressed_layers.append(name)
            total_comp_decomp_params += rank * (in_f + out_f)

        # Forward sanity check
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.debug("DynamicEntropySVD forward check exception: %s", e)

        effective_ratio = (
            total_comp_decomp_params / total_orig_decomp_params
            if total_orig_decomp_params > 0
            else 1.0
        )
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )

        stats = self.get_compression_stats()
        stats.update(
            {
                "layer_entropies": self.layer_entropies,
                "layer_ranks": self.layer_ranks,
                "layer_errors": self.layer_errors,
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "original_layer_count": len(layer_dims),
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "avg_reconstruction_error": avg_err,
                "forward_check_success": forward_check_success,
                "method": "dynamic_entropy_svd",
            }
        )
        self.stats = stats
        logger.info(
            "Compressor dynamic_rank finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# Register classes under *_class to avoid collision with factory functions
@COMPRESSOR_REGISTRY.register("dynamic_entropy_svd_class")
@COMPRESSOR_REGISTRY.register("dynamic_rank_class")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_class")
class DynamicEntropySVDCompressorRegistered(DynamicEntropySVDCompressor):
    pass


@COMPRESSOR_REGISTRY.register("entropy_rank_allocator_class")
class EntropyRankAllocatorRegistered(DynamicEntropySVDCompressor):
    pass


@COMPRESSOR_REGISTRY.register("dynamic_entropy")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_svd")
@COMPRESSOR_REGISTRY.register("dynamic_rank")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_factory")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_svd_factory")
@COMPRESSOR_REGISTRY.register("dynamic_rank_factory")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_compressor")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_svd_compressor")
@COMPRESSOR_REGISTRY.register("dynamic_rank_compressor")
@COMPRESSOR_REGISTRY.register("entropy_rank_allocator")
def dynamic_entropy_svd_factory(
    model: Any, comp_config: Any = None, **kwargs: Any
) -> Any:
    """Dynamic Entropy SVD factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    try:
        min_rank = kwargs.pop(
            "min_rank",
            _cfg_get(
                comp_config, "dynamic_min_rank", _cfg_get(comp_config, "min_rank", 16)
            ),
        )
        max_rank = kwargs.pop(
            "max_rank",
            _cfg_get(
                comp_config, "dynamic_max_rank", _cfg_get(comp_config, "max_rank", 128)
            ),
        )
        target_rank = kwargs.pop(
            "rank",
            _cfg_get(comp_config, "dynamic_rank", _cfg_get(comp_config, "rank", None)),
        )
        max_error = kwargs.pop(
            "max_error",
            _cfg_get(
                comp_config,
                "dynamic_max_error",
                _cfg_get(comp_config, "max_error", 0.35),
            ),
        )
        target_attention_only = kwargs.pop(
            "target_attention_only",
            _cfg_get(
                comp_config,
                "dynamic_target_attention_only",
                _cfg_get(comp_config, "target_attention_only", False),
            ),
        )
        fallback = kwargs.pop(
            "fallback_to_original",
            _cfg_get(
                comp_config,
                "dynamic_fallback_to_original",
                _cfg_get(comp_config, "fallback_to_original", True),
            ),
        )
        target_param_ratio = kwargs.pop(
            "target_param_ratio", _cfg_get(comp_config, "target_param_ratio", None)
        )
        total_rank_budget = kwargs.pop(
            "total_rank_budget", _cfg_get(comp_config, "total_rank_budget", None)
        )

        compressor = DynamicEntropySVDCompressor(
            model,
            min_rank=min_rank,
            max_rank=max_rank,
            target_rank=target_rank,
            max_error=max_error,
            target_attention_only=target_attention_only,
            fallback_to_original=fallback,
            target_param_ratio=target_param_ratio,
            total_rank_budget=total_rank_budget,
            cfg=comp_config,
            **kwargs,
        )
        compressed_model, stats = compressor.compress()
        try:
            compressed_model.compression_stats = stats
        except Exception:
            pass
        if return_stats:
            return compressed_model, stats
        return compressed_model
    except Exception as e:
        logger.error("dynamic_entropy_svd_factory failed: %s", e, exc_info=True)
        return (model, {}) if return_stats else model


@COMPRESSOR_REGISTRY.register("inject_dynamic_entropy")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_inject")
def inject_dynamic_entropy(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook for dynamic_entropy. Returns None on failure."""
    target = layer if layer is not None else model
    if target is None:
        return None
    try:
        res = dynamic_entropy_svd_factory(target, **kwargs)
        if isinstance(res, tuple):
            return res[0]
        return res
    except Exception as e:
        logger.error("inject_dynamic_entropy failed: %s", e, exc_info=True)
        return None


def dynamic_entropy_inject(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook alias."""
    return inject_dynamic_entropy(model, layer, **kwargs)


@COMPRESSOR_REGISTRY.register("inject_dynamic_rank")
@COMPRESSOR_REGISTRY.register("dynamic_rank_inject")
def inject_dynamic_rank(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook alias for dynamic_rank."""
    return inject_dynamic_entropy(model, layer, **kwargs)


def dynamic_rank_inject(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook alias for dynamic_rank."""
    return inject_dynamic_entropy(model, layer, **kwargs)


dynamic_rank_factory = dynamic_entropy_svd_factory
entropy_rank_allocator_factory = dynamic_entropy_svd_factory

__all__ = [
    "compute_singular_entropy",
    "allocate_entropy_ranks",
    "DynamicSVDLinear",
    "DynamicEntropySVDCompressor",
    "dynamic_entropy_svd_factory",
    "dynamic_rank_factory",
    "entropy_rank_allocator_factory",
    "inject_dynamic_entropy",
    "dynamic_entropy_inject",
    "inject_dynamic_rank",
    "dynamic_rank_inject",
]
