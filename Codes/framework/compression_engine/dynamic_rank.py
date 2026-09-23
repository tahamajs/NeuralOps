"""
Dynamic Entropy SVD Compressor — Debug-Enhanced.

Bug-fix map:
  DE-FIX-01..10 (previous fixes preserved)
  DE-FIX-11  Explicit debug logging at every filter decision.
  DE-FIX-12  DynamicSVDLinear.is_custom_attention = False (it's a linear, not attention).
  DE-FIX-13  skip_should_compress_filter flag (default False) to bypass framework filter.
  DE-FIX-14  If no targets found, log a diagnostic breakdown.
  DE-FIX-15  Fallback to weight-only SVD if calibration yields nothing.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ===========================================================================
# Singular entropy
# ===========================================================================

def compute_singular_entropy(
    spectrum_or_matrix: torch.Tensor,
    eps: float = 1e-12,
    normalize: bool = False,
) -> float:
    if not isinstance(spectrum_or_matrix, torch.Tensor):
        spectrum_or_matrix = torch.tensor(spectrum_or_matrix, dtype=torch.float32)

    spectrum_or_matrix = spectrum_or_matrix.detach().float()

    if spectrum_or_matrix.ndim == 1:
        singular_values = spectrum_or_matrix
    elif spectrum_or_matrix.ndim == 2:
        r, c = spectrum_or_matrix.shape
        if r == c:
            sym = 0.5 * (spectrum_or_matrix + spectrum_or_matrix.t())
            if torch.allclose(spectrum_or_matrix, sym, atol=1e-4):
                try:
                    eigenvals = torch.linalg.eigvalsh(sym)
                    singular_values = torch.clamp(eigenvals, min=0.0)
                except RuntimeError:
                    singular_values = torch.linalg.svdvals(spectrum_or_matrix)
            else:
                singular_values = torch.linalg.svdvals(spectrum_or_matrix)
        else:
            singular_values = torch.linalg.svdvals(spectrum_or_matrix)
    else:
        raise ValueError(f"expected 1D/2D tensor, got {spectrum_or_matrix.shape}")

    valid_sv = singular_values[singular_values > eps]
    if valid_sv.numel() == 0:
        return 0.0
    total_sum = valid_sv.sum()
    if total_sum <= eps:
        return 0.0

    probs = valid_sv / total_sum
    probs = torch.clamp(probs, min=eps, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs)).item()
    if normalize and valid_sv.numel() > 1:
        max_entropy = math.log(float(valid_sv.numel()))
        if max_entropy > 0.0:
            entropy = entropy / max_entropy
    return max(0.0, float(entropy))


# ===========================================================================
# Rank allocation
# ===========================================================================

def allocate_entropy_ranks(
    entropies: Dict[str, float],
    min_rank: int = 4,
    max_rank: int = 128,
    total_rank_budget: Optional[int] = None,
    total_param_budget: Optional[int] = None,
    target_param_ratio: Optional[float] = None,
    layer_dims: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, int]:
    if not entropies:
        return {}

    layer_names = list(entropies.keys())
    num_layers = len(layer_names)
    min_rank = max(1, int(min_rank))
    max_rank = max(min_rank, int(max_rank))

    per_layer_min: Dict[str, int] = {}
    per_layer_max: Dict[str, int] = {}
    for name in layer_names:
        if layer_dims and name in layer_dims:
            out_f, in_f = layer_dims[name]
            phys_max = min(out_f, in_f)
        else:
            phys_max = max_rank
        l_min = max(1, min(min_rank, phys_max))
        l_max = max(l_min, min(max_rank, phys_max))
        per_layer_min[name] = l_min
        per_layer_max[name] = l_max

    raw_weights = {n: max(entropies[n], 1e-6) for n in layer_names}
    total_entropy = sum(raw_weights.values())
    if total_entropy <= 0:
        raw_weights = {n: 1.0 for n in layer_names}
        total_entropy = float(num_layers)

    # Case 1: Parameter budget
    if (total_param_budget is not None or target_param_ratio is not None) and layer_dims:
        total_orig_params = sum(layer_dims[n][0] * layer_dims[n][1] for n in layer_names)
        if target_param_ratio is not None:
            effective_param_budget = int(total_orig_params * float(target_param_ratio))
        else:
            effective_param_budget = int(total_param_budget)

        def compute_total_params(ranks):
            return sum(ranks[n] * (layer_dims[n][0] + layer_dims[n][1]) for n in layer_names)

        low, high = 0.0, float(total_entropy * max_rank)
        best_ranks = {n: per_layer_min[n] for n in layer_names}

        for _ in range(60):
            mid = 0.5 * (low + high)
            candidate = {}
            for n in layer_names:
                target_r = int(round(mid * raw_weights[n] / total_entropy))
                candidate[n] = max(per_layer_min[n], min(per_layer_max[n], target_r))
            if compute_total_params(candidate) <= effective_param_budget:
                best_ranks = candidate
                low = mid
            else:
                high = mid

        current_total = compute_total_params(best_ranks)
        sorted_desc = sorted(layer_names, key=lambda n: raw_weights[n], reverse=True)
        while current_total < effective_param_budget:
            progress = False
            for n in sorted_desc:
                if best_ranks[n] < per_layer_max[n]:
                    add_cost = layer_dims[n][0] + layer_dims[n][1]
                    if current_total + add_cost <= effective_param_budget:
                        best_ranks[n] += 1
                        current_total += add_cost
                        progress = True
            if not progress:
                break

        sorted_asc = sorted(layer_names, key=lambda n: raw_weights[n])
        while current_total > effective_param_budget:
            progress = False
            for n in sorted_asc:
                if best_ranks[n] > per_layer_min[n]:
                    sub_cost = layer_dims[n][0] + layer_dims[n][1]
                    best_ranks[n] -= 1
                    current_total -= sub_cost
                    progress = True
                    if current_total <= effective_param_budget:
                        break
            if not progress:
                break
        return best_ranks

    # Case 2: Total rank budget
    if total_rank_budget is None:
        total_rank_budget = sum((per_layer_min[n] + per_layer_max[n]) // 2 for n in layer_names)

    min_possible = sum(per_layer_min.values())
    max_possible = sum(per_layer_max.values())
    target_budget = max(min_possible, min(max_possible, int(total_rank_budget)))

    low, high = 0.0, float(target_budget * 4)
    allocated = {n: per_layer_min[n] for n in layer_names}
    for _ in range(60):
        mid = 0.5 * (low + high)
        candidate = {}
        cand_sum = 0
        for n in layer_names:
            w = raw_weights[n] / total_entropy
            r = max(per_layer_min[n], min(per_layer_max[n], int(round(mid * w))))
            candidate[n] = r
            cand_sum += r
        allocated = candidate
        if cand_sum == target_budget:
            break
        elif cand_sum < target_budget:
            low = mid
        else:
            high = mid

    current_sum = sum(allocated.values())
    diff = target_budget - current_sum
    if diff > 0:
        sorted_desc = sorted(layer_names, key=lambda n: raw_weights[n], reverse=True)
        while diff > 0:
            progress = False
            for n in sorted_desc:
                if allocated[n] < per_layer_max[n] and diff > 0:
                    allocated[n] += 1
                    diff -= 1
                    progress = True
            if not progress:
                break
    elif diff < 0:
        sorted_asc = sorted(layer_names, key=lambda n: raw_weights[n])
        while diff < 0:
            progress = False
            for n in sorted_asc:
                if allocated[n] > per_layer_min[n] and diff < 0:
                    allocated[n] -= 1
                    diff += 1
                    progress = True
            if not progress:
                break
    return allocated


# ===========================================================================
# Helpers
# ===========================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj",
               "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin"}


def _is_conv1d(module: nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    return "conv1d" in cls or hasattr(module, "nf")


def _canonical_weight(module: nn.Module) -> Optional[torch.Tensor]:
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    if w is None:
        return None
    if _is_conv1d(base):
        return w.t().contiguous()
    return w


# ===========================================================================
# DynamicSVDLinear (DE-FIX-12: NOT an attention layer)
# ===========================================================================

class DynamicSVDLinear(nn.Module):
    # DE-FIX-12: DynamicSVDLinear is a LINEAR replacement, not an attention layer.
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.U, self.V)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = F.linear(x, self.V)
        return F.linear(intermediate, self.U, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int,
        covariance: Optional[torch.Tensor] = None,
    ) -> "DynamicSVDLinear":
        target_mod = getattr(linear, "base_layer", linear)
        is_conv1d = _is_conv1d(target_mod)
        raw_w = target_mod.weight.data

        if is_conv1d:
            W = raw_w.t().contiguous().float()
            out_f, in_f = W.shape[0], W.shape[1]
        else:
            W = raw_w.float()
            out_f, in_f = W.shape[0], W.shape[1]

        target_rank = max(1, min(rank, in_f, out_f))
        new_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=target_rank,
            bias=target_mod.bias is not None,
            dtype=target_mod.weight.dtype,
            device=target_mod.weight.device,
        )

        if covariance is not None and covariance.numel() > 0 and covariance.shape[0] == in_f:
            try:
                cov = covariance.to(device=W.device, dtype=torch.float32)
                reg = 1e-4 * torch.eye(in_f, device=W.device, dtype=torch.float32)
                eigvals, eigvecs = torch.linalg.eigh(cov + reg)
                eigvals_clamped = torch.clamp(eigvals, min=1e-4)
                sqrt_ev = torch.sqrt(eigvals_clamped)
                inv_sqrt_ev = 1.0 / sqrt_ev
                cov_sqrt = (eigvecs * sqrt_ev.unsqueeze(0)) @ eigvecs.t().contiguous()
                cov_inv_sqrt = (eigvecs * inv_sqrt_ev.unsqueeze(0)) @ eigvecs.t().contiguous()

                W_scaled = W @ cov_sqrt
                U_svd, S_svd, Vh_svd = torch.linalg.svd(W_scaled, full_matrices=False)

                sqrt_S = torch.sqrt(torch.clamp(S_svd[:target_rank], min=1e-12))
                U_r = U_svd[:, :target_rank] * sqrt_S.unsqueeze(0)
                V_r = sqrt_S.unsqueeze(1) * (Vh_svd[:target_rank, :] @ cov_inv_sqrt)

                norm_u = torch.norm(U_r, p="fro").clamp_min(1e-8)
                norm_v = torch.norm(V_r, p="fro").clamp_min(1e-8)
                scale = torch.sqrt(norm_u / norm_v)
                U_r = U_r / scale
                V_r = V_r * scale
            except Exception as e:
                logger.warning("Activation-aware SVD failed (%s); using weight SVD.", e)
                U_svd, S_svd, Vh_svd = torch.linalg.svd(W, full_matrices=False)
                sqrt_S = torch.sqrt(torch.clamp(S_svd[:target_rank], min=1e-12))
                U_r = U_svd[:, :target_rank] * sqrt_S.unsqueeze(0)
                V_r = sqrt_S.unsqueeze(1) * Vh_svd[:target_rank, :]
        else:
            U_svd, S_svd, Vh_svd = torch.linalg.svd(W, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S_svd[:target_rank], min=1e-12))
            U_r = U_svd[:, :target_rank] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vh_svd[:target_rank, :]

        new_layer.U.data.copy_(U_r.to(dtype=target_mod.weight.dtype))
        new_layer.V.data.copy_(V_r.to(dtype=target_mod.weight.dtype))
        if target_mod.bias is not None and new_layer.bias is not None:
            new_layer.bias.data.copy_(target_mod.bias.data)
        return new_layer

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, bias={self.bias is not None}")


# ===========================================================================
# Compressor (with extensive debug logging)
# ===========================================================================

class DynamicEntropySVDCompressor(BaseCompressor):
    """Dynamic Entropy SVD Compressor with debug diagnostics."""

    def __init__(
        self,
        model: nn.Module,
        min_rank: int = 4,
        max_rank: int = 128,
        target_param_ratio: Optional[float] = None,
        total_rank_budget: Optional[int] = None,
        target_rank: Optional[int] = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        calibration_loader: Any = None,
        calibration_texts: Optional[List[str]] = None,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        max_calibration_batches: int = 8,
        max_calibration_tokens: int = 32768,
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="dynamic_entropy_svd", **kwargs)
        cfg_obj = cfg if cfg is not None else kwargs.get("comp_config", None)
        self.cfg = cfg_obj

        min_rank = _cfg_get(cfg_obj, "dynamic_min_rank", _cfg_get(cfg_obj, "min_rank", min_rank))
        max_rank = _cfg_get(cfg_obj, "dynamic_max_rank", _cfg_get(cfg_obj, "max_rank", max_rank))
        target_rank = _cfg_get(cfg_obj, "dynamic_rank", _cfg_get(cfg_obj, "rank", target_rank))
        total_rank_budget = _cfg_get(cfg_obj, "total_rank_budget", total_rank_budget)
        target_param_ratio = _cfg_get(cfg_obj, "target_param_ratio", target_param_ratio)
        max_error = _cfg_get(cfg_obj, "dynamic_max_error", _cfg_get(cfg_obj, "max_error", max_error))
        target_attention_only = _cfg_get(
            cfg_obj, "dynamic_target_attention_only",
            _cfg_get(cfg_obj, "target_attention_only", target_attention_only),
        )
        fallback_to_original = _cfg_get(
            cfg_obj, "dynamic_fallback_to_original",
            _cfg_get(cfg_obj, "fallback_to_original", fallback_to_original),
        )
        max_calibration_batches = _cfg_get(cfg_obj, "max_calibration_batches", max_calibration_batches)
        max_calibration_tokens = _cfg_get(cfg_obj, "max_calibration_tokens", max_calibration_tokens)
        min_layer_dim = _cfg_get(cfg_obj, "min_layer_dim", min_layer_dim)

        # DE-FIX-13: allow bypassing should_compress_linear when debugging.
        skip_filter = _cfg_get(cfg_obj, "skip_should_compress_filter", skip_should_compress_filter)

        self.min_rank = max(1, int(min_rank))
        self.max_rank = max(self.min_rank, int(max_rank))
        if "rank" in kwargs and kwargs["rank"] is not None:
            target_rank = int(kwargs["rank"])
            self.min_rank = max(1, min(self.min_rank, target_rank))
            self.max_rank = max(self.max_rank, target_rank)

        self.target_param_ratio = float(target_param_ratio) if target_param_ratio is not None else None
        self.total_rank_budget = int(total_rank_budget) if total_rank_budget is not None else None
        self.target_rank = int(target_rank) if target_rank is not None else None
        self.max_error = float(max_error)
        self.target_attention_only = bool(target_attention_only)
        self.fallback_to_original = bool(fallback_to_original)
        self.max_calibration_batches = max(1, int(max_calibration_batches))
        self.max_calibration_tokens = max(1, int(max_calibration_tokens))
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.skip_should_compress_filter = bool(skip_filter)

        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_modules = target_modules
        self.tokenizer = (
            tokenizer
            or getattr(model, "tokenizer", None)
            or getattr(getattr(model, "config", None), "tokenizer", None)
        )

        self.layer_entropies: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.layer_errors: Dict[str, float] = {}
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # DE-FIX-11: debug-enriched target check
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()

        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            if _debug:
                logger.info(f"  [skip] {name}: contains lora/base_layer part")
            return False
        if any(skip in name_lower for skip in
               ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            if _debug:
                logger.info(f"  [skip] {name}: matches skip keyword")
            return False

        target_mod = getattr(module, "base_layer", module)
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = _is_conv1d(target_mod)
        if not (is_linear or is_conv1d):
            if _debug:
                logger.info(f"  [skip] {name}: not Linear/Conv1D ({target_mod.__class__.__name__})")
            return False
        if isinstance(target_mod, DynamicSVDLinear) or isinstance(module, DynamicSVDLinear):
            if _debug:
                logger.info(f"  [skip] {name}: already DynamicSVDLinear")
            return False

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
        elif _debug:
            logger.info(f"  [BYPASS] {name}: skip_should_compress_filter=True")

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight is None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if self.target_attention_only:
            last = name.split(".")[-1]
            is_attn = (last in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                if _debug:
                    logger.info(f"  [skip] {name}: target_attention_only=True and not attn")
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
    # Calibration
    # ------------------------------------------------------------------
    def collect_activation_covariances(self) -> Dict[str, torch.Tensor]:
        accum: Dict[str, Dict[str, Any]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook_fn(mod, inp, out):
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                if x.ndim < 2:
                    return
                x_flat = x.reshape(-1, x.shape[-1]).float()
                if x_flat.shape[0] > self.max_calibration_tokens:
                    x_flat = x_flat[: self.max_calibration_tokens]
                n = x_flat.shape[0]
                if n == 0:
                    return
                cov_local = (x_flat.t().to(torch.float64) @ x_flat.to(torch.float64)).cpu()
                if layer_name not in accum:
                    accum[layer_name] = {"cov": cov_local, "n": n}
                else:
                    accum[layer_name]["cov"] += cov_local
                    accum[layer_name]["n"] += n
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
            hooks.append(target_mod.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters()).device
        self.model.eval()
        try:
            with torch.no_grad():
                self._run_calibration(device)
        except Exception as e:
            logger.warning("Calibration forward raised: %s", e)
        finally:
            for h in hooks:
                h.remove()

        normalized = {}
        for name, d in accum.items():
            n = max(1, d["n"])
            normalized[name] = (d["cov"] / n).float()
        return normalized

    def _run_calibration(self, device: torch.device) -> None:
        if self.calibration_loader is not None:
            count = 0
            for batch in self.calibration_loader:
                if count >= self.max_calibration_batches:
                    break
                try:
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
                        self.model(*batch)
                    elif isinstance(batch, torch.Tensor):
                        self.model(batch.to(device))
                    count += 1
                except Exception as e:
                    logger.debug("Calibration batch %d failed: %s", count, e)
            logger.info("DynamicEntropySVD: ran %d calibration batches.", count)
            return

        if self.calibration_texts and len(self.calibration_texts) > 0 and self.tokenizer is not None:
            count = 0
            for text in self.calibration_texts:
                if count >= self.max_calibration_batches:
                    break
                try:
                    enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    self.model(**enc)
                    count += 1
                except Exception as e:
                    logger.debug("Text calibration exception: %s", e)
                    break
            logger.info("DynamicEntropySVD: ran %d text calibration passes.", count)
            return

        # Fallback: dummy inputs
        logger.info("DynamicEntropySVD: using dummy calibration inputs.")
        dummy = torch.randint(0, 1000, (2, 32), device=device)
        try:
            self.model(dummy)
        except Exception:
            try:
                self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
            except Exception as e:
                logger.debug("Dummy calibration failed: %s", e)

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------
    def compute_layer_entropies(self, covariances: Dict[str, torch.Tensor]) -> Dict[str, float]:
        entropies: Dict[str, float] = {}
        for name, module in self.model.named_modules():
            if not self._is_target_module(name, module):
                continue
            if name in covariances:
                ent = compute_singular_entropy(covariances[name])
            else:
                W = _canonical_weight(module)
                if W is None:
                    continue
                ent = compute_singular_entropy(W)
            entropies[name] = ent
        return entropies

    # ------------------------------------------------------------------
    # Main compress (with DE-FIX-14 diagnostic)
    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}

        # 1. Covariances
        covariances = self.collect_activation_covariances()
        logger.info("DynamicEntropySVD: %d activation covariances collected.", len(covariances))

        # 2. Entropies
        self.layer_entropies = self.compute_layer_entropies(covariances)
        if not self.layer_entropies:
            # DE-FIX-14: diagnostic breakdown
            logger.warning("DynamicEntropySVD: no compressible layers found. Running diagnostic...")
            total_named = 0
            total_targets = 0
            for name, module in self.model.named_modules():
                total_named += 1
                if self._is_target_module(name, module, _debug=True):
                    total_targets += 1
            logger.warning(
                "DynamicEntropySVD diagnostic: %d named modules scanned, %d targets found.",
                total_named, total_targets,
            )
            return self.model, self.get_compression_stats()

        logger.info("DynamicEntropySVD: %d target layers identified.", len(self.layer_entropies))

        # 3. Layer dims
        layer_dims: Dict[str, Tuple[int, int]] = {}
        for name, module in self.model.named_modules():
            if name not in self.layer_entropies:
                continue
            W = _canonical_weight(module)
            if W is None:
                continue
            layer_dims[name] = (W.shape[0], W.shape[1])

        # 4. Freeze non-targets if attention_only
        if self.target_attention_only:
            for name, module in list(self.model.named_modules()):
                target_mod = getattr(module, "base_layer", module)
                is_lin = isinstance(target_mod, nn.Linear) or _is_conv1d(target_mod)
                if is_lin and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        # 5. Budget
        total_rank_budget = self.total_rank_budget
        if total_rank_budget is None and self.target_rank is not None:
            total_rank_budget = self.target_rank * len(self.layer_entropies)

        # 6. Allocate
        self.layer_ranks = allocate_entropy_ranks(
            entropies=self.layer_entropies,
            min_rank=self.min_rank,
            max_rank=self.max_rank,
            total_rank_budget=total_rank_budget,
            target_param_ratio=self.target_param_ratio,
            layer_dims=layer_dims,
        )
        logger.info("DynamicEntropySVD: allocation done for %d layers.", len(self.layer_ranks))

        # 7. Replace
        total_orig = 0
        total_comp = 0
        for name, rank in self.layer_ranks.items():
            try:
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                module = getattr(parent, child_name)
                target_mod = getattr(module, "base_layer", module)

                out_f, in_f = layer_dims[name]
                orig_params = out_f * in_f
                total_orig += orig_params

                cov = covariances.get(name, None)
                factorized = DynamicSVDLinear.from_linear(module, rank=rank, covariance=cov)

                W = _canonical_weight(module)
                if W is None:
                    continue
                W = W.float()
                W_recon = factorized.U.float() @ factorized.V.float()
                norm_w = torch.norm(W, p="fro").item()
                err = float(torch.norm(W - W_recon, p="fro").item() / max(norm_w, 1e-12))
                self.layer_errors[name] = err

                if err > self.max_error and self.fallback_to_original:
                    logger.warning("DynamicEntropySVD: error %.4f on %s > %.4f; keeping original.",
                                   err, name, self.max_error)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp += orig_params
                    continue

                if hasattr(module, "base_layer"):
                    module.base_layer = factorized
                elif hasattr(module, "linear"):
                    module.linear = factorized
                else:
                    replace_module(self.model, name, factorized)

                for p_name, p in module.named_parameters():
                    if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                        p.requires_grad = True

                self.compressed_layers.append(name)
                total_comp += rank * (in_f + out_f)
            except Exception as e:
                logger.error("DynamicEntropySVD: failed to replace %s: %s", name, e)
                continue

        logger.info("DynamicEntropySVD: %d layers compressed, %d fallback.",
                    len(self.compressed_layers), len(self.fallback_layers))

        # 8. Forward sanity
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                try:
                    self.model(dummy)
                except Exception:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
            forward_check_success = True
        except Exception as e:
            logger.debug("Post-compression forward check warning: %s", e)

        effective_ratio = total_comp / total_orig if total_orig > 0 else 1.0
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        stats = self.get_compression_stats()
        stats.update({
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
        })
        self.stats = stats
        return self.model, stats


# ===========================================================================
# Registry aliases
# ===========================================================================

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
def dynamic_entropy_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    try:
        min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "dynamic_min_rank",
                                                   _cfg_get(comp_config, "min_rank", 4)))
        max_rank = kwargs.pop("max_rank", _cfg_get(comp_config, "dynamic_max_rank",
                                                   _cfg_get(comp_config, "max_rank", 128)))
        target_rank = kwargs.pop("rank", _cfg_get(comp_config, "dynamic_rank",
                                                  _cfg_get(comp_config, "rank", None)))
        max_error = kwargs.pop("max_error", _cfg_get(comp_config, "dynamic_max_error",
                                                     _cfg_get(comp_config, "max_error", 0.35)))
        target_attention_only = kwargs.pop(
            "target_attention_only",
            _cfg_get(comp_config, "dynamic_target_attention_only",
                     _cfg_get(comp_config, "target_attention_only", False)),
        )
        fallback = kwargs.pop("fallback_to_original",
                              _cfg_get(comp_config, "dynamic_fallback_to_original",
                                       _cfg_get(comp_config, "fallback_to_original", True)))
        target_param_ratio = kwargs.pop("target_param_ratio",
                                        _cfg_get(comp_config, "target_param_ratio", None))
        total_rank_budget = kwargs.pop("total_rank_budget",
                                       _cfg_get(comp_config, "total_rank_budget", None))
        max_calibration_batches = kwargs.pop(
            "max_calibration_batches",
            _cfg_get(comp_config, "max_calibration_batches", 8),
        )
        skip_filter = kwargs.pop("skip_should_compress_filter",
                                 _cfg_get(comp_config, "skip_should_compress_filter", False))

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
            max_calibration_batches=max_calibration_batches,
            skip_should_compress_filter=skip_filter,
            cfg=comp_config,
            **kwargs,
        )
        compressed_model, stats = compressor.compress()
        if return_stats:
            return compressed_model, stats
        return compressed_model
    except Exception as e:
        logger.error("dynamic_entropy_svd_factory failed: %s", e, exc_info=True)
        return (model, {}) if return_stats else model


@COMPRESSOR_REGISTRY.register("inject_dynamic_entropy")
@COMPRESSOR_REGISTRY.register("dynamic_entropy_inject")
def inject_dynamic_entropy(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    target = layer if layer is not None else model
    if target is None:
        return None
    try:
        res = dynamic_entropy_svd_factory(target, **kwargs)
        return res[0] if isinstance(res, tuple) else res
    except Exception as e:
        logger.error("inject_dynamic_entropy failed: %s", e, exc_info=True)
        return None


@COMPRESSOR_REGISTRY.register("inject_dynamic_rank")
@COMPRESSOR_REGISTRY.register("dynamic_rank_inject")
def inject_dynamic_rank(model: Any = None, layer: Any = None, **kwargs: Any) -> Any:
    return inject_dynamic_entropy(model, layer, **kwargs)


dynamic_entropy_inject = inject_dynamic_entropy
dynamic_rank_inject = inject_dynamic_rank
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