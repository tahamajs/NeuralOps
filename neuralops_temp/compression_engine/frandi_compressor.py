"""
FRanDI: Feature-Regression Aligned SVD Compressor — Production-Hardened (v2).

Registered as 'frandi' and 'frandi_compressor'.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.pruning_utils import adaptive_max_error
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
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
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "out_proj",
    "dense",
}


# ===========================================================================
# FRanDILinear
# ===========================================================================
class FRanDILinear(nn.Module):
    """
    Feature-Regression Aligned Factorized Linear Layer with optional LoRA.

    W ≈ B @ A
      - A (rank, in_features): input feature-aligned basis
      - B (out_features, rank): output expansion matrix

    Forward:
      y = (x @ A^T) @ B^T + bias
          + (alpha/r) * (x @ lora_A^T @ lora_B^T)
    """

    # FR-FIX-09: NOT an attention module.
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / lora_r) if lora_r > 0 else 1.0

        requires_grad_base = not (self.lora_r > 0)

        # FR-FIX-02: init in fp32 for numerical stability, cast to dtype.
        A_init = torch.empty(self.rank, in_features, dtype=torch.float32, device=device)
        nn.init.orthogonal_(A_init)
        self.A = nn.Parameter(A_init.to(dtype=dtype), requires_grad=requires_grad_base)

        self.B = nn.Parameter(
            torch.empty(out_features, self.rank, dtype=dtype, device=device),
            requires_grad=requires_grad_base,
        )
        nn.init.xavier_uniform_(self.B.float())
        with torch.no_grad():
            self.B.data = self.B.data.to(dtype=dtype) if dtype is not None else self.B.data

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=requires_grad_base,
            )
        else:
            self.register_parameter("bias", None)

        # FR-FIX-03: LoRA in the same dtype as base.
        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=dtype, device=device)
            )
            with torch.no_grad():
                tmp = self.lora_A.data.float()
                nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
                self.lora_A.data.copy_(tmp.to(dtype=dtype) if dtype is not None else tmp)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # FR-FIX-15: weight cache
        self._cached_weight: torch.Tensor | None = None
        self._cached_version: int = -1

    # ---- cache management ----
    def _current_version(self) -> int:
        v = 0
        for p in (self.A, self.B):
            v += getattr(p, "_version", 0)
        if self.lora_A is not None:
            v += getattr(self.lora_A, "_version", 0)
        if self.lora_B is not None:
            v += getattr(self.lora_B, "_version", 0)
        return v

    @property
    def weight(self) -> torch.Tensor:
        # FR-FIX-15: cache in eval mode.
        if (
            not self.training
            and self._cached_weight is not None
            and self._cached_version == self._current_version()
        ):
            return self._cached_weight
        W = torch.matmul(self.B.float(), self.A.float()).to(self.A.dtype)
        if not self.training:
            self._cached_weight = W
            self._cached_version = self._current_version()
        return W

    def get_weight(self) -> torch.Tensor:
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_a = F.linear(x_flat, self.A)  # (N, rank)
        out = F.linear(x_a, self.B, self.bias)  # (N, out)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat, self.lora_A), self.lora_B)
            out = out + self.scaling * lora_out

        return out.view(*orig_shape[:-1], self.out_features)

    # FR-FIX-16: merge() for PEFT/HF compatibility
    @torch.no_grad()
    def merge(self) -> None:
        """Fold LoRA delta into base factors (approximate, additive)."""
        if self.lora_r <= 0 or self.lora_A is None or self.lora_B is None:
            return
        # Merge into B: B_new = B + (alpha/r) * (lora_B @ lora_A) @ pinv(A)
        # Simpler: absorb into B by solving B_new A = B A + delta_BA
        # We use a least-squares approach via pseudoinverse.
        A_f = self.A.float()
        delta = self.scaling * torch.matmul(self.lora_B.float(), self.lora_A.float())
        # Solve for B_new such that B_new @ A = B @ A + delta:
        # B_new = B + delta @ A^+
        A_pinv = torch.linalg.pinv(A_f)
        B_new = self.B.float() + delta @ A_pinv
        self.B.data.copy_(B_new.to(self.B.dtype))
        # Reset LoRA to zero
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
        self._cached_weight = None

    def merge_and_unload(self) -> FRanDILinear:
        self.merge()
        return self

    def unmerge(self) -> None:
        # Not implemented (would require tracking original B).
        raise NotImplementedError("FRanDILinear.unmerge is not supported.")

    # ------------------------------------------------------------------
    @staticmethod
    def _select_rank(
        S: torch.Tensor,
        rank: int = 8,
        energy_threshold: float | None = None,
    ) -> int:
        # FR-FIX-11: cap at `rank` even when energy threshold is used.
        max_rank = min(int(rank), len(S))
        if energy_threshold is None:
            return max(1, max_rank)
        total_energy = (S**2).sum().clamp_min(1e-12)
        cum_energy = (S**2).cumsum(dim=0) / total_energy
        idx = torch.where(cum_energy >= energy_threshold)[0]
        if len(idx) > 0:
            return max(1, min(int(idx[0].item()) + 1, max_rank))
        return max(1, max_rank)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear | Any,
        rank: int = 8,
        energy_threshold: float | None = None,
        cov_matrix: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> FRanDILinear:
        target_mod = getattr(linear, "base_layer", linear)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
            W_float = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape
            W_float = raw_w.float()

        bias = (
            target_mod.bias.data
            if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None
            else None
        )

        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)

        B_data: torch.Tensor | None = None
        A_data: torch.Tensor | None = None
        r: int = 1

        # ---- Whitening path ----
        if cov_matrix is not None and cov_matrix.shape == (in_f, in_f):
            try:
                cov_f = cov_matrix.float()
                trace_val = torch.diagonal(cov_f).sum().item()
                reg = max(1e-6, 1e-4 * trace_val / max(in_f, 1))

                # FR-FIX-05: add regularization without materializing eye.
                cov_reg = cov_f.clone()
                idx = torch.arange(in_f, device=cov_reg.device)
                cov_reg[idx, idx] += reg

                L = torch.linalg.cholesky(cov_reg)  # (in, in) lower triangular
                W_tilde = torch.matmul(W_float, L)  # (out, in)

                U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
                r = cls._select_rank(S, rank=rank, energy_threshold=energy_threshold)
                r = max(1, min(r, out_f, in_f))

                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                B_data = U[:, :r] * sqrt_S.unsqueeze(0)  # (out, r)
                A_tilde = sqrt_S.unsqueeze(1) * Vt[:r, :]  # (r, in)

                # FR-FIX-10: solve for A such that A @ L = A_tilde.
                # Transposed: L^T @ A^T = A_tilde^T. L^T is upper triangular.
                A_solved = torch.linalg.solve_triangular(
                    L.t().contiguous(),
                    A_tilde.t().contiguous(),
                    upper=True,
                )
                A_data = A_solved.t().contiguous()  # (r, in)
            except Exception as e:
                logger.debug("FRanDI whitening failed: %s; falling back to plain SVD", e)
                B_data = None
                A_data = None

        # ---- Plain SVD path ----
        if B_data is None or A_data is None:
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
            r = cls._select_rank(S, rank=rank, energy_threshold=energy_threshold)
            r = max(1, min(r, out_f, in_f))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            B_data = U[:, :r] * sqrt_S.unsqueeze(0)
            A_data = sqrt_S.unsqueeze(1) * Vt[:r, :]

        # Determine effective LoRA rank
        effective_lora_r = int(lora_r)
        if existing_lora_A is not None and hasattr(existing_lora_A, "weight"):
            try:
                effective_lora_r = int(existing_lora_A.weight.shape[0])
            except (AttributeError, IndexError, TypeError):
                pass

        frandi_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            lora_r=effective_lora_r,
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )

        with torch.no_grad():
            frandi_layer.A.data.copy_(A_data[:r, :].to(dtype=raw_w.dtype, device=raw_w.device))
            frandi_layer.B.data.copy_(B_data[:, :r].to(dtype=raw_w.dtype, device=raw_w.device))

            if bias is not None and frandi_layer.bias is not None:
                frandi_layer.bias.data.copy_(bias.to(dtype=raw_w.dtype, device=raw_w.device))

            if (
                existing_lora_A is not None
                and existing_lora_B is not None
                and frandi_layer.lora_A is not None
            ):
                w_A = getattr(existing_lora_A, "weight", existing_lora_A)
                w_B = getattr(existing_lora_B, "weight", existing_lora_B)
                if hasattr(w_A, "data") and hasattr(w_B, "data"):
                    try:
                        if w_A.shape == frandi_layer.lora_A.shape:
                            frandi_layer.lora_A.data.copy_(w_A.data)
                        if w_B.shape == frandi_layer.lora_B.shape:
                            frandi_layer.lora_B.data.copy_(w_B.data)
                        frandi_layer.lora_A.requires_grad = True
                        frandi_layer.lora_B.requires_grad = True
                    except Exception as e:
                        logger.debug("FRanDI LoRA copy failed: %s", e)

        return frandi_layer


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("frandi_compressor_class")
class FRanDICompressor(BaseCompressor):
    """
    FRanDI (Feature-Regression aligned SVD) Compressor.
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int = 8,
        energy_threshold: float | None = None,
        calibration_loader: Any | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.5,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = False,
        cfg: Any | None = None,
        target_layers: list[str] | None = None,
        num_calibration_batches: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="frandi", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.rank = int(_cfg_get(cfg, "frandi_rank", _cfg_get(cfg, "rank", rank)))
        self.energy_threshold = _cfg_get(
            cfg,
            "frandi_energy_threshold",
            _cfg_get(cfg, "energy_threshold", energy_threshold),
        )
        self.calibration_loader = calibration_loader or _cfg_get(cfg, "calibration_loader", None)
        self.lora_r = int(_cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        c_max_error = _cfg_get(cfg, "frandi_max_error", _cfg_get(cfg, "max_error", max_error))
        if c_max_error is None or c_max_error == 0.5:
            c_max_error = adaptive_max_error(self.rank, 768, 768, "distilbert")

        self.max_error = float(c_max_error)
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "frandi_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "frandi_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.skip_attention_qkv = bool(_cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))

        # FR-FIX-12: treat empty list as None.
        tl = target_layers or _cfg_get(cfg, "target_layers", None)
        if tl is not None and len(tl) == 0:
            tl = None
        self.target_layers = tl

        self.num_calibration_batches = max(
            1, int(_cfg_get(cfg, "num_calibration_batches", num_calibration_batches))
        )

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        # FR-FIX-17: GQA guard always active.
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is None or num_kv_heads is None:
            return False
        if num_kv_heads >= num_heads:
            return False
        name_lower = name.lower()
        return any(
            p in name_lower
            for p in ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn")
        )

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
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

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        if self._is_gqa_asymmetric(name, module):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d) or isinstance(target_mod, FRanDILinear):
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    # ------------------------------------------------------------------
    # FR-FIX-04: bounded calibration loop, FR-FIX-14: fp32 accumulation
    # ------------------------------------------------------------------
    def _collect_covariances(self) -> dict[str, torch.Tensor]:
        if self.calibration_loader is None or not hasattr(self.model, "named_modules"):
            return {}

        covariances: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                    x = inputs[0].detach().float()
                    target_mod = getattr(module, "base_layer", module)
                    cls_name = target_mod.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                    in_f = target_mod.weight.shape[0] if is_conv1d else target_mod.weight.shape[1]
                    if x.shape[-1] != in_f:
                        return
                    x_flat = x.reshape(-1, in_f)
                    # FR-FIX-14: accumulate in fp32
                    cov = torch.matmul(x_flat.t().contiguous(), x_flat).float()
                    if layer_name not in covariances:
                        covariances[layer_name] = cov
                        counts[layer_name] = x_flat.size(0)
                    else:
                        covariances[layer_name] = covariances[layer_name] + cov
                        counts[layer_name] += x_flat.size(0)

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters(), torch.tensor(0.0)).device
        self.model.eval()
        try:
            with torch.no_grad():
                n_seen = 0
                for batch in self.calibration_loader:
                    if n_seen >= self.num_calibration_batches:
                        break
                    try:
                        if isinstance(batch, dict):
                            b = {
                                k: v.to(device)
                                for k, v in batch.items()
                                if isinstance(v, torch.Tensor)
                            }
                            _ = self.model(**b)
                        elif isinstance(batch, (list, tuple)):
                            _ = self.model(batch[0].to(device))
                        elif isinstance(batch, torch.Tensor):
                            _ = self.model(batch.to(device))
                        n_seen += 1
                    except Exception as e:
                        logger.debug("FRanDI calibration batch failed: %s", e)
        finally:
            for h in hooks:
                h.remove()

        # Average
        for name in list(covariances.keys()):
            covariances[name] = covariances[name] / max(1, counts[name])
            covariances[name] = covariances[name].cpu()

        return covariances

    # ------------------------------------------------------------------
    def compress(self, model: nn.Module | None = None) -> tuple[nn.Module, dict[str, Any]]:
        from neuralops.core.model_utils import replace_module

        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(p.numel() for p in self.model.parameters())

        covs = self._collect_covariances()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set = set()

        total_orig_params = 0
        total_comp_params = 0

        # FR-FIX-13: threshold scales with requested rank.
        min_rank_threshold = max(4, self.rank // 2)

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data
            W_target = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            out_f, in_f = W_target.shape
            orig_params = in_f * out_f
            total_orig_params += orig_params

            max_rank = min(in_f, out_f)
            bound_r = (in_f * out_f) // max(1, in_f + out_f)
            # FR-FIX-13
            if max_rank < min_rank_threshold or bound_r < max(1, self.rank // 4):
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params
                continue

            cov = covs.get(name, None)
            try:
                frandi_layer = FRanDILinear.from_linear(
                    module,
                    rank=self.rank,
                    energy_threshold=self.energy_threshold,
                    cov_matrix=cov,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_comp = frandi_layer.get_weight().float()
                W_norm = torch.norm(W_target, p="fro").clamp_min(1e-8)
                err = float(torch.norm(W_target - W_comp, p="fro").item() / W_norm)
                self.layer_errors[name] = err

                if err > self.max_error:
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        total_comp_params += orig_params
                        continue

                # Preserve wrapper if present (e.g. LoRALinear)
                if hasattr(module, "base_layer"):
                    module.base_layer = frandi_layer
                else:
                    replace_module(self.model, name, frandi_layer)

                self.compressed_layers.append(name)
                actual_rank = int(getattr(frandi_layer, "rank", self.rank))
                comp_params = actual_rank * (in_f + out_f)
                if getattr(frandi_layer, "lora_r", 0) > 0:
                    comp_params += int(frandi_layer.lora_r) * (in_f + out_f)
                total_comp_params += comp_params
            except Exception as e:
                logger.debug("FRanDI error on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params

        # FR-FIX-07: forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                attn = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        first_linear = next(
                            (
                                m
                                for m in self.model.modules()
                                if isinstance(m, (nn.Linear, FRanDILinear))
                            ),
                            None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        dt = first_p.dtype if first_p is not None else torch.float32
                        dummy_float = torch.randn(1, 8, in_dim, device=dev, dtype=dt)
                        self.model(dummy_float)
        except Exception as e:
            logger.debug("FRanDI forward sanity check failed: %s", e)
            forward_check_success = False

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (
            sum(self.layer_errors.values()) / len(self.layer_errors) if self.layer_errors else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "frandi_rank": self.rank,
                "energy_threshold": self.energy_threshold,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "max_error": self.max_error,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
                "method": "frandi",
            }
        )
        logger.info(
            "Compressor frandi finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("frandi")
@COMPRESSOR_REGISTRY.register("frandi_compressor")
@COMPRESSOR_REGISTRY.register("frandi_factory")
@COMPRESSOR_REGISTRY.register("frandi_compressor_factory")
def frandi_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Factory function for FRanDI compressor."""
    effective_rank = rank or _cfg_get(comp_config, "frandi_rank", _cfg_get(comp_config, "rank", 8))
    energy = _cfg_get(
        comp_config,
        "frandi_energy_threshold",
        _cfg_get(comp_config, "energy_threshold", None),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "frandi_max_error", _cfg_get(comp_config, "max_error", 0.5)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "frandi_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "frandi_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = FRanDICompressor(
        model,
        rank=effective_rank,
        energy_threshold=energy,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
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


__all__ = ["FRanDILinear", "FRanDICompressor", "frandi_factory"]
