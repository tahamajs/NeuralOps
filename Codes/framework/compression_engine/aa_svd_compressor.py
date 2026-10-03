"""
Activation-Aware SVD (AASVD) Compressor — Production-Hardened (v2).

Streaming EMA activation covariance accumulation:
    C_X^(t) = β·C_X^(t-1) + (1-β)·(X^T X) / B
Projects weights onto principal activation subspaces and replaces dense
nn.Linear/Conv1D layers with structural low-rank SVDLinear modules:
    y = (x · V^T) · U^T + LoRA(x) + b

"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import get_dynamic_compression_rank, should_compress_linear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default multi-domain calibration corpus
# ---------------------------------------------------------------------------
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _autocast_off(x: torch.Tensor) -> contextlib.AbstractContextManager:
    """AASVD-FIX-05: robust autocast-disable context."""
    dev_type = x.device.type
    if hasattr(torch, "autocast") and dev_type in ("cuda", "cpu"):
        try:
            return torch.autocast(device_type=dev_type, enabled=False)
        except Exception:
            pass
    return contextlib.nullcontext()


# ===========================================================================
# SVDLinear
# ===========================================================================
class SVDLinear(nn.Module):
    """
    Low-rank factorized linear layer:
        y = (x @ V.T) @ U.T + LoRA(x) * scaling + b
    Parameter count: r·(d_in + d_out) + LoRA params.
    """

    # AASVD-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        use_fused_kernel: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_r: int | None = None,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(rank, self.in_features, self.out_features))
        self.use_fused_kernel = bool(use_fused_kernel)

        # Accept both lora_rank and legacy lora_r
        eff_lora = (
            int(lora_r)
            if (lora_r is not None and lora_r > 0)
            else max(0, int(lora_rank))
        )
        self.lora_rank = eff_lora
        self.lora_r = eff_lora
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (
            (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        )
        self.scaling = self.lora_scaling

        requires_grad = not (self.lora_rank > 0 or freeze_base)
        fk = {"dtype": dtype, "device": device}

        self.V = nn.Parameter(
            torch.empty(self.rank, self.in_features, **fk), requires_grad=requires_grad
        )
        self.U = nn.Parameter(
            torch.empty(self.out_features, self.rank, **fk), requires_grad=requires_grad
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, **fk), requires_grad=requires_grad
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, **fk)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.lora_rank, **fk)
            )
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # AASVD-FIX-07: input mean is not tracked; remove the buffer.
        self._input_mean: torch.Tensor | None = None

        self.reset_parameters()

    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        return self.get_weight()

    def get_weight(self) -> torch.Tensor:
        w = torch.matmul(self.U, self.V)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    @property
    def use_lora(self) -> bool:
        return bool(
            self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None
        )

    def freeze_base_weights(self) -> None:
        if self.V.requires_grad:
            self.V.requires_grad = False
        if self.U.requires_grad:
            self.U.requires_grad = False
        if self.bias is not None and self.bias.requires_grad:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for param in (self.V, self.U):
                if param.dtype in (torch.float16, torch.bfloat16):
                    tmp = torch.empty_like(param, dtype=torch.float32, device="cpu")
                    _orthogonal_init(tmp)
                    param.copy_(tmp.to(dtype=param.dtype, device=param.device))
                else:
                    _orthogonal_init(param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            if (
                self.lora_rank > 0
                and self.lora_A is not None
                and self.lora_B is not None
            ):
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._input_mean is not None:
            x = x - self._input_mean.to(x.device, x.dtype)

        if self.use_fused_kernel and (self.lora_rank == 0 or self.lora_A is None):
            try:
                from framework.attentions.fused_svd_kernel import fused_svd_linear

                return fused_svd_linear(
                    x=x, V=self.V, U=self.U, bias=self.bias, use_rms_norm=False
                )
            except Exception as e:
                logger.debug("SVDLinear fused kernel failed (%s); using F.linear.", e)

        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            b = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            lora_latent = F.linear(x, a)
            out = out + F.linear(lora_latent, b) * self.lora_scaling

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def set_low_rank_factors(
        self, U_factor: torch.Tensor, V_factor: torch.Tensor
    ) -> None:
        with torch.no_grad():
            r = min(self.rank, U_factor.size(1), V_factor.size(0))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U_factor[:, :r].to(self.U.device, self.U.dtype))
            self.V.data[:r, :].copy_(V_factor[:r, :].to(self.V.device, self.V.dtype))


def _orthogonal_init(t: torch.Tensor) -> None:
    r, c = t.shape
    if r == c:
        nn.init.orthogonal_(t)
        return
    a = torch.randn(max(r, c), min(r, c), device=t.device, dtype=t.dtype)
    q, _ = torch.linalg.qr(a)
    if r >= c:
        t.copy_(q[:r, :])
    else:
        t.copy_(q[:, :c].t())


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("asvd")
@COMPRESSOR_REGISTRY.register("asvd_compressor")
@COMPRESSOR_REGISTRY.register("aasvd")
@COMPRESSOR_REGISTRY.register("aa_svd")
@COMPRESSOR_REGISTRY.register("aa_svd_compressor")
class AASVDCompressor(BaseCompressor):
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        rank: int = 288,
        ema_beta: float = 0.9,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        target_modules: list[str] | None = None,
        target_attention_only: bool = True,
        force_rank: bool = False,
        lora_rank: int = 0,  # AASVD-FIX-03
        lora_alpha: float = 1.0,  # AASVD-FIX-03
        cov_dtype: str = "float32",  # AASVD-FIX-09
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="aasvd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        # AASVD-FIX-02
        self.comp_config = cfg

        # AASVD-FIX-04: use _cfg_get everywhere
        rank = _cfg_get(cfg, "aasvd_rank", _cfg_get(cfg, "rank", rank))
        ema_beta = float(
            _cfg_get(cfg, "aasvd_ema_beta", _cfg_get(cfg, "ema_beta", ema_beta))
        )
        tokenizer = _cfg_get(cfg, "tokenizer", tokenizer)
        calibration_texts = _cfg_get(cfg, "calibration_texts", calibration_texts)
        max_error = float(
            _cfg_get(cfg, "aasvd_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        fallback_to_original = bool(
            _cfg_get(
                cfg,
                "aasvd_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        target_attention_only = bool(
            _cfg_get(
                cfg,
                "aasvd_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        lora_rank = int(
            _cfg_get(cfg, "aasvd_lora_rank", _cfg_get(cfg, "lora_rank", lora_rank))
        )
        lora_alpha = float(
            _cfg_get(cfg, "aasvd_lora_alpha", _cfg_get(cfg, "lora_alpha", lora_alpha))
        )
        cov_dtype = str(_cfg_get(cfg, "cov_dtype", cov_dtype))

        # override from kwargs (highest priority after explicit args)
        lora_rank = int(kwargs.get("lora_rank", lora_rank))
        lora_alpha = float(kwargs.get("lora_alpha", lora_alpha))
        cov_dtype = str(kwargs.get("cov_dtype", cov_dtype))

        # force_rank resolution
        force_rank = bool(
            force_rank
            or kwargs.get("force_rank", False)
            or (cfg is not None and _cfg_get(cfg, "force_rank", False))
            or getattr(getattr(model, "config", None), "force_rank", False)
        )
        self.force_rank = force_rank
        self.use_strs = bool(kwargs.get("use_strs", not force_rank)) and not force_rank
        # AASVD-FIX-10: dynamic STRS candidates are computed per-layer

        # ---- tokenizer auto-resolve ----
        if tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None) or getattr(
                getattr(model, "config", None), "tokenizer", None
            )
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            mp = getattr(getattr(model, "config", None), "_name_or_path", None)
            if mp:
                try:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(mp)
                    if getattr(self.tokenizer, "pad_token", None) is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception as e:
                    logger.debug("Auto-tokenizer load failed for %s: %s", mp, e)
                    self.tokenizer = None

        # ---- basic params ----
        self.rank = max(1, int(rank))
        self.ema_beta = float(max(0.0, min(ema_beta, 0.999)))
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.calibration_texts = calibration_texts or DEFAULT_CALIBRATION_CORPUS
        self.target_modules = target_modules
        self.lora_rank = max(0, int(lora_rank))
        self.lora_alpha = float(lora_alpha)
        # AASVD-FIX-03
        self.lora_r = self.lora_rank  # legacy alias

        # AASVD-FIX-09
        self.cov_dtype_name = str(cov_dtype).lower()
        if self.cov_dtype_name not in ("float32", "float16"):
            self.cov_dtype_name = "float32"
        self.cov_dtype = (
            torch.float32 if self.cov_dtype_name == "float32" else torch.float16
        )

        # AASVD-FIX-08: single source of truth for device
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )

        # ---- state ----
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}

        # AASVD-FIX-14
        self._compressed_params: int = 0
        try:
            self._original_params = self._count_params(self.model)
        except Exception:
            self._original_params = sum(p.numel() for p in self.model.parameters())

        # AASVD-FIX-11
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # AASVD-FIX-13: skip already compressed
        if isinstance(module, SVDLinear):
            return False
        if any(
            name.endswith(sfx)
            for sfx in (".base_layer", ".lora_A", ".lora_B", ".lora_dropout")
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
            for skip in (
                "lm_head",
                "classifier",
                "head",
                "output_layer",
                "score",
                "pooler",
                "embedding",
                "layernorm",
                "rmsnorm",
            )
        ):
            return False

        if self.target_attention_only:
            ffn_patterns = (
                "ffn",
                "mlp",
                "lin1",
                "lin2",
                "intermediate",
                "gate_proj",
                "up_proj",
                "down_proj",
                "c_fc",
            )
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = (
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
            )
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            try:
                if not should_compress_linear(self.model, name):
                    return False
            except Exception:
                pass

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    # AASVD-FIX-10: dynamic STRS candidates
    # ------------------------------------------------------------------
    @staticmethod
    def _build_strs_candidates(r_floor: int, r_ceiling: int) -> list[int]:
        if r_ceiling < r_floor:
            return [r_ceiling]
        cands = set()
        n = max(4, min(10, int(math.log2(max(2, r_ceiling // max(1, r_floor)))) + 2))
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            r = int(round(r_floor * (r_ceiling / max(1, r_floor)) ** t))
            cands.add(max(r_floor, min(r, r_ceiling)))
        cands.add(r_floor)
        cands.add(r_ceiling)
        return sorted(cands)

    # ------------------------------------------------------------------
    def compress_linear(
        self,
        layer: nn.Module,
        rank: int | None = None,
        cov: torch.Tensor | None = None,
        cov_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
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
            weight_tensor = getattr(
                target_mod, "weight", getattr(layer, "weight", None)
            )
            if weight_tensor is None:
                return None, float("inf"), 0

            raw_w = weight_tensor.data
            if is_conv1d:
                in_feat, out_feat = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()
            else:
                out_feat, in_feat = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            r_default = rank if rank is not None else self.rank
            r = get_dynamic_compression_rank(
                in_feat,
                out_feat,
                model_or_family=self.model,
                default_rank=r_default,
                comp_config=self.comp_config,
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

            # ---- covariance preconditioning ----
            cov_sqrt = cov_inv_sqrt = None
            if cov_factors is not None:
                cov_sqrt, cov_inv_sqrt = cov_factors
            elif cov is not None and cov.shape == (in_feat, in_feat):
                try:
                    cov_fp32 = cov.to(self.device, torch.float32)
                    cov_sym = 0.5 * (cov_fp32 + cov_fp32.T)
                    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
                    eps = 1e-6 * torch.max(eigvals).clamp(min=1e-12) + 1e-8
                    eigvals_reg = eigvals.clamp(min=eps)
                    cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_reg)) @ eigvecs.T
                    cov_inv_sqrt = (
                        eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_reg)) @ eigvecs.T
                    )
                except Exception as e:
                    logger.debug("AASVD covariance sqrt failed for %s: %s", e, e)
                    cov_sqrt, cov_inv_sqrt = None, None

            W_weighted = W @ cov_sqrt if cov_sqrt is not None else W

            W_w_dev = W_weighted.to(self.device, torch.float32)
            U, S, Vt = torch.linalg.svd(W_w_dev, full_matrices=False)

            # ---- STRS / fixed rank ----
            if not self.use_strs:
                selected_r = min(r, len(S))
                sqrt_S = torch.sqrt(torch.clamp(S[:selected_r], min=1e-12))
                best_Ur = U[:, :selected_r] * sqrt_S.unsqueeze(0)
                best_Vr = sqrt_S.unsqueeze(1) * Vt[:selected_r, :]
                if cov_inv_sqrt is not None:
                    best_Vr = best_Vr @ cov_inv_sqrt
                best_err = float(
                    torch.norm(W - best_Ur @ best_Vr, p="fro").item()
                    / max(W_norm, 1e-12)
                )
            else:
                r_floor = max(1, min(r, max_allowed_rank, len(S)))
                r_ceiling = max_allowed_rank
                cand_ranks = self._build_strs_candidates(r_floor, r_ceiling)
                cand_ranks = [c for c in cand_ranks if c <= len(S)]

                best_err = float("inf")
                selected_r = cand_ranks[0] if cand_ranks else r_floor
                best_Ur = best_Vr = None

                for test_r in cand_ranks:
                    sqrt_S = torch.sqrt(torch.clamp(S[:test_r], min=1e-12))
                    U_cand = U[:, :test_r] * sqrt_S.unsqueeze(0)
                    V_cand = sqrt_S.unsqueeze(1) * Vt[:test_r, :]
                    if cov_inv_sqrt is not None:
                        V_cand = V_cand @ cov_inv_sqrt
                    W_cand = U_cand @ V_cand
                    test_err = float(
                        torch.norm(W - W_cand, p="fro").item() / max(W_norm, 1e-12)
                    )
                    if test_err < best_err:
                        best_err = test_err
                        selected_r = test_r
                        best_Ur = U_cand
                        best_Vr = V_cand
                    if test_err <= self.max_error:
                        break

                if best_Ur is None or best_err > self.max_error:
                    # Fallback to break-even rank
                    be_r = max(
                        1,
                        min(
                            int(0.75 * (in_feat * out_feat) / (in_feat + out_feat)),
                            len(S),
                        ),
                    )
                    selected_r = be_r
                    sqrt_S = torch.sqrt(torch.clamp(S[:be_r], min=1e-12))
                    best_Ur = U[:, :be_r] * sqrt_S.unsqueeze(0)
                    best_Vr = sqrt_S.unsqueeze(1) * Vt[:be_r, :]
                    if cov_inv_sqrt is not None:
                        best_Vr = best_Vr @ cov_inv_sqrt
                    best_err = float(
                        torch.norm(W - best_Ur @ best_Vr, p="fro").item()
                        / max(W_norm, 1e-12)
                    )

            dtype = weight_tensor.dtype
            device = weight_tensor.device
            bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))

            svd_layer = SVDLinear(
                in_features=in_feat,
                out_features=out_feat,
                rank=selected_r,
                bias=bias_tensor is not None,
                dtype=dtype,
                device=device,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            svd_layer.set_low_rank_factors(
                best_Ur.to(device=device, dtype=dtype),
                best_Vr.to(device=device, dtype=dtype),
            )
            if bias_tensor is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(bias_tensor.data.to(svd_layer.bias.dtype))

            svd_layer.reconstruction_error = best_err
            svd_layer.rank = selected_r
            return svd_layer, best_err, selected_r
        except Exception as e:
            logger.warning("AASVD factorization failed: %s", e)
            return None, float("inf"), 0

    # ------------------------------------------------------------------
    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        self.model.eval()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        covariances: dict[str, torch.Tensor] = {}
        # AASVD-FIX-06: track per-layer update count
        update_counts: dict[str, int] = {}

        if self.tokenizer is None:
            logger.info("AASVD: no tokenizer; using unweighted SVD.")
        else:
            hooks = []

            def make_hook(layer_name: str):
                nonlocal covariances, update_counts

                def hook(module, inp, outp):
                    if not inp or inp[0] is None:
                        return
                    x = inp[0].detach()
                    # AASVD-FIX-05
                    with _autocast_off(x):
                        target_mod = getattr(module, "base_layer", module)
                        cls_name = module.__class__.__name__.lower()
                        target_cls = target_mod.__class__.__name__.lower()
                        is_conv1d = (
                            "conv1d" in cls_name
                            or "conv1d" in target_cls
                            or hasattr(module, "nf")
                            or hasattr(target_mod, "nf")
                        )
                        wt = getattr(
                            target_mod, "weight", getattr(module, "weight", None)
                        )
                        if wt is None:
                            return
                        in_f = wt.shape[0] if is_conv1d else wt.shape[1]

                        x_fp32 = x.float()
                        x_flat = x_fp32.reshape(-1, in_f)
                        # AASVD-FIX-12
                        valid = x_flat.abs().amax(dim=-1) > 0
                        if valid.sum() == 0:
                            return
                        x_active = x_flat[valid]
                        n_tokens = max(1, x_active.size(0))
                        batch_cov = (x_active.t() @ x_active / n_tokens).cpu()
                        if self.cov_dtype == torch.float16:
                            batch_cov = batch_cov.half()

                        # AASVD-FIX-06: EMA with first-observation handling
                        if layer_name not in covariances:
                            covariances[layer_name] = batch_cov
                            update_counts[layer_name] = 1
                        else:
                            beta = self.ema_beta
                            covariances[layer_name] = (
                                beta * covariances[layer_name]
                                + (1.0 - beta) * batch_cov
                            )
                            update_counts[layer_name] += 1

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
                inputs = {
                    k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")
                }
                with torch.no_grad():
                    self.model(**inputs)
            except Exception as e:
                logger.debug("AASVD calibration forward failed: %s", e)
            finally:
                for h in hooks:
                    h.remove()

        # ---- Precompute covariance square roots ----
        cov_factor_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, cov in covariances.items():
            try:
                cov_dev = cov.to(self.device, torch.float32)
                cov_sym = 0.5 * (cov_dev + cov_dev.T)
                eigvals, eigvecs = torch.linalg.eigh(cov_sym)
                eps = 1e-6 * torch.max(eigvals).clamp(min=1e-12) + 1e-8
                eigvals_reg = eigvals.clamp(min=eps)
                cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_reg)) @ eigvecs.T
                cov_inv_sqrt = (
                    eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_reg)) @ eigvecs.T
                )
                cov_factor_cache[name] = (cov_sqrt, cov_inv_sqrt)
            except Exception as e:
                logger.debug("AASVD covariance factor failed for %s: %s", name, e)

        # ---- Compress each target ----
        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            try:
                for p in target_mod.parameters():
                    self._orig_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

            cov_factors = cov_factor_cache.get(name)
            cov = covariances.get(name) if cov_factors is None else None

            svd_layer, err, r = self.compress_linear(
                module,
                rank=self.rank,
                cov=cov,
                cov_factors=cov_factors,
            )
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            too_big = (err > self.max_error) and not self.use_strs
            if svd_layer is None or too_big:
                if self.fallback_to_original:
                    logger.warning(
                        "AASVD: %s (err=%.4f > %.4f); falling back to original.",
                        name,
                        err,
                        self.max_error,
                    )
                    self.fallback_layers.append(name)
                    self._comp_storage_bytes += (
                        self._orig_storage_bytes
                    )  # approximation; corrected below
                    continue
                logger.warning(
                    "AASVD: %s exceeds max_error but fallback is False.",
                    name,
                )
                if svd_layer is None:
                    self.fallback_layers.append(name)
                    self._comp_storage_bytes += self._orig_storage_bytes
                    continue

            if hasattr(module, "base_layer"):
                module.base_layer = svd_layer
                for attr in ("lora_A", "lora_B"):
                    sub = getattr(module, attr, None)
                    if sub is None:
                        continue
                    params = (
                        sub.parameters() if hasattr(sub, "parameters") else iter([sub])
                    )
                    for p in params:
                        if hasattr(p, "requires_grad"):
                            p.requires_grad = True
            else:
                replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)

            try:
                for p in svd_layer.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

        # Freeze fallback layers
        for name, mod in self.model.named_modules():
            if name in self.fallback_layers:
                target = getattr(mod, "base_layer", mod)
                for p in target.parameters():
                    p.requires_grad = False

        # ---- Free memory ----
        covariances.clear()
        cov_factor_cache.clear()
        del covariances
        del cov_factor_cache
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )

        # AASVD-FIX-11: proper storage accounting
        # comp_storage was approximated for fallback layers; recompute honestly.
        comp_bytes_real = 0
        for name, mod in self.model.named_modules():
            if name in self.compressed_layers or name in self.fallback_layers:
                target = getattr(mod, "base_layer", mod)
                try:
                    for p in target.parameters():
                        comp_bytes_real += p.numel() * p.element_size()
                except Exception:
                    pass
        self._comp_storage_bytes = comp_bytes_real

        storage_ratio = (
            self._orig_storage_bytes / self._comp_storage_bytes
            if self._comp_storage_bytes > 0
            else 1.0
        )
        effective_ratio = (
            self._comp_storage_bytes / self._orig_storage_bytes
            if self._orig_storage_bytes > 0
            else 1.0
        )

        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "use_strs": self.use_strs,
                "rank": self.rank,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "orig_storage_bytes": self._orig_storage_bytes,
                "comp_storage_bytes": self._comp_storage_bytes,
                "storage_compression_ratio": round(storage_ratio, 3),
                "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
                "method": "aasvd_activation_aware_svd",
            }
        )
        logger.info(
            "AASVD: %d compressed, %d fallback, avg_err=%.4f, storage_ratio=%.2fx",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            storage_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("asvd_factory")
@COMPRESSOR_REGISTRY.register("asvd_compressor_factory")
@COMPRESSOR_REGISTRY.register("aasvd_factory")
@COMPRESSOR_REGISTRY.register("aa_svd_factory")
@COMPRESSOR_REGISTRY.register("aa_svd_compressor_factory")
def aasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    rank = kwargs.pop(
        "rank", _cfg_get(comp_config, "aasvd_rank", _cfg_get(comp_config, "rank", 64))
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "aasvd_max_error", _cfg_get(comp_config, "max_error", 0.20)
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "aasvd_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = AASVDCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


aa_svd_compressor_factory = aasvd_factory
aa_svd_factory = aasvd_factory
aasvd_compressor_factory = aasvd_factory

__all__ = [
    "AASVDCompressor",
    "SVDLinear",
    "aasvd_factory",
    "aa_svd_factory",
    "aa_svd_compressor_factory",
    "aasvd_compressor_factory",
    "DEFAULT_CALIBRATION_CORPUS",
]
