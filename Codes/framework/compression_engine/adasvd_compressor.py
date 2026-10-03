"""
AdaSVD: Adaptive SVD with Activation Whitening, Two-Sided Weighted ALS,
and Mean-Shift Bias Absorption.

Registered as 'adasvd', 'adasvd_compressor', 'adasvd_factory',
'adasvd_compressor_factory'.

Key components:
  1. Mean-Shift Bias Absorption:
       b_new = b + W @ mu_x
       y = A B (x - mu_x) + b_new
  2. Two-Sided Weighted ALS:
       minimize || D^{1/2} (W - AB) C^{1/2} ||_F^2
       with C = C_X and D = diag(Var(Y)).
  3. Information-Theoretic Effective Rank:
       erank = exp(-sum(p_i * log p_i)), p_i = s_i^2 / sum(s_j^2)
  4. Fused Q/K/V splitting for GPT-2 c_attn (Conv1D).
"""

from __future__ import annotations

import logging
import math
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    try:
        from transformers.modeling_utils import Conv1D
    except ImportError:
        Conv1D = None

logger = logging.getLogger(__name__)


# ============================================================================
# Default calibration corpus
# ============================================================================
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


# ============================================================================
# Fused Q/K/V SVD Linear (GPT-2 c_attn replacement)
# ============================================================================
class FusedQKVSVDLinear(nn.Module):
    """
    Compressed replacement for GPT-2's fused c_attn.

    Stores three independent low-rank factors (Q, K, V). Forward concatenates
    them so the output shape matches the original fused projection.

    Supports mean-shift bias absorption: if `_input_mean` is set, the input
    is centered before the low-rank projection and the shift is added back
    through the bias.
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if out_features % 3 != 0:
            raise ValueError(
                f"FusedQKVSVDLinear requires out_features divisible by 3, got {out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.split_size = out_features // 3
        self.rank = int(rank)
        # Conv1D compatible
        self.nf = out_features
        self.nx = in_features
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        # Q, K, V factor pairs
        self.q_A = nn.Parameter(
            torch.empty(self.split_size, self.rank, dtype=dtype, device=device)
        )
        self.q_B = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        self.k_A = nn.Parameter(
            torch.empty(self.split_size, self.rank, dtype=dtype, device=device)
        )
        self.k_B = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        self.v_A = nn.Parameter(
            torch.empty(self.split_size, self.rank, dtype=dtype, device=device)
        )
        self.v_B = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )

        for A, B in ((self.q_A, self.q_B), (self.k_A, self.k_B), (self.v_A, self.v_B)):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_r, in_features, dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, self.lora_r, dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Mean-shift buffer (persistent so it survives checkpoint save/load)
        self.register_buffer(
            "_input_mean",
            torch.zeros(in_features, dtype=torch.float32, device=device),
            persistent=True,
        )
        self._has_input_mean = False

    # ------------------------------------------------------------------
    def set_fused_factors(self, W_fused: torch.Tensor) -> None:
        """Split a fused (in, out) or (out, in) weight into Q/K/V factors."""
        if W_fused.shape == (self.in_features, self.out_features):
            W = W_fused.t().contiguous().float()
        elif W_fused.shape == (self.out_features, self.in_features):
            W = W_fused.contiguous().float()
        else:
            raise ValueError(
                f"Unexpected fused weight shape {tuple(W_fused.shape)}; "
                f"expected ({self.in_features}, {self.out_features}) or "
                f"({self.out_features}, {self.in_features})"
            )

        S = self.split_size
        splits = [
            (W[:S, :], self.q_A, self.q_B),
            (W[S : 2 * S, :], self.k_A, self.k_B),
            (W[2 * S :, :], self.v_A, self.v_B),
        ]
        for W_part, A_param, B_param in splits:
            U, Svals, Vt = torch.linalg.svd(W_part, full_matrices=False)
            r = min(self.rank, Svals.shape[0])
            sqrt_S = torch.sqrt(torch.clamp(Svals[:r], min=1e-12))
            A_new = U[:, :r] * sqrt_S.unsqueeze(0)
            B_new = sqrt_S.unsqueeze(1) * Vt[:r, :]
            with torch.no_grad():
                A_param.data.zero_()
                B_param.data.zero_()
                A_param.data[:, :r].copy_(
                    A_new.to(device=A_param.device, dtype=A_param.dtype)
                )
                B_param.data[:r, :].copy_(
                    B_new.to(device=B_param.device, dtype=B_param.dtype)
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _forward_factor(
        x: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        h = F.linear(x.to(B.dtype), B)
        return F.linear(h, A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Mean-shift centering
        if self._has_input_mean:
            mu = self._input_mean.to(x.device, x.dtype)
            x_c = x - mu
        else:
            x_c = x

        q = self._forward_factor(x_c, self.q_A, self.q_B)
        k = self._forward_factor(x_c, self.k_A, self.k_B)
        v = self._forward_factor(x_c, self.v_A, self.v_B)
        out = torch.cat([q, k, v], dim=-1)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype)
            b = self.lora_B.to(x.dtype)
            lora_out = F.linear(F.linear(x, a), b) * self.scaling
            out = out + lora_out

        return out.to(orig_dtype)

    @torch.no_grad()
    def set_input_mean(self, mu: torch.Tensor) -> None:
        """Register the calibration activation mean (input centering)."""
        mu_f = mu.detach().float().to(self._input_mean.device)
        if mu_f.shape[0] != self.in_features:
            raise ValueError(
                f"input_mean shape {tuple(mu_f.shape)} != ({self.in_features},)"
            )
        self._input_mean.copy_(mu_f)
        self._has_input_mean = True

    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the fused (in, out) Conv1D weight for compatibility."""
        W_q = self.q_A.float() @ self.q_B.float()
        W_k = self.k_A.float() @ self.k_B.float()
        W_v = self.v_A.float() @ self.v_B.float()
        W_linear = torch.cat([W_q, W_k, W_v], dim=0)
        return W_linear.t().contiguous()

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        self.set_fused_factors(value)


# ============================================================================
# AdaSVD Compressor
# ============================================================================
@COMPRESSOR_REGISTRY.register("adasvd")
@COMPRESSOR_REGISTRY.register("adasvd_compressor")
class AdaSVDCompressor(BaseCompressor):
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int | None = None,
        target_param_ratio: float = 0.7,
        num_iters: int = 10,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = True,
        min_rank: int = 16,
        target_modules: list[str] | None = None,
        tokenizer: Any | None = None,
        calibration_texts: list[str] | None = None,
        use_weighted_als: bool = False,
        use_mean_shift: bool = True,
        use_gating_d: bool = True,
        device: str | torch.device | None = None,
        comp_config: Any = None,
        max_calib_tokens: int = 4096,
        fused_qkv_error_threshold: float = 0.30,
        gelu_layer_patterns: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adasvd", **kwargs)
        cfg = comp_config or kwargs.get(
            "comp_config", kwargs.get("cfg", getattr(model, "config", None))
        )

        # ---- CLI overrides ----
        cli_rank = None
        cli_use_gating_d = None
        cli_use_weighted_als = None
        for idx, arg in enumerate(sys.argv):
            if arg in ("--rank", "--adasvd_rank") and idx + 1 < len(sys.argv):
                try:
                    cli_rank = int(sys.argv[idx + 1])
                except ValueError:
                    pass
            elif arg.startswith(("--rank=", "--adasvd_rank=")):
                try:
                    cli_rank = int(arg.split("=")[1])
                except ValueError:
                    pass
            elif arg in ("--adasvd_use_gating_d", "--use_gating_d") and idx + 1 < len(
                sys.argv
            ):
                cli_use_gating_d = sys.argv[idx + 1].lower() in ("true", "1", "yes")
            elif arg.startswith(("--adasvd_use_gating_d=", "--use_gating_d=")):
                cli_use_gating_d = arg.split("=")[1].lower() in ("true", "1", "yes")
            elif arg in (
                "--adasvd_use_weighted_als",
                "--use_weighted_als",
            ) and idx + 1 < len(sys.argv):
                cli_use_weighted_als = sys.argv[idx + 1].lower() in ("true", "1", "yes")
            elif arg.startswith(("--adasvd_use_weighted_als=", "--use_weighted_als=")):
                cli_use_weighted_als = arg.split("=")[1].lower() in ("true", "1", "yes")

        self.rank = (
            cli_rank
            if cli_rank is not None
            else _cfg_get(cfg, "adasvd_rank", _cfg_get(cfg, "rank", rank))
        )
        self.target_param_ratio = float(
            _cfg_get(
                cfg,
                "adasvd_target_ratio",
                _cfg_get(cfg, "target_param_ratio", target_param_ratio),
            )
        )
        self.num_iters = int(
            _cfg_get(cfg, "adasvd_iters", _cfg_get(cfg, "num_iters", num_iters))
        )
        self.max_error = float(
            _cfg_get(cfg, "adasvd_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "adasvd_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "adasvd_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.min_rank = int(
            _cfg_get(cfg, "adasvd_min_rank", _cfg_get(cfg, "min_rank", min_rank))
        )

        self.use_weighted_als = (
            cli_use_weighted_als
            if cli_use_weighted_als is not None
            else bool(
                _cfg_get(
                    cfg,
                    "adasvd_use_weighted_als",
                    _cfg_get(cfg, "use_weighted_als", use_weighted_als),
                )
            )
        )
        self.use_mean_shift = bool(
            _cfg_get(
                cfg,
                "adasvd_use_mean_shift",
                _cfg_get(cfg, "use_mean_shift", use_mean_shift),
            )
        )
        self.use_gating_d = (
            cli_use_gating_d
            if cli_use_gating_d is not None
            else bool(
                _cfg_get(
                    cfg,
                    "adasvd_use_gating_d",
                    _cfg_get(cfg, "use_gating_d", use_gating_d),
                )
            )
        )

        self.target_modules = target_modules
        self.max_calib_tokens = int(max_calib_tokens)
        self.fused_qkv_error_threshold = float(
            _cfg_get(cfg, "adasvd_fused_qkv_error_threshold", fused_qkv_error_threshold)
        )

        if gelu_layer_patterns is None:
            gelu_layer_patterns = (
                "lin1",
                "intermediate.dense",
                "gate_proj",
                "up_proj",
                "dense_h_to_4h",
                "c_fc",
                "wi_0",
                "wi_1",
            )
        self.gelu_layer_patterns = tuple(gelu_layer_patterns)

        # Tokenizer auto-resolve
        if tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None) or getattr(
                getattr(model, "config", None), "tokenizer", None
            )
        if tokenizer is None:
            mp = getattr(getattr(model, "config", None), "_name_or_path", None)
            if mp:
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(mp)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as e:
                    logger.debug("Could not auto-load tokenizer from %s: %s", mp, e)
        self.tokenizer = tokenizer

        self.calibration_texts = (
            calibration_texts
            or _cfg_get(cfg, "calibration_texts", None)
            or DEFAULT_CALIBRATION_CORPUS
        )
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.comp_config = cfg

        # State
        self.calibration_activations: dict[str, torch.Tensor] = {}
        self.layer_means: dict[str, torch.Tensor] = {}
        self.layer_out_means: dict[str, torch.Tensor] = {}
        self.layer_out_vars: dict[str, torch.Tensor] = {}
        self.mean_shifts: dict[str, torch.Tensor] = {}

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.layer_iters: dict[str, int] = {}

        self.calibration_time_s: float = 0.0

        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0
        self._compressed_params: int = 0
        self._compressed_names_seen: set = set()

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, (SVDLinear, FusedQKVSVDLinear)):
            return False
        if any(
            name.endswith(sfx)
            for sfx in (".base_layer", ".lora_A", ".lora_B", ".lora_dropout")
        ):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = module.__class__.__name__.lower()
        target_mod.__class__.__name__.lower()

        weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
        if (
            weight_tensor is None
            or not hasattr(weight_tensor, "dim")
            or weight_tensor.dim() != 2
        ):
            return False

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (
                Conv1D is not None
                and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D))
            )
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
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "out_proj",
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

    @staticmethod
    def _is_fused_qkv_layer(name: str, module: nn.Module) -> bool:
        if "c_attn" not in name.lower():
            return False
        target_mod = getattr(module, "base_layer", module)
        nf = getattr(module, "nf", None) or getattr(target_mod, "nf", None)
        if nf is None or nf % 3 != 0:
            return False
        w = getattr(target_mod, "weight", None)
        return w is not None and w.dim() == 2

    @staticmethod
    def _gelu_derivative_squared(y: torch.Tensor) -> torch.Tensor:
        phi = 0.5 * (1.0 + torch.erf(y / math.sqrt(2.0)))
        pdf = torch.exp(-y.pow(2) / 2.0) / math.sqrt(2.0 * math.pi)
        gelu_prime = phi + y * pdf
        return gelu_prime.pow(2)

    def _is_followed_by_gelu(self, layer_name: str) -> bool:
        name_lower = layer_name.lower()
        return any(p in name_lower for p in self.gelu_layer_patterns)

    @staticmethod
    def _compute_effective_rank(W: torch.Tensor) -> float:
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            s_sq = s.pow(2)
            denom = s_sq.sum().clamp(min=1e-12)
            p = s_sq / denom
            p = p[p > 1e-12]
            return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())

    _compute_nuclear_norm = _compute_effective_rank

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _collect_activations(
        self, calibration_data: Any | None = None
    ) -> dict[str, torch.Tensor]:
        activations: dict[str, list[torch.Tensor]] = {}
        out_activations: dict[str, list[torch.Tensor]] = {}
        hooks = []
        max_tokens = self.max_calib_tokens

        def make_hook(layer_name: str):
            def hook(module, inp, outp):
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                target_mod = getattr(module, "base_layer", module)
                is_conv1d = (
                    (
                        Conv1D is not None
                        and (
                            isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)
                        )
                    )
                    or hasattr(module, "nf")
                    or hasattr(target_mod, "nf")
                )
                weight_tensor = getattr(
                    target_mod, "weight", getattr(module, "weight", None)
                )
                if weight_tensor is None or weight_tensor.dim() != 2:
                    return

                in_f = weight_tensor.shape[0] if is_conv1d else weight_tensor.shape[1]
                x_flat = x.float().reshape(-1, in_f)
                mask = x_flat.abs().sum(dim=-1) > 1e-5
                x_active = x_flat[mask] if mask.sum() > 0 else x_flat

                bucket = activations.setdefault(layer_name, [])
                curr_n = sum(t.size(0) for t in bucket)
                if curr_n < max_tokens:
                    avail = max_tokens - curr_n
                    bucket.append(x_active[:avail].cpu())

                if isinstance(outp, torch.Tensor):
                    out_f = (
                        weight_tensor.shape[1] if is_conv1d else weight_tensor.shape[0]
                    )
                    y_flat = outp.detach().float().reshape(-1, out_f)
                    y_bucket = out_activations.setdefault(layer_name, [])
                    curr_ny = sum(t.size(0) for t in y_bucket)
                    if curr_ny < max_tokens:
                        avail_y = max_tokens - curr_ny
                        y_bucket.append(y_flat[:avail_y].cpu())

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        if not hooks:
            return {}

        self.model.eval()
        dev = next(self.model.parameters()).device

        try:
            with torch.no_grad():
                if calibration_data is not None:
                    self._run_calibration(calibration_data, dev)
                elif self.tokenizer is not None and self.calibration_texts:
                    inputs = self.tokenizer(
                        self.calibration_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    self.model(**inputs)
                else:
                    gen = torch.Generator(device="cpu").manual_seed(42)
                    dummy_ids = torch.randint(100, 2000, (8, 64), generator=gen).to(dev)
                    try:
                        self.model(input_ids=dummy_ids)
                    except Exception:
                        try:
                            self.model(dummy_ids)
                        except Exception:
                            try:
                                dummy_x = torch.randn(128, 64, device=dev)
                                self.model(dummy_x)
                            except Exception as e:
                                logger.debug(
                                    "AdaSVD synthetic calibration failed: %s", e
                                )
        except Exception as e:
            logger.debug("AdaSVD calibration pass exception: %s", e)
        finally:
            for h in hooks:
                h.remove()

        result: dict[str, torch.Tensor] = {}
        for name, tensor_list in activations.items():
            if tensor_list:
                cat_x = torch.cat(tensor_list, dim=0)
                result[name] = cat_x
                self.layer_means[name] = cat_x.mean(dim=0).cpu()

        for name, y_list in out_activations.items():
            if y_list:
                cat_y = torch.cat(y_list, dim=0)
                self.layer_out_means[name] = cat_y.mean(dim=0).cpu()

                if self.use_gating_d and self._is_followed_by_gelu(name):
                    gelu_sq = self._gelu_derivative_squared(cat_y)
                    d_values = gelu_sq.mean(dim=0)
                else:
                    d_values = cat_y.var(dim=0, unbiased=False)

                self.layer_out_vars[name] = d_values.cpu()

        return result

    def _run_calibration(self, calibration_data: Any, dev: torch.device) -> None:
        if isinstance(calibration_data, torch.Tensor):
            try:
                self.model(calibration_data.to(dev))
            except Exception:
                self.model(input_ids=calibration_data.to(dev))
        elif isinstance(calibration_data, (list, tuple)) and self.tokenizer is not None:
            inputs = self.tokenizer(
                list(calibration_data),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            self.model(**inputs)
        elif hasattr(calibration_data, "__iter__"):
            for batch in calibration_data:
                try:
                    if isinstance(batch, torch.Tensor):
                        self.model(batch.to(dev))
                    elif isinstance(batch, dict):
                        b_dev = {
                            k: v.to(dev)
                            for k, v in batch.items()
                            if isinstance(v, torch.Tensor)
                        }
                        self.model(**b_dev)
                    else:
                        continue
                except Exception as e:
                    logger.debug("AdaSVD calibration batch skipped: %s", e)
                    continue

    # ------------------------------------------------------------------
    # SVD helpers
    # ------------------------------------------------------------------
    def _init_whitened_svd(
        self,
        W: torch.Tensor,
        r: int,
        C_X: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_out, d_in = W.shape
        device = W.device
        dtype = W.dtype

        if C_X is not None and C_X.shape == (d_in, d_in):
            C_dev = C_X.to(device=device, dtype=torch.float32)
            eps = max(1e-8, float(1e-6 * torch.trace(C_dev).item() / max(1, d_in)))
            eye = torch.eye(d_in, device=device, dtype=torch.float32)
            S_white = None
            try:
                S_white = torch.linalg.cholesky(C_dev + eps * eye)
            except Exception:
                for exp in (-5, -4, -3, -2, -1):
                    try:
                        reg = 10**exp * (torch.trace(C_dev).item() / max(1, d_in))
                        S_white = torch.linalg.cholesky(C_dev + reg * eye)
                        break
                    except Exception:
                        continue

            if S_white is not None:
                try:
                    S_inv = torch.linalg.inv(S_white)
                    W_white = W.float() @ S_white
                    U, S, Vt = torch.linalg.svd(W_white, full_matrices=False)
                    Vt = Vt @ S_inv
                except Exception:
                    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
            else:
                U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        else:
            U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)

        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A = U[:, :r] * sqrt_S.unsqueeze(0)
        B = sqrt_S.unsqueeze(1) * Vt[:r, :]
        return A.to(dtype=dtype, device=device), B.to(dtype=dtype, device=device)

    def _two_sided_als(
        self,
        W: torch.Tensor,
        r: int,
        C_X: torch.Tensor,
        D_var: torch.Tensor,
        X_calib: torch.Tensor | None = None,
        mu_x: torch.Tensor | None = None,
        num_iters: int = 10,
        patience: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        device = W.device
        dtype = W.dtype

        A, B = self._init_whitened_svd(W, r, C_X)
        A = A.float()
        B = B.float()

        E = C_X.float().to(device)
        D = torch.diag(torch.clamp(D_var.float().to(device), min=1e-6))
        WE = W.float() @ E
        DW = D @ W.float()

        def _err(recon):
            if X_calib is not None:
                X_eval = (X_calib - mu_x.to(device)) if mu_x is not None else X_calib
                WX_c = W.float() @ X_eval.float().t()
                reconX_c = recon @ X_eval.float().t()
                return float(
                    torch.norm(WX_c - reconX_c, p="fro").item()
                    / (torch.norm(WX_c, p="fro").item() + 1e-9)
                )
            diff = W.float() - recon
            num = math.sqrt(max(0.0, torch.trace(diff @ E @ diff.t()).item()))
            den = math.sqrt(
                max(1e-9, torch.trace(W.float() @ E @ W.float().t()).item())
            )
            return float(num / (den + 1e-9))

        best_A, best_B = A.clone(), B.clone()
        best_err = _err(A @ B)
        patience_counter = 0
        actual_iters = 0

        for _ in range(num_iters):
            BEBt = B @ E @ B.t()
            eps_B = max(1e-9, float(1e-6 * BEBt.trace().item() / max(1, r)))
            try:
                A_new = (
                    WE
                    @ B.t()
                    @ torch.linalg.inv(
                        BEBt + eps_B * torch.eye(r, device=device, dtype=BEBt.dtype)
                    )
                )
            except Exception:
                break

            AtDA = A_new.t() @ D @ A_new
            eps_A = max(1e-9, float(1e-6 * AtDA.trace().item() / max(1, r)))
            try:
                B_new = (
                    torch.linalg.inv(
                        AtDA + eps_A * torch.eye(r, device=device, dtype=AtDA.dtype)
                    )
                    @ A_new.t()
                    @ DW
                )
            except Exception:
                break

            if not torch.isfinite(A_new).all() or not torch.isfinite(B_new).all():
                break

            actual_iters += 1
            err = _err(A_new @ B_new)

            if err < best_err - 1e-6:
                best_A, best_B = A_new.clone(), B_new.clone()
                best_err = err
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            A, B = A_new, B_new

        return best_A.to(dtype), best_B.to(dtype), best_err, actual_iters

    # ------------------------------------------------------------------
    # Fused QKV compression
    # ------------------------------------------------------------------
    def _compress_fused_qkv(
        self,
        layer: nn.Module,
        rank: int,
        name: str,
    ) -> tuple[FusedQKVSVDLinear | None, float, int]:
        target_mod = getattr(layer, "base_layer", layer)
        w = getattr(target_mod, "weight", None)
        if w is None or w.dim() != 2:
            return None, float("inf"), 0

        raw_w = w.data
        in_f, out_f = raw_w.shape
        if out_f % 3 != 0:
            logger.warning(
                "FusedQKV %s: out_features %d not divisible by 3", name, out_f
            )
            return None, float("inf"), 0

        split_size = out_f // 3
        max_rank = min(in_f, split_size)
        r_floor = max(16, min(int(0.15 * split_size), self.min_rank * 4))
        r = max(r_floor, min(int(rank), max_rank))

        device = raw_w.device
        dtype = raw_w.dtype
        has_bias = getattr(target_mod, "bias", None) is not None

        fused_layer = FusedQKVSVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=has_bias,
            dtype=dtype,
            device=device,
            lora_r=0,
            lora_alpha=1.0,
        )

        try:
            fused_layer.set_fused_factors(raw_w)
        except Exception as e:
            logger.warning("FusedQKV %s factor split failed: %s", name, e)
            return None, float("inf"), 0

        # ---- Mean-shift ----
        mean_shift = None
        mu_x = None
        if self.use_mean_shift:
            mu_x = self.layer_means.get(name, None)
            if mu_x is not None:
                mu_x = mu_x.to(device=device, dtype=torch.float32)
                mean_shift = raw_w.float().t() @ mu_x
                self.mean_shifts[name] = mean_shift.cpu()
                fused_layer.set_input_mean(mu_x)

        if has_bias and fused_layer.bias is not None:
            with torch.no_grad():
                base_b = target_mod.bias.data.float().to(device)
                if mean_shift is not None:
                    base_b = base_b + mean_shift
                fused_layer.bias.data.copy_(base_b.to(dtype=dtype, device=device))

        # ---- Reconstruction error ----
        with torch.no_grad():
            W_recon = fused_layer.weight.to(device=device, dtype=torch.float32)
            W_orig = raw_w.float()
            norm_orig = torch.norm(W_orig, p="fro").clamp_min(1e-8)
            rel_err = (torch.norm(W_orig - W_recon, p="fro") / norm_orig).item()

        threshold = self.fused_qkv_error_threshold
        if r >= 0.5 * max_rank:
            threshold = max(threshold, 0.45)

        if rel_err > threshold:
            logger.warning(
                "AdaSVD fused-QKV %s: err=%.4f > threshold %.4f; "
                "falling back to original.",
                name,
                rel_err,
                threshold,
            )
            return None, rel_err, 0

        logger.info(
            "AdaSVD fused-QKV: %s in=%d out=%d split=%d rank=%d err=%.4f",
            name,
            in_f,
            out_f,
            split_size,
            r,
            rel_err,
        )
        return fused_layer, rel_err, 0

    # ------------------------------------------------------------------
    # Single-layer compression
    # ------------------------------------------------------------------
    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
        name: str = "",
        covariance: torch.Tensor | None = None,
        X_calib: torch.Tensor | None = None,
        out_variance: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[nn.Module | None, float, int]:
        if self._is_fused_qkv_layer(name, layer):
            return self._compress_fused_qkv(layer, rank, name)

        target_mod = getattr(layer, "base_layer", layer)
        is_conv1d = (
            (
                Conv1D is not None
                and (isinstance(layer, Conv1D) or isinstance(target_mod, Conv1D))
            )
            or hasattr(layer, "nf")
            or hasattr(target_mod, "nf")
        )

        weight_tensor = getattr(target_mod, "weight", getattr(layer, "weight", None))
        if weight_tensor is None or weight_tensor.dim() != 2:
            return None, float("inf"), 0

        raw_w = weight_tensor.data
        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()

        max_param_r = max(
            1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f)))
        )
        if self.rank is not None:
            r_floor = max(1, min(self.min_rank, rank))
        else:
            r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))
        r = max(r_floor, min(rank, max_param_r))

        device = weight_tensor.device
        dtype = weight_tensor.dtype

        # ---- Mean-shift ----
        mu_x = self.layer_means.get(name, None) if self.use_mean_shift else None
        mean_shift = None
        if mu_x is not None:
            mu_x = mu_x.to(device=device, dtype=torch.float32)
            mean_shift = (W.float() @ mu_x).to(dtype=torch.float32)
            self.mean_shifts[name] = mean_shift.cpu()

        X = (
            X_calib
            if X_calib is not None
            else self.calibration_activations.get(name, None)
        )
        X_centered = None
        C_X = None
        if X is not None:
            X = X.to(device=device, dtype=torch.float32)
            X_centered = (X - mu_x) if mu_x is not None else X
            C_X = (X_centered.t() @ X_centered) / max(1, X_centered.shape[0])
        elif covariance is not None and covariance.shape == (in_f, in_f):
            C_X = covariance.to(device=device, dtype=torch.float32)

        D_var = (
            out_variance
            if out_variance is not None
            else self.layer_out_vars.get(name, None)
        )

        if (
            C_X is not None
            and D_var is not None
            and self.use_weighted_als
            and self.num_iters > 0
        ):
            A, B, final_err, actual_iters = self._two_sided_als(
                W,
                r,
                C_X,
                D_var,
                X_calib=X,
                mu_x=mu_x,
                num_iters=self.num_iters,
            )
        else:
            A, B = self._init_whitened_svd(W, r, C_X)
            actual_iters = 0
            recon = A @ B
            if X is not None:
                X_eval = X_centered if X_centered is not None else X
                WX = W.float() @ X_eval.t()
                reconX = recon.float() @ X_eval.t()
                final_err = float(
                    torch.norm(WX - reconX, p="fro").item()
                    / (torch.norm(WX, p="fro").item() + 1e-9)
                )
            else:
                W_norm = torch.norm(W, p="fro").item()
                final_err = (
                    0.0
                    if W_norm < 1e-12
                    else float(torch.norm(W - recon, p="fro").item() / (W_norm + 1e-9))
                )

        bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))
        has_bias = (bias_tensor is not None) or (mean_shift is not None)

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=has_bias,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )

        if has_bias and svd_layer.bias is not None:
            if bias_tensor is not None:
                b_base = bias_tensor.data.float().to(device)
            else:
                b_base = torch.zeros(out_f, device=device, dtype=torch.float32)
            new_bias = b_base + mean_shift if mean_shift is not None else b_base
            svd_layer.bias.data.copy_(new_bias.to(dtype=dtype, device=device))

        # ---- Mean-shift registration on SVDLinear ----
        # SVDLinear exposes `_input_mean` (may be None). If it is a buffer,
        # we register and copy; otherwise we set attribute.
        if mu_x is not None:
            mu_f = mu_x.to(device=device, dtype=dtype)
            if hasattr(svd_layer, "register_buffer"):
                try:
                    # Only register if not already registered
                    if not any(
                        n == "_input_mean" for n, _ in svd_layer.named_buffers()
                    ):
                        svd_layer.register_buffer("_input_mean", mu_f, persistent=True)
                    else:
                        svd_layer._input_mean.copy_(mu_f)
                except Exception:
                    svd_layer._input_mean = mu_f
        # else: keep SVDLinear's default (None)

        return svd_layer, final_err, actual_iters

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(
        self, calibration_data: Any | None = None
    ) -> tuple[nn.Module, dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_iters = {}
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0
        self._compressed_names_seen = set()

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaSVD: no compressible layers found.")
            return self.model, self.get_compression_stats()

        # Track original bytes
        for name, mod in target_layers:
            target_mod = getattr(mod, "base_layer", mod)
            try:
                for p in target_mod.parameters():
                    self._orig_storage_bytes += p.numel() * p.element_size()
            except Exception:
                pass

        # ---- Step 1: calibration ----
        t_calib_start = time.monotonic()
        self.calibration_activations = self._collect_activations(
            calibration_data=calibration_data
        )
        self.calibration_time_s = time.monotonic() - t_calib_start
        logger.info("AdaSVD calibration time: %.2f s", self.calibration_time_s)

        # ---- Step 2: spectral importance ----
        layer_importances: dict[str, float] = {}
        total_importance = 0.0
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            weight_tensor = getattr(
                target_mod, "weight", getattr(module, "weight", None)
            )
            if weight_tensor is None or weight_tensor.dim() != 2:
                continue
            is_conv1d = (
                (
                    Conv1D is not None
                    and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D))
                )
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            raw_w = weight_tensor.data
            W_2d = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            imp = self._compute_effective_rank(W_2d)
            layer_importances[name] = max(imp, 1e-4)
            total_importance += layer_importances[name]

        # ---- Step 3: compress each layer ----
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            weight_tensor = getattr(
                target_mod, "weight", getattr(module, "weight", None)
            )
            if weight_tensor is None or weight_tensor.dim() != 2:
                continue
            is_conv1d = (
                (
                    Conv1D is not None
                    and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D))
                )
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            raw_w = weight_tensor.data
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            max_param_r = max(
                1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f)))
            )
            if self.rank is not None:
                r_floor = max(1, min(self.min_rank, self.rank))
                denom = max(1, len(target_layers))
                frac = layer_importances.get(name, 1.0) / max(
                    total_importance / denom, 1e-8
                )
                allocated_rank = int(round(self.rank * frac))
            else:
                r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))
                frac = layer_importances.get(name, 1.0) / max(total_importance, 1e-8)
                allocated_rank = int(
                    round(
                        min(in_f, out_f)
                        * self.target_param_ratio
                        * frac
                        * len(target_layers)
                    )
                )

            if self._is_fused_qkv_layer(name, module):
                if self.rank is not None:
                    r_floor = max(r_floor, min(288, self.rank))
                    max_param_r = min(max_param_r, 432)
                else:
                    r_floor = max(r_floor, 288)
                    max_param_r = min(max_param_r, 432)

            allocated_rank = max(r_floor, min(allocated_rank, max_param_r))

            svd_layer, err, iters = self.compress_linear(
                module,
                rank=allocated_rank,
                name=name,
            )
            self.layer_errors[name] = err
            self.layer_ranks[name] = allocated_rank
            self.layer_iters[name] = iters

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "AdaSVD: %s reconstruction error %.4f > max_error %.4f; "
                    "falling back to original.",
                    name,
                    err,
                    self.max_error,
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    # Fallback keeps original bytes, so add them to comp
                    try:
                        for p in target_mod.parameters():
                            self._comp_storage_bytes += p.numel() * p.element_size()
                    except Exception:
                        pass
                    continue

            # ---- Splice ----
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
            self._compressed_names_seen.add(name)

            # ---- Storage accounting ----
            try:
                for p in svd_layer.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
                for b in svd_layer.buffers():
                    if b is not None:
                        self._comp_storage_bytes += b.numel() * b.element_size()
            except Exception:
                pass

        # ---- Freeze fallback layers ----
        for name, mod in self.model.named_modules():
            if name in self.fallback_layers:
                target = getattr(mod, "base_layer", mod)
                for p in target.parameters():
                    p.requires_grad = False

        # ---- Recompute honest storage ----
        comp_bytes_real = 0
        for name, mod in self.model.named_modules():
            if name in self.compressed_layers or name in self.fallback_layers:
                target = getattr(mod, "base_layer", mod)
                try:
                    for p in target.parameters():
                        comp_bytes_real += p.numel() * p.element_size()
                    for b in target.buffers():
                        if b is not None:
                            comp_bytes_real += b.numel() * b.element_size()
                except Exception:
                    pass
        self._comp_storage_bytes = comp_bytes_real

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
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "layer_iters": self.layer_iters,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "avg_reconstruction_error": avg_err,
                "calibration_time_s": self.calibration_time_s,
                "use_weighted_als": self.use_weighted_als,
                "use_mean_shift": self.use_mean_shift,
                "use_gating_d": self.use_gating_d,
                "orig_storage_bytes": self._orig_storage_bytes,
                "comp_storage_bytes": self._comp_storage_bytes,
                "storage_compression_ratio": round(storage_ratio, 3),
                "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
            }
        )
        try:
            self.model.compression_stats = stats
        except Exception:
            pass

        logger.info(
            "AdaSVD done: %d compressed, %d fallback, avg_err=%.4f, storage_ratio=%.2fx",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            storage_ratio,
        )
        return self.model, stats


# ============================================================================
# Factory
# ============================================================================
@COMPRESSOR_REGISTRY.register("adasvd_factory")
@COMPRESSOR_REGISTRY.register("adasvd_compressor_factory")
def adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    rank = kwargs.pop(
        "rank",
        _cfg_get(comp_config, "adasvd_rank", _cfg_get(comp_config, "rank", None)),
    )
    target_param_ratio = kwargs.pop(
        "target_param_ratio",
        _cfg_get(
            comp_config,
            "adasvd_target_ratio",
            _cfg_get(comp_config, "target_param_ratio", 0.7),
        ),
    )
    num_iters = kwargs.pop(
        "num_iters",
        _cfg_get(comp_config, "adasvd_iters", _cfg_get(comp_config, "num_iters", 10)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "adasvd_max_error", _cfg_get(comp_config, "max_error", 0.50)
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "adasvd_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "adasvd_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", True),
        ),
    )
    min_rank = kwargs.pop(
        "min_rank",
        _cfg_get(comp_config, "adasvd_min_rank", _cfg_get(comp_config, "min_rank", 16)),
    )
    use_weighted_als = kwargs.pop(
        "use_weighted_als",
        _cfg_get(
            comp_config,
            "adasvd_use_weighted_als",
            _cfg_get(comp_config, "use_weighted_als", False),
        ),
    )
    use_mean_shift = kwargs.pop(
        "use_mean_shift",
        _cfg_get(
            comp_config,
            "adasvd_use_mean_shift",
            _cfg_get(comp_config, "use_mean_shift", True),
        ),
    )
    use_gating_d = kwargs.pop(
        "use_gating_d",
        _cfg_get(
            comp_config,
            "adasvd_use_gating_d",
            _cfg_get(comp_config, "use_gating_d", True),
        ),
    )
    tokenizer = kwargs.pop("tokenizer", _cfg_get(comp_config, "tokenizer", None))
    calibration_texts = kwargs.pop(
        "calibration_texts", _cfg_get(comp_config, "calibration_texts", None)
    )
    calibration_data = kwargs.pop("calibration_data", None)
    max_calib_tokens = kwargs.pop(
        "max_calib_tokens", _cfg_get(comp_config, "max_calib_tokens", 4096)
    )
    fused_qkv_error_threshold = kwargs.pop(
        "fused_qkv_error_threshold",
        _cfg_get(comp_config, "adasvd_fused_qkv_error_threshold", 0.30),
    )

    compressor = AdaSVDCompressor(
        model,
        rank=rank,
        target_param_ratio=target_param_ratio,
        num_iters=num_iters,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        min_rank=min_rank,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        use_weighted_als=use_weighted_als,
        use_mean_shift=use_mean_shift,
        use_gating_d=use_gating_d,
        comp_config=comp_config,
        max_calib_tokens=max_calib_tokens,
        fused_qkv_error_threshold=fused_qkv_error_threshold,
        **kwargs,
    )
    compressed_model, stats = compressor.compress(calibration_data=calibration_data)
    if return_stats:
        return compressed_model, stats
    return compressed_model


adasvd_compressor_factory = adasvd_factory

__all__ = [
    "AdaSVDCompressor",
    "FusedQKVSVDLinear",
    "adasvd_factory",
    "adasvd_compressor_factory",
    "DEFAULT_CALIBRATION_CORPUS",
]
