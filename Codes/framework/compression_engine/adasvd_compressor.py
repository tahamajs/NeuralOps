"""
AdaSVD: Adaptive SVD with Activation Whitening (Cholesky), Effective Rank,
Two-Sided Weighted ALS, and Mean-Shift Bias Absorption.
Paper Reference: Internal / Custom method.
Registered as 'adasvd', 'adasvd_compressor', 'adasvd_factory', 'adasvd_compressor_factory'.

Key Components:
1. Mean-Shift Bias Absorption:
   - Absorbs the activation mean offset mu_x into layer bias (b_new = b + W @ mu_x) with zero rank cost.
   - Centers activations at inference (y = AB(x - mu_x) + b_new), freeing low-rank capacity to model variance.
2. Two-Sided Weighted ALS:
   - Minimizes || D^{1/2} (W - AB) C^{1/2} ||_F^2 where C = C_X (input covariance) and D = diag(Var(Y)).
   - Alternating updates: A = W E B.T (B E B.T + eps_B I)^{-1}, B = (A.T D A + eps_A I)^{-1} A.T D W.
3. Information-Theoretic Effective Rank:
   - Shannon entropy spectral rank: p_i = s_i^2 / sum(s_j^2), erank = exp(-sum(p * ln(p))).
4. Activation Error Validation:
   - Measures downstream effect on centered activation distribution.

=== Fix (2026-09-25): Q/K/V-Aware SVD for GPT-2 Fused Conv1D ===
Problem: GPT-2's c_attn is a fused Conv1D (768, 2304) packing [Q|K|V] into one
matrix. Matrix SVD treats this as one homogeneous matrix, but Q, K, V have
different statistical patterns. Compressing the fused matrix destroys softmax
attention semantics, causing PPL to blow up (41 -> 500+).

Solution: Detect fused QKV, split it into 3 independent SVDLinear factors,
concatenate outputs in forward pass. Each factor compressed independently.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

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
# FusedQKVSVDLinear: Q/K/V-Aware SVD for GPT-2 fused Conv1D projections
# ============================================================================
class FusedQKVSVDLinear(nn.Module):
    """
    Compressed replacement for GPT-2's fused c_attn (Q/K/V) Conv1D.

    Internally holds 3 independent low-rank factors (one for Q, K, V).
    Forward pass concatenates them so the output shape matches the original
    fused projection exactly: (..., in_features) -> (..., 3 * split_size).

    Layout compatibility: matches GPT-2 Conv1D semantics (weight is (in, out))
    via the reconstructed `.weight` property.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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
        # Conv1D compatibility
        self.nf = out_features
        self.nx = in_features
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        # Determine construction device/dtype
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        # ===== Q factor =====
        self.q_A = nn.Parameter(
            torch.empty(self.split_size, self.rank, dtype=dtype, device=device)
        )
        self.q_B = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.zeros_(self.q_B)

        # ===== K factor =====
        self.k_A = nn.Parameter(
            torch.empty(self.split_size, self.rank, dtype=dtype, device=device)
        )
        self.k_B = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.k_A, a=math.sqrt(5))
        nn.init.zeros_(self.k_B)

        # ===== V factor =====
        self.v_A = nn.Parameter(
            torch.empty(self.split_size, self.rank, dtype=dtype, device=device)
        )
        self.v_B = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))
        nn.init.zeros_(self.v_B)

        # ===== Bias =====
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        # Optional LoRA on the fused output
        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_r, in_features, dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, self.lora_r, dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    def set_fused_factors(self, W_fused: torch.Tensor) -> None:
        """
        Split a fused (in_features, out_features) weight and store each
        third as low-rank factors (U*S, V) in the corresponding slot.

        Handles both Conv1D layout (in, out) and nn.Linear layout (out, in).
        """
        if W_fused.shape == (self.in_features, self.out_features):
            W = W_fused.t().contiguous().float()  # Conv1D -> Linear layout
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
            (W[S:2 * S, :], self.k_A, self.k_B),
            (W[2 * S:, :], self.v_A, self.v_B),
        ]
        for W_part, A_param, B_param in splits:
            U, Svals, Vt = torch.linalg.svd(W_part, full_matrices=False)
            r = min(self.rank, Svals.shape[0])
            # A_new = U[:, :r] * sqrt(S),  B_new = sqrt(S) * Vt[:r, :]
            sqrt_S = torch.sqrt(torch.clamp(Svals[:r], min=1e-12))
            A_new = U[:, :r] * sqrt_S.unsqueeze(0)
            B_new = sqrt_S.unsqueeze(1) * Vt[:r, :]
            with torch.no_grad():
                A_param.data.zero_()
                B_param.data.zero_()
                # A_param has shape (split_size, rank); set first r columns
                A_param.data[:, :r].copy_(A_new.to(A_param.dtype).to(A_param.device))
                # B_param has shape (rank, in_features); set first r rows
                B_param.data[:r, :].copy_(B_new.to(B_param.dtype).to(B_param.device))

    # ------------------------------------------------------------------
    def _forward_factor(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> out: (..., split_size)
        # B: (rank, in_features), A: (split_size, rank)
        h = torch.nn.functional.linear(x.to(B.dtype), B)  # (..., rank)
        return torch.nn.functional.linear(h, A)  # (..., split_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._forward_factor(x, self.q_A, self.q_B)
        k = self._forward_factor(x, self.k_A, self.k_B)
        v = self._forward_factor(x, self.v_A, self.v_B)
        out = torch.cat([q, k, v], dim=-1)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = torch.nn.functional.linear(
                torch.nn.functional.linear(x.float(), self.lora_A),
                self.lora_B,
            ) * self.scaling
            out = out + lora_out.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """
        Reconstruct the fused (in_features, out_features) weight for
        compatibility with code that inspects `.weight`.
        """
        W_q = (self.q_A @ self.q_B).float()  # (split_size, in_features)
        W_k = (self.k_A @ self.k_B).float()
        W_v = (self.v_A @ self.v_B).float()
        W_linear = torch.cat([W_q, W_k, W_v], dim=0)  # (out_features, in_features)
        return W_linear.t().contiguous()  # Conv1D layout (in, out)

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        """Set the fused weight by re-splitting and reloading factors."""
        self.set_fused_factors(value)


# Standard multi-domain calibration corpus (identical to AASVD)
DEFAULT_CALIBRATION_CORPUS: List[str] = [
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


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("adasvd")
@COMPRESSOR_REGISTRY.register("adasvd_compressor")
class AdaSVDCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Adaptive SVD Compressor (AdaSVD).
    Paper Reference: Internal / Custom method.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        target_param_ratio: float = 0.7,
        num_iters: int = 10,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = True,
        min_rank: int = 16,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        use_weighted_als: bool = False,
        use_mean_shift: bool = True,
        use_gating_d: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        comp_config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adasvd", **kwargs)
        cfg = comp_config or kwargs.get("comp_config", kwargs.get("cfg", getattr(model, "config", None)))

        # Check CLI args for explicit rank override and boolean flags
        cli_rank = None
        cli_use_gating_d = None
        cli_use_weighted_als = None
        import sys
        for _idx, _arg in enumerate(sys.argv):
            if _arg in ("--rank", "--adasvd_rank") and _idx + 1 < len(sys.argv):
                try:
                    cli_rank = int(sys.argv[_idx + 1])
                except ValueError:
                    pass
            elif _arg.startswith("--rank=") or _arg.startswith("--adasvd_rank="):
                try:
                    cli_rank = int(_arg.split("=")[1])
                except ValueError:
                    pass
            elif _arg in ("--adasvd_use_gating_d", "--use_gating_d") and _idx + 1 < len(sys.argv):
                cli_use_gating_d = sys.argv[_idx + 1].lower() in ("true", "1", "yes")
            elif _arg.startswith("--adasvd_use_gating_d=") or _arg.startswith("--use_gating_d="):
                cli_use_gating_d = _arg.split("=")[1].lower() in ("true", "1", "yes")
            elif _arg in ("--adasvd_use_weighted_als", "--use_weighted_als") and _idx + 1 < len(sys.argv):
                cli_use_weighted_als = sys.argv[_idx + 1].lower() in ("true", "1", "yes")
            elif _arg.startswith("--adasvd_use_weighted_als=") or _arg.startswith("--use_weighted_als="):
                cli_use_weighted_als = _arg.split("=")[1].lower() in ("true", "1", "yes")

        self.rank = cli_rank if cli_rank is not None else _cfg_get(cfg, "adasvd_rank", _cfg_get(cfg, "rank", rank))

        self.target_param_ratio = float(
            _cfg_get(cfg, "adasvd_target_ratio", _cfg_get(cfg, "target_param_ratio", target_param_ratio))
        )
        self.num_iters = int(_cfg_get(cfg, "adasvd_iters", _cfg_get(cfg, "num_iters", num_iters)))
        self.max_error = float(_cfg_get(cfg, "adasvd_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.fallback_to_original = bool(
            _cfg_get(cfg, "adasvd_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.target_attention_only = bool(
            _cfg_get(cfg, "adasvd_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.min_rank = int(_cfg_get(cfg, "adasvd_min_rank", _cfg_get(cfg, "min_rank", min_rank)))
        if cli_use_weighted_als is not None:
            self.use_weighted_als = cli_use_weighted_als
        else:
            self.use_weighted_als = bool(
                _cfg_get(cfg, "adasvd_use_weighted_als", _cfg_get(cfg, "use_weighted_als", use_weighted_als))
            )
        self.use_mean_shift = bool(
            _cfg_get(cfg, "adasvd_use_mean_shift", _cfg_get(cfg, "use_mean_shift", use_mean_shift))
        )
        if cli_use_gating_d is not None:
            self.use_gating_d = cli_use_gating_d
        else:
            self.use_gating_d = bool(
                _cfg_get(cfg, "adasvd_use_gating_d", _cfg_get(cfg, "use_gating_d", use_gating_d))
            )
        self.target_modules = target_modules

        if tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None) or getattr(getattr(model, "config", None), "tokenizer", None)
        if tokenizer is None:
            model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
            if model_name_or_path:
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as e:
                    logger.debug("Could not auto-load tokenizer from %s: %s", model_name_or_path, e)
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

        self.calibration_activations: Dict[str, torch.Tensor] = {}
        self.layer_means: Dict[str, torch.Tensor] = {}
        self.layer_out_means: Dict[str, torch.Tensor] = {}
        self.layer_out_vars: Dict[str, torch.Tensor] = {}
        self.mean_shifts: Dict[str, torch.Tensor] = {}

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.layer_iters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, (SVDLinear, FusedQKVSVDLinear)):
            return False
        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return False

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
            or "loralinear" in cls_name
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score", "pooler"]):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj", "query", "key", "value",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _is_fused_qkv_layer(name: str, module: nn.Module) -> bool:
        """
        Detect GPT-2 style fused QKV Conv1D projections.

        Conditions:
        - Layer name contains 'c_attn'
        - Module exposes `nf` attribute (Conv1D) and nf % 3 == 0
        - Weight is 2D
        """
        if "c_attn" not in name.lower():
            return False
        target_mod = getattr(module, "base_layer", module)
        nf = getattr(module, "nf", None) or getattr(target_mod, "nf", None)
        if nf is None or nf % 3 != 0:
            return False
        w = getattr(target_mod, "weight", None)
        return w is not None and w.dim() == 2

    # ------------------------------------------------------------------
    @staticmethod
    def _gelu_derivative_squared(y: torch.Tensor) -> torch.Tensor:
        """
        Compute (GELU'(y))^2 for each element.
        GELU(y) = y * Φ(y), where Φ is standard normal CDF.
        GELU'(y) = Φ(y) + y * φ(y), where φ is standard normal PDF.
        """
        import math
        phi = 0.5 * (1.0 + torch.erf(y / math.sqrt(2.0)))
        pdf = torch.exp(-y ** 2 / 2.0) / math.sqrt(2.0 * math.pi)
        gelu_prime = phi + y * pdf
        return gelu_prime ** 2

    def _is_followed_by_gelu(self, layer_name: str) -> bool:
        name_lower = layer_name.lower()
        if "lin1" in name_lower:
            return True
        if "intermediate.dense" in name_lower:
            return True
        if "gate_proj" in name_lower or "up_proj" in name_lower:
            return True
        if "ffn" in name_lower and "lin1" in name_lower:
            return True
        return False

    @staticmethod
    def _compute_effective_rank(W: torch.Tensor) -> float:
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            s_sq = s**2
            denom = s_sq.sum().clamp(min=1e-12)
            p = s_sq / denom
            p = p[p > 1e-12]
            return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())

    _compute_nuclear_norm = _compute_effective_rank

    # ------------------------------------------------------------------
    def _collect_activations(self, calibration_data: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """
        Register hooks on target layers, run calibration forward pass,
        record input activations, layer input means mu_x, output means mu_y,
        and output variances D.
        """
        activations: Dict[str, List[torch.Tensor]] = {}
        out_activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                target_mod = getattr(module, "base_layer", module)
                is_conv1d = (
                    (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                    or hasattr(module, "nf")
                    or hasattr(target_mod, "nf")
                )
                weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
                if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
                    return

                in_f = weight_tensor.shape[0] if is_conv1d else weight_tensor.shape[1]
                x_flat = x.float().reshape(-1, in_f)
                mask = x_flat.abs().sum(dim=-1) > 1e-5
                x_active = x_flat[mask] if mask.sum() > 0 else x_flat

                if layer_name not in activations:
                    activations[layer_name] = [x_active[:512].cpu()]
                else:
                    curr_n = sum(t.size(0) for t in activations[layer_name])
                    if curr_n < 1024:
                        avail = 1024 - curr_n
                        activations[layer_name].append(x_active[:avail].cpu())

                if isinstance(outp, torch.Tensor):
                    out_f = weight_tensor.shape[1] if is_conv1d else weight_tensor.shape[0]
                    y_flat = outp.detach().float().reshape(-1, out_f)
                    if layer_name not in out_activations:
                        out_activations[layer_name] = [y_flat[:512].cpu()]
                    else:
                        curr_ny = sum(t.size(0) for t in out_activations[layer_name])
                        if curr_ny < 1024:
                            avail_y = 1024 - curr_ny
                            out_activations[layer_name].append(y_flat[:avail_y].cpu())
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
                    if isinstance(calibration_data, torch.Tensor):
                        cal_dev = calibration_data.to(dev)
                        try:
                            self.model(cal_dev)
                        except Exception:
                            self.model(input_ids=cal_dev)
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
                            if isinstance(batch, torch.Tensor):
                                self.model(batch.to(dev))
                            elif isinstance(batch, dict):
                                b_dev = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                                self.model(**b_dev)
                            break
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
                    try:
                        dummy_ids = torch.randint(100, 2000, (8, 64), device=dev)
                        self.model(input_ids=dummy_ids)
                    except Exception:
                        try:
                            dummy_ids = torch.randint(100, 2000, (8, 64), device=dev)
                            self.model(dummy_ids)
                        except Exception:
                            try:
                                dummy_x = torch.randn(128, 64, device=dev)
                                self.model(dummy_x)
                            except Exception as e:
                                logger.debug("AdaSVD synthetic calibration fallback failed: %s", e)
        except Exception as e:
            logger.debug("AdaSVD calibration pass exception: %s", e)
        finally:
            for h in hooks:
                h.remove()

        result: Dict[str, torch.Tensor] = {}
        for name, tensor_list in activations.items():
            if tensor_list:
                cat_x = torch.cat(tensor_list, dim=0)
                result[name] = cat_x
                self.layer_means[name] = cat_x.mean(dim=0).cpu()

        for name, y_list in out_activations.items():
            if y_list:
                cat_y = torch.cat(y_list, dim=0)
                self.layer_out_means[name] = cat_y.mean(dim=0).cpu()

                is_followed_by_gelu = self.use_gating_d and self._is_followed_by_gelu(name)
                if is_followed_by_gelu:
                    gelu_sq = self._gelu_derivative_squared(cat_y)
                    d_values = gelu_sq.mean(dim=0)
                    logger.info(
                        "Gating-Aware D computed for %s (mean=%.4f)",
                        name, d_values.mean().item(),
                    )
                else:
                    d_values = cat_y.var(dim=0, unbiased=False)

                self.layer_out_vars[name] = d_values.cpu()

        return result

    # ------------------------------------------------------------------
    def _init_whitened_svd(
        self,
        W: torch.Tensor,
        r: int,
        C_X: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_out, d_in = W.shape
        device = W.device
        dtype = W.dtype

        if C_X is not None and C_X.shape == (d_in, d_in):
            C_dev = C_X.to(device=device, dtype=torch.float32)
            eps = max(1e-8, float(1e-6 * torch.trace(C_dev).item() / max(1, d_in)))
            eye = torch.eye(d_in, device=device, dtype=torch.float32)
            try:
                S_white = torch.linalg.cholesky(C_dev + eps * eye)
            except Exception:
                for exp in [-5, -4, -3, -2, -1]:
                    try:
                        S_white = torch.linalg.cholesky(
                            C_dev + (10**exp * (torch.trace(C_dev).item() / max(1, d_in))) * eye
                        )
                        break
                    except Exception:
                        continue
                else:
                    S_white = None

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
        X_calib: Optional[torch.Tensor] = None,
        mu_x: Optional[torch.Tensor] = None,
        num_iters: int = 10,
        patience: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        d_out, d_in = W.shape
        device = W.device
        dtype = W.dtype

        A, B = self._init_whitened_svd(W, r, C_X)
        A = A.float()
        B = B.float()

        E = C_X.float().to(device)
        D = torch.diag(torch.clamp(D_var.float().to(device), min=1e-6))
        WE = W.float() @ E
        DW = D @ W.float()

        best_A = A.clone()
        best_B = B.clone()
        best_err = float("inf")
        patience_counter = 0
        actual_iters = 0

        recon_0 = A @ B
        if X_calib is not None:
            X_eval = (X_calib - mu_x.to(device)) if mu_x is not None else X_calib
            WX_c = W.float() @ X_eval.float().T
            reconX_c = recon_0 @ X_eval.float().T
            init_err = float(torch.norm(WX_c - reconX_c, p="fro").item() / (torch.norm(WX_c, p="fro").item() + 1e-9))
        else:
            diff = W.float() - recon_0
            init_err = float(
                math.sqrt(max(0.0, torch.trace(diff @ E @ diff.T).item()))
                / (math.sqrt(max(1e-9, torch.trace(W.float() @ E @ W.float().T).item())) + 1e-9)
            )
        best_err = init_err

        for it in range(num_iters):
            BEBt = B @ E @ B.T
            eps_B = max(1e-9, float(1e-6 * BEBt.trace().item() / max(1, r)))
            try:
                A_new = WE @ B.T @ torch.linalg.inv(
                    BEBt + eps_B * torch.eye(r, device=device, dtype=BEBt.dtype)
                )
            except Exception:
                break

            AtDA = A_new.T @ D @ A_new
            eps_A = max(1e-9, float(1e-6 * AtDA.trace().item() / max(1, r)))
            try:
                B_new = torch.linalg.inv(
                    AtDA + eps_A * torch.eye(r, device=device, dtype=AtDA.dtype)
                ) @ A_new.T @ DW
            except Exception:
                break

            if torch.isnan(A_new).any() or torch.isnan(B_new).any() or torch.isinf(A_new).any() or torch.isinf(B_new).any():
                break

            actual_iters += 1
            recon = A_new @ B_new
            if X_calib is not None:
                X_eval = (X_calib - mu_x.to(device)) if mu_x is not None else X_calib
                WX_c = W.float() @ X_eval.float().T
                reconX_c = recon @ X_eval.float().T
                err = float(torch.norm(WX_c - reconX_c, p="fro").item() / (torch.norm(WX_c, p="fro").item() + 1e-9))
            else:
                diff = W.float() - recon
                err = float(
                    math.sqrt(max(0.0, torch.trace(diff @ E @ diff.T).item()))
                    / (math.sqrt(max(1e-9, torch.trace(W.float() @ E @ W.float().T).item())) + 1e-9)
                )

            if err < best_err - 1e-6:
                best_A = A_new.clone()
                best_B = B_new.clone()
                best_err = err
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            A, B = A_new, B_new

        return best_A.to(dtype), best_B.to(dtype), best_err, actual_iters

    # ==================================================================
    # NEW: _compress_fused_qkv
    # ==================================================================
    def _compress_fused_qkv(
        self,
        layer: nn.Module,
        rank: int,
        name: str,
    ) -> Tuple[Optional[FusedQKVSVDLinear], float, int]:
        """
        Compress a fused Q/K/V projection by splitting it into 3 independent
        low-rank factors. Each third is compressed to the same rank r.

        Returns (new_layer, reconstruction_error, iters=0).
        """
        target_mod = getattr(layer, "base_layer", layer)
        w = getattr(target_mod, "weight", None)
        if w is None or w.dim() != 2:
            return None, float("inf"), 0

        raw_w = w.data
        in_f, out_f = raw_w.shape  # Conv1D layout
        if out_f % 3 != 0:
            logger.warning("FusedQKV %s: out_features %d not divisible by 3", name, out_f)
            return None, float("inf"), 0

        # Compute rank bounds (must fit within each split's min dimension)
        split_size = out_f // 3
        max_rank = min(in_f, split_size)
        # Rank floor: keep at least 16, at most 1/4 of split
        r_floor = max(16, min(int(0.15 * split_size), self.min_rank * 4))
        r = max(r_floor, min(int(rank), max_rank))
        if r > max_rank:
            r = max_rank

        device = raw_w.device
        dtype = raw_w.dtype

        # ===== Build fused layer =====
        has_bias = getattr(target_mod, "bias", None) is not None
        fused_layer = FusedQKVSVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=has_bias,
            dtype=dtype,
            device=device,
            lora_r=0,  # no LoRA by default for the fused case
            lora_alpha=1.0,
        )

        # ===== Split and set factors =====
        try:
            fused_layer.set_fused_factors(raw_w)
        except Exception as e:
            logger.warning("FusedQKV %s factor split failed: %s", name, e)
            return None, float("inf"), 0

        # ===== Copy bias =====
        if has_bias and fused_layer.bias is not None:
            with torch.no_grad():
                fused_layer.bias.data.copy_(target_mod.bias.data)

        # ===== Compute reconstruction error =====
        with torch.no_grad():
            W_recon = fused_layer.weight.to(device=device, dtype=torch.float32)
            W_orig = raw_w.float()
            norm_orig = torch.norm(W_orig, p="fro").clamp_min(1e-8)
            rel_err = (torch.norm(W_orig - W_recon, p="fro") / norm_orig).item()

        # Fused QKV is extremely sensitive to reconstruction error because
        # softmax(QK^T) is exponentially sensitive to Q/K perturbations.
        # We use a stricter threshold (0.15) than the default max_error (0.50)
        # and fall back to the original layer when exceeded.
        FUSED_QKV_ERROR_THRESHOLD = 0.15
        if rel_err > FUSED_QKV_ERROR_THRESHOLD:
            logger.warning(
                "AdaSVD fused-QKV split: %s err=%.4f exceeds strict threshold %.4f; "
                "falling back to original (c_attn compression unsafe at this rank).",
                name, rel_err, FUSED_QKV_ERROR_THRESHOLD,
            )
            return None, rel_err, 0  # triggers fallback in the caller

        logger.info(
            "AdaSVD fused-QKV split: %s in=%d out=%d split=%d rank=%d err=%.4f",
            name, in_f, out_f, split_size, r, rel_err,
        )
        return fused_layer, rel_err, 0
    # ==================================================================
    # compress_linear (now routes fused QKV to _compress_fused_qkv)
    # ==================================================================
    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
        name: str = "",
        covariance: Optional[torch.Tensor] = None,
        X_calib: Optional[torch.Tensor] = None,
        out_variance: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[nn.Module], float, int]:
        """
        Compress a single linear/Conv1D layer.

        Routes fused Q/K/V projections (GPT-2 c_attn) to a special handler.
        """
        # === Fused Q/K/V branch (NEW) ===
        if self._is_fused_qkv_layer(name, layer):
            logger.info("AdaSVD: routing %s to fused-QKV split handler", name)
            return self._compress_fused_qkv(layer, rank, name)

        # === Standard path (unchanged from original) ===
        target_mod = getattr(layer, "base_layer", layer)
        is_conv1d = (
            (Conv1D is not None and (isinstance(layer, Conv1D) or isinstance(target_mod, Conv1D)))
            or hasattr(layer, "nf")
            or hasattr(target_mod, "nf")
        )

        weight_tensor = getattr(target_mod, "weight", getattr(layer, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return None, float("inf"), 0

        raw_w = weight_tensor.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()

        max_param_r = max(1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f))))
        if self.rank is not None:
            r_floor = max(1, min(self.min_rank, rank))
        else:
            r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))

        r = max(r_floor, min(rank, max_param_r))
        if r > max_param_r:
            r = max_param_r

        device = weight_tensor.device
        dtype = weight_tensor.dtype

        # Mean-shift bias absorption
        mu_x = self.layer_means.get(name, None) if self.use_mean_shift else None
        if mu_x is not None:
            mu_x = mu_x.to(device=device, dtype=torch.float32)
            W_fp32 = W.float()
            mean_shift = (W_fp32 @ mu_x).to(device=device, dtype=torch.float32)
            self.mean_shifts[name] = mean_shift.cpu()
        else:
            mean_shift = None

        X = X_calib if X_calib is not None else self.calibration_activations.get(name, None)
        if X is not None:
            X = X.to(device=device, dtype=torch.float32)
            X_centered = (X - mu_x) if mu_x is not None else X
            C_X = (X_centered.T @ X_centered) / max(1, X_centered.shape[0])
        elif covariance is not None and covariance.shape == (in_f, in_f):
            C_X = covariance.to(device=device, dtype=torch.float32)
            X_centered = None
        else:
            C_X = None
            X_centered = None

        D_var = out_variance if out_variance is not None else self.layer_out_vars.get(name, None)

        if C_X is not None and D_var is not None and self.use_weighted_als and self.num_iters > 0:
            A, B, final_err, actual_iters = self._two_sided_als(
                W, r, C_X, D_var, X_calib=X, mu_x=mu_x, num_iters=self.num_iters
            )
        else:
            A, B = self._init_whitened_svd(W, r, C_X)
            actual_iters = 0
            recon = torch.matmul(A, B)
            if X is not None:
                X_eval = X_centered if X_centered is not None else X
                WX = W @ X_eval.T
                reconX = recon @ X_eval.T
                final_err = float(torch.norm(WX - reconX, p="fro").item() / (torch.norm(WX, p="fro").item() + 1e-9))
            else:
                W_norm = torch.norm(W, p="fro").item()
                if W_norm < 1e-12:
                    final_err = 0.0
                else:
                    final_err = float(torch.norm(W - recon, p="fro").item() / (W_norm + 1e-9))

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

        if mu_x is not None:
            svd_layer.register_buffer("input_mean", mu_x.to(device=device, dtype=dtype))
        else:
            svd_layer.register_buffer("input_mean", None)

        return svd_layer, final_err, actual_iters

    # ==================================================================
    # compress
    # ==================================================================
    def compress(self, calibration_data: Optional[Any] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_iters = {}

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaSVD: No compressible layers found.")
            return self.model, self.get_compression_stats()

        # Step 1: Collect activation statistics
        t_calib_start = time.monotonic()
        self.calibration_activations = self._collect_activations(calibration_data=calibration_data)
        self.calibration_time_s = time.monotonic() - t_calib_start
        logger.info("Calibration time: %.2f s", self.calibration_time_s)

        # Step 2: Compute spectral effective rank
        layer_importances = {}
        total_importance = 0.0
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
            if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
                continue
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            raw_w = weight_tensor.data
            W_2d = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            imp = self._compute_effective_rank(W_2d)
            layer_importances[name] = max(imp, 1e-4)
            total_importance += layer_importances[name]

        # Step 3: Compress target layers
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
            if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
                continue
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            raw_w = weight_tensor.data
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            max_param_r = max(1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f))))
            if self.rank is not None:
                r_floor = max(1, min(self.min_rank, self.rank))
                frac = layer_importances.get(name, 1.0) / (total_importance / max(1, len(target_layers)))
                allocated_rank = int(round(self.rank * frac))
            else:
                r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))
                frac = layer_importances.get(name, 1.0) / max(total_importance, 1e-8)
                allocated_rank = int(round(min(in_f, out_f) * self.target_param_ratio * frac * len(target_layers)))

            # === Fused QKV: force rank >= 288 and cap at 432 ===
            if self._is_fused_qkv_layer(name, module):
                if self.rank is not None:
                    r_floor = max(r_floor, min(288, self.rank))
                    max_param_r = min(max_param_r, 432)
                else:
                    r_floor = max(r_floor, 288)
                    max_param_r = min(max_param_r, 432)

            allocated_rank = max(r_floor, min(allocated_rank, max_param_r))
            if allocated_rank > max_param_r:
                allocated_rank = max_param_r

            svd_layer, err, iters = self.compress_linear(
                module, rank=allocated_rank, name=name,
            )
            self.layer_errors[name] = err
            self.layer_ranks[name] = allocated_rank
            self.layer_iters[name] = iters

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "AdaSVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err, name, self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping AdaSVD compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            if hasattr(module, "base_layer"):
                module.base_layer = svd_layer
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    for p in (module.lora_A.parameters() if hasattr(module.lora_A, "parameters") else [module.lora_A]):
                        p.requires_grad = True
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    for p in (module.lora_B.parameters() if hasattr(module.lora_B, "parameters") else [module.lora_B]):
                        p.requires_grad = True
            else:
                replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)

        # Freeze fallback layers
        for name, mod in self.model.named_modules():
            if name in self.fallback_layers:
                target = getattr(mod, "base_layer", mod)
                for p in target.parameters():
                    p.requires_grad = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "layer_iters": self.layer_iters,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )
        stats["avg_reconstruction_error"] = avg_err
        stats["calibration_time_s"] = getattr(self, "calibration_time_s", 0.0)
        logger.info(
            "AdaSVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers), len(self.fallback_layers), avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("adasvd_factory")
@COMPRESSOR_REGISTRY.register("adasvd_compressor_factory")
def adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """AdaSVD factory function with robust config handling."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "adasvd_rank", _cfg_get(comp_config, "rank", None)))
    target_param_ratio = kwargs.pop(
        "target_param_ratio",
        _cfg_get(comp_config, "adasvd_target_ratio", _cfg_get(comp_config, "target_param_ratio", 0.7)),
    )
    num_iters = kwargs.pop("num_iters", _cfg_get(comp_config, "adasvd_iters", _cfg_get(comp_config, "num_iters", 10)))
    max_error = kwargs.pop(
        "max_error", _cfg_get(comp_config, "adasvd_max_error", _cfg_get(comp_config, "max_error", 0.50))
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "adasvd_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "adasvd_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", True)),
    )
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "adasvd_min_rank", _cfg_get(comp_config, "min_rank", 16)))
    use_weighted_als = kwargs.pop(
        "use_weighted_als",
        _cfg_get(comp_config, "adasvd_use_weighted_als",
                 _cfg_get(comp_config, "use_weighted_als", False)),
    )
    use_mean_shift = kwargs.pop(
        "use_mean_shift",
        _cfg_get(comp_config, "adasvd_use_mean_shift",
                 _cfg_get(comp_config, "use_mean_shift", True)),
    )
    use_gating_d = kwargs.pop(
        "use_gating_d",
        _cfg_get(comp_config, "adasvd_use_gating_d",
                 _cfg_get(comp_config, "use_gating_d", True)),
    )
    tokenizer = kwargs.pop("tokenizer", _cfg_get(comp_config, "tokenizer", None))
    calibration_texts = kwargs.pop("calibration_texts", _cfg_get(comp_config, "calibration_texts", None))
    calibration_data = kwargs.pop("calibration_data", None)

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
        **kwargs,
    )
    compressed_model, _ = compressor.compress(calibration_data=calibration_data)
    return compressed_model


adasvd_compressor_factory = adasvd_factory

__all__ = [
    "AdaSVDCompressor",
    "FusedQKVSVDLinear",
    "adasvd_factory",
    "adasvd_compressor_factory",
]