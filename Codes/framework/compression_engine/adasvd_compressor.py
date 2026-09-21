"""
AdaSVD: Adaptive SVD with Activation Whitening (Cholesky), Effective Rank, and Activation Error.
Paper Reference: Internal / Custom method.
Registered as 'adasvd', 'adasvd_compressor', 'adasvd_factory', 'adasvd_compressor_factory'.

3-Step Architecture:
1. Cholesky Whitening:
   - Collects activations X for each target layer using AASVD calibration corpus.
   - Computes activation covariance C_X = (X.T @ X) / N.
   - Computes Cholesky factor S_white = cholesky(C_X + eps * I).
   - Whitens weight W_white = W @ S_white.
   - SVD on whitened weight U, S, Vt = svd(W_white).
   - Unwhitens right factor Vt = Vt @ inv(S_white).
2. Effective Rank:
   - Allocates rank proportional to Shannon entropy effective rank:
     p = s**2 / (s**2).sum(), erank = exp(- (p * log(p)).sum()).
3. Activation Error Validation:
   - Measures actual downstream effect:
     err = ||WX - reconX||_F / (||WX||_F + 1e-9).
"""

from __future__ import annotations

import contextlib
import logging
import math
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
        num_iters: int = 0,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = True,
        min_rank: int = 16,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        comp_config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adasvd", **kwargs)
        cfg = comp_config or kwargs.get("comp_config", kwargs.get("cfg", getattr(model, "config", None)))

        self.rank = _cfg_get(cfg, "adasvd_rank", _cfg_get(cfg, "rank", rank))
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
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.layer_iters: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, SVDLinear):
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

    # =========================================================================
    # STEP 2: Replace Nuclear Norm with Effective Rank
    # =========================================================================
    @staticmethod
    def _compute_effective_rank(W: torch.Tensor) -> float:
        """
        Compute Shannon entropy effective rank (Roy & Vetterli, 2007).
        p = s**2 / (s**2).sum().clamp(min=1e-12)
        return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
        """
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            s_sq = s**2
            denom = s_sq.sum().clamp(min=1e-12)
            p = s_sq / denom
            p = p[p > 1e-12]
            return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())

    # Backward compatibility alias
    _compute_nuclear_norm = _compute_effective_rank

    def _collect_activations(self, calibration_data: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """
        Step 1 Helper: Register hooks on target layers, run one forward pass on calibration data,
        and return dict[name] = X tensor.
        """
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                target_mod = getattr(module, "base_layer", module)
                cls_name = module.__class__.__name__.lower()
                target_cls = target_mod.__class__.__name__.lower()
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
                if mask.sum() > 0:
                    x_active = x_flat[mask]
                else:
                    x_active = x_flat

                if layer_name not in activations:
                    activations[layer_name] = [x_active[:512].cpu()]
                else:
                    curr_n = sum(t.size(0) for t in activations[layer_name])
                    if curr_n < 1024:
                        avail = 1024 - curr_n
                        activations[layer_name].append(x_active[:avail].cpu())

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
                    # Synthetic calibration token or float input fallback with sufficient rank
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
                result[name] = torch.cat(tensor_list, dim=0)

        return result

    # =========================================================================
    # STEP 1: Add Cholesky Whitening & STEP 3: Activation Error
    # =========================================================================
    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
        name: str = "",
        covariance: Optional[torch.Tensor] = None,
        X_calib: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """
        Compress a single linear/conv1d layer using Cholesky activation whitening,
        truncated SVD, and downstream activation reconstruction validation.
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = layer.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()
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
        r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))
        if "c_attn" in name.lower() or (in_f >= 768 and out_f >= 2000):
            r_floor = max(r_floor, 288)
            max_param_r = min(max_param_r, 432)

        r = max(r_floor, min(rank, max_param_r))
        if r > max_param_r:
            r = max_param_r

        device = weight_tensor.device
        dtype = weight_tensor.dtype

        # ---------------------------------------------------------------------
        # STEP 1: Add Cholesky Whitening
        # ---------------------------------------------------------------------
        X = X_calib if X_calib is not None else self.calibration_activations.get(name, None)
        if X is not None:
            X = X.to(device=W.device, dtype=W.dtype)
            C_X = (X.T @ X) / max(1, X.shape[0])
        elif covariance is not None and covariance.shape == (in_f, in_f):
            C_X = covariance.to(device=W.device, dtype=W.dtype)
        else:
            C_X = None

        if C_X is not None:
            eps = 1e-6 * torch.trace(C_X) / max(1, C_X.shape[0])
            eye = torch.eye(C_X.shape[0], device=C_X.device, dtype=C_X.dtype)
            try:
                S_white = torch.linalg.cholesky(C_X + eps * eye)
            except Exception:
                for exp in [-5, -4, -3, -2, -1]:
                    try:
                        S_white = torch.linalg.cholesky(C_X + (10**exp * (torch.trace(C_X).item() / max(1, C_X.shape[0]))) * eye)
                        break
                    except Exception:
                        continue
                else:
                    S_white = None

            if S_white is not None:
                try:
                    S_inv = torch.linalg.inv(S_white)
                    W_white = W @ S_white
                    U, S, Vt = torch.linalg.svd(W_white, full_matrices=False)
                    Vt = Vt @ S_inv
                except Exception as e:
                    logger.debug("Whitening SVD failed on %s: %s", name, e)
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            else:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        else:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A = U[:, :r] * sqrt_S.unsqueeze(0)       # [out_features, r]
        B = sqrt_S.unsqueeze(1) * Vt[:r, :]      # [r, in_features]
        recon = torch.matmul(A, B)

        # ---------------------------------------------------------------------
        # STEP 3: Use Activation Error
        # ---------------------------------------------------------------------
        if X is not None:
            WX = W @ X.T
            reconX = recon @ X.T
            final_err = float(torch.norm(WX - reconX, p="fro").item() / (torch.norm(WX, p="fro").item() + 1e-9))
        else:
            W_norm = torch.norm(W, p="fro").item()
            if W_norm < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - recon, p="fro").item() / (W_norm + 1e-9))

        # Instantiate factorized SVDLinear layer
        bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))
        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias_tensor is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        if bias_tensor is not None and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(bias_tensor.data)

        return svd_layer, final_err, 0

    def compress(self, calibration_data: Optional[Any] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Executes adaptive rank assignment and SVD factorization across all target linear/conv1d layers.
        """
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

        # Step 2: Compute spectral effective rank in FP32
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

        # Step 1: Collect activation statistics across target layers
        self.calibration_activations = self._collect_activations(calibration_data=calibration_data)

        # Compress target layers
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
            r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))
            if "c_attn" in name.lower() or (in_f >= 768 and out_f >= 2000):
                r_floor = max(r_floor, 288)
                max_param_r = min(max_param_r, 432)

            if self.rank is not None:
                frac = layer_importances.get(name, 1.0) / (total_importance / max(1, len(target_layers)))
                allocated_rank = int(round(self.rank * frac))
            else:
                frac = layer_importances.get(name, 1.0) / max(total_importance, 1e-8)
                allocated_rank = int(round(min(in_f, out_f) * self.target_param_ratio * frac * len(target_layers)))

            allocated_rank = max(r_floor, min(allocated_rank, max_param_r))
            if allocated_rank > max_param_r:
                allocated_rank = max_param_r

            svd_layer, err, iters = self.compress_linear(module, rank=allocated_rank, name=name)
            self.layer_errors[name] = err
            self.layer_ranks[name] = allocated_rank
            self.layer_iters[name] = iters

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "AdaSVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
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
            if self.compressed_layers
            else 0.0
        )
        stats["avg_reconstruction_error"] = avg_err
        logger.info(
            "AdaSVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("adasvd_factory")
@COMPRESSOR_REGISTRY.register("adasvd_compressor_factory")
def adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """AdaSVD factory function with robust config handling."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "adasvd_rank", _cfg_get(comp_config, "rank", None)))
    target_param_ratio = kwargs.pop(
        "target_param_ratio", _cfg_get(comp_config, "adasvd_target_ratio", _cfg_get(comp_config, "target_param_ratio", 0.7))
    )
    num_iters = kwargs.pop("num_iters", _cfg_get(comp_config, "adasvd_iters", _cfg_get(comp_config, "num_iters", 0)))
    max_error = kwargs.pop(
        "max_error", _cfg_get(comp_config, "adasvd_max_error", _cfg_get(comp_config, "max_error", 0.50))
    )
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "adasvd_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "adasvd_target_attention_only", _cfg_get(comp_config, "target_attention_only", True))
    )
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "adasvd_min_rank", _cfg_get(comp_config, "min_rank", 16)))
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
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress(calibration_data=calibration_data)
    return compressed_model


adasvd_compressor_factory = adasvd_factory

__all__ = ["AdaSVDCompressor", "adasvd_factory", "adasvd_compressor_factory"]
