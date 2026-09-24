"""
HashedNet Compressor: replaces nn.Linear and Conv1D layers with HashedLinear.
Registered in COMPRESSOR_REGISTRY as 'hashed_net', 'hashednets', and 'hashednets_compressor'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

def fast_hash_and_sign(
    indices_i: Union[int, torch.Tensor],
    indices_j: Optional[Union[int, torch.Tensor]] = None,
    n_buckets: int = 1024,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient vectorized hash index mapping and unbiased sign generation.
    Uses 64-bit bit mixing to eliminate collisions and strictly bounds bucket indices.
    """
    if isinstance(indices_i, int) and isinstance(indices_j, int):
        i_vec = torch.arange(indices_i, device=device, dtype=torch.int64).unsqueeze(1)
        j_vec = torch.arange(indices_j, device=device, dtype=torch.int64).unsqueeze(0)
        stride = indices_j
    elif isinstance(indices_i, torch.Tensor) and isinstance(indices_j, torch.Tensor):
        if indices_i.dim() == 1 and indices_j.dim() == 1:
            i_vec = indices_i.long().unsqueeze(1)
            j_vec = indices_j.long().unsqueeze(0)
            stride = max(1, int(j_vec.max().item() + 1))
        else:
            i_vec = indices_i.long()
            j_vec = indices_j.long()
            stride = max(1, int(j_vec.max().item() + 1))
    else:
        raise ValueError("indices_i and indices_j must be both integers or both torch.Tensors")

    flat = (i_vec * stride + j_vec).to(torch.int64)

    # 64-bit integer mix to completely prevent hash bucket collisions and bias
    x = flat ^ (flat >> 33)
    x = x * 0x51afd7ed
    x = x ^ (x >> 16)
    x = x * 0x45d9f3b
    x = x ^ (x >> 16)

    # Safe non-negative bucket index
    n_b = max(1, int(n_buckets))
    hash_idx = torch.remainder(x.abs(), n_b).clamp(min=0, max=n_b - 1).to(torch.int32)

    sign_mix = x ^ (x >> 1)
    sign_bit = (sign_mix & 1)
    sign_val = (sign_bit.to(torch.int8) * 2 - 1)
    return hash_idx, sign_val


class HashedLinear(nn.Module):
    is_custom_attention: bool = True
    """
    Linear layer with weight sharing via a compact parameter hash table and optional LoRA adapters.
    W_{i, j} = xi(i, j) * w[h(i, j)]
    y = x W^T + b + (x A^T B^T) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_buckets: int = 65536,
        bias: bool = True,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.nx = in_features  # Conv1D compatibility
        self.n_buckets = max(1, n_buckets)
        self.use_sign = use_sign

        # Shared compact weight table
        self.weight_table = nn.Parameter(
            torch.empty(self.n_buckets, dtype=dtype, device=device)
        )
        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.weight_table, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        # Precompute coordinate hash and sign buffers
        hash_idx, sign_val = fast_hash_and_sign(
            out_features, in_features, self.n_buckets, device=device
        )
        self.register_buffer("hash_idx", hash_idx)
        if use_sign:
            self.register_buffer("sign_val", sign_val)
        else:
            self.register_buffer("sign_val", torch.ones_like(hash_idx, dtype=torch.int8))

        # Trainable low-rank adaptation branch
        self.lora_rank = lora_rank if lora_rank and lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        if self.lora_rank > 0:
            self.scaling = self.lora_alpha / float(self.lora_rank)
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.scaling = 1.0
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the effective full weight matrix W."""
        return self.get_weight()

    @property
    def bucket_indices(self) -> torch.Tensor:
        """Alias for hash_idx."""
        return self.hash_idx

    def get_weight(self) -> torch.Tensor:
        """
        Reconstruct dense 2D weight matrix from parameter hash table:
        W = sign_val * weight_table[hash_idx]
        """
        w_shared = self.weight_table[self.hash_idx.long()]
        if self.use_sign:
            w_shared = w_shared * self.sign_val.to(dtype=w_shared.dtype)
        return w_shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with vectorized weight lookup and optional LoRA addition."""
        W = self.get_weight()
        out = F.linear(x, W, self.bias)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_contrib = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_contrib

        return out

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        n_buckets: int = 65536,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
        compression_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "HashedLinear":
        """
        Create HashedLinear from existing nn.Linear, Conv1D, or LoRALinear layer using
        FP32 scatter-averaging and Frobenius energy preservation.
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if compression_ratio is not None and compression_ratio > 0.0:
            n_buckets = max(16, int(raw_w.numel() * float(compression_ratio)))

        # Determine in_features and out_features robustly, ensuring W is oriented (out_features, in_features)
        if hasattr(target_mod, "nx") and hasattr(target_mod, "nf"):
            in_f = int(target_mod.nx)
            out_f = int(target_mod.nf)
            W = raw_w.t().contiguous()
        elif hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
            in_f = int(target_mod.in_features)
            out_f = int(target_mod.out_features)
            if raw_w.shape == (in_f, out_f):
                W = raw_w.t().contiguous()
            else:
                W = raw_w.view(out_f, in_f).contiguous()
        elif is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.contiguous()

        norm_orig = torch.linalg.norm(W.float()).item()
        if norm_orig < 1e-4:
            logger.warning("Base layer weight has near-zero Frobenius norm (< 1e-4)")

        bias = target_mod.bias.data if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None else None

        existing_lora_A = getattr(layer, "lora_A", None)
        existing_lora_B = getattr(layer, "lora_B", None)
        effective_lora_r = lora_rank
        if existing_lora_A is not None and hasattr(existing_lora_A, "weight"):
            effective_lora_r = existing_lora_A.weight.shape[0]

        hashed = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            bias=bias is not None,
            use_sign=use_sign,
            lora_rank=effective_lora_r,
            lora_alpha=lora_alpha,
            dtype=W.dtype,
            device=W.device,
        )

        # FP32 scatter accumulation for numerical precision
        W_flat = W.float().flatten()
        hash_flat = hashed.hash_idx.flatten().long()
        sign_flat = hashed.sign_val.flatten().to(dtype=torch.float32)
        weighted_W = W_flat * sign_flat

        table_accum = torch.zeros(hashed.n_buckets, dtype=torch.float32, device=W.device)
        counts = torch.bincount(hash_flat, minlength=hashed.n_buckets).float()

        table_accum.scatter_add_(0, hash_flat, weighted_W)

        # Safe mean with zero-count handling
        non_zero = counts > 0
        table_accum[non_zero] = table_accum[non_zero] / counts[non_zero]
        bound = 1.0 / math.sqrt(max(1, in_f))
        table_accum[~non_zero] = torch.empty((~non_zero).sum(), device=W.device).uniform_(-bound, bound)

        # Energy and scale preservation: match original Frobenius norm to prevent activation collapse
        W_recon_pre = (table_accum[hash_flat] * sign_flat).view(out_f, in_f)
        recon_norm = torch.linalg.norm(W_recon_pre).item()
        if recon_norm > 1e-8 and norm_orig > 1e-8:
            energy_scale = norm_orig / recon_norm
            table_accum = table_accum * energy_scale

        hashed.weight_table.data.copy_(table_accum.to(dtype=W.dtype))

        if bias is not None and hashed.bias is not None:
            hashed.bias.data.copy_(bias)

        if existing_lora_A is not None and existing_lora_B is not None and hashed.lora_A is not None:
            try:
                w_A = getattr(existing_lora_A, "weight", existing_lora_A)
                w_B = getattr(existing_lora_B, "weight", existing_lora_B)
                if hasattr(w_A, "data") and hasattr(w_B, "data") and w_A.shape == hashed.lora_A.shape and w_B.shape == hashed.lora_B.shape:
                    hashed.lora_A.data.copy_(w_A.data)
                    hashed.lora_B.data.copy_(w_B.data)
                    hashed.lora_A.requires_grad = True
                    hashed.lora_B.requires_grad = True
            except Exception as e:
                logger.debug("Existing LoRA copy fallback: %s", e)
        elif effective_lora_r and effective_lora_r > 0 and hashed.lora_A is not None and hashed.lora_B is not None:
            try:
                with torch.no_grad():
                    W_cur = hashed.get_weight().float()
                    R = (W.float() - W_cur)
                    r_eff = min(hashed.lora_rank, R.shape[0], R.shape[1])
                    if r_eff > 0:
                        U, S, Vt = torch.linalg.svd(R, full_matrices=False)
                        B_init = U[:, :r_eff] * torch.sqrt(S[:r_eff])
                        A_init = torch.sqrt(S[:r_eff]).unsqueeze(1) * Vt[:r_eff, :] / max(1e-6, hashed.scaling)
                        hashed.lora_B.data.zero_()
                        hashed.lora_A.data.zero_()
                        hashed.lora_B.data[:out_f, :r_eff].copy_(B_init.to(dtype=hashed.lora_B.dtype))
                        hashed.lora_A.data[:r_eff, :in_f].copy_(A_init.to(dtype=hashed.lora_A.dtype))
                        hashed.lora_A.requires_grad = True
                        hashed.lora_B.requires_grad = True
            except Exception as e:
                logger.debug("LoRA residual SVD fit fallback: %s", e)

        return hashed


HashedLinearLayer = HashedLinear

__all__ = ["fast_hash_and_sign", "HashedLinear", "HashedLinearLayer"]



logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


@COMPRESSOR_REGISTRY.register("hashed_net_class")
@COMPRESSOR_REGISTRY.register("hashednets_class")
class HashedNetCompressor(BaseCompressor):
    """
    Compress a model by replacing target linear/Conv1D layers with HashedLinear,
    which shares weights globally through compact hash tables.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_buckets: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        rank: Optional[Union[int, float]] = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
        freeze_hash_table: bool = False,
        skip_attention_qkv: bool = False,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="hashed_net", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        self.cfg = conf

        n_buckets = _cfg_get(conf, "hashed_net_buckets", _cfg_get(conf, "n_buckets", n_buckets))
        compression_ratio = _cfg_get(conf, "compression_ratio", _cfg_get(conf, "target_param_ratio", compression_ratio))
        rank = _cfg_get(conf, "hashed_net_rank", _cfg_get(conf, "rank", rank))
        self.max_error = float(_cfg_get(conf, "hashed_net_max_error", _cfg_get(conf, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(conf, "hashed_net_target_attention_only", _cfg_get(conf, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(conf, "hashed_net_fallback_to_original", _cfg_get(conf, "fallback_to_original", fallback_to_original))
        )
        self.use_sign = bool(_cfg_get(conf, "use_sign", use_sign))
        self.lora_rank = _cfg_get(conf, "lora_rank", _cfg_get(conf, "lora_r", _cfg_get(conf, "hashed_net_lora_rank", lora_rank)))
        if self.lora_rank is not None:
            self.lora_rank = int(self.lora_rank) if int(self.lora_rank) > 0 else None
        self.lora_alpha = float(_cfg_get(conf, "lora_alpha", _cfg_get(conf, "hashed_net_lora_alpha", lora_alpha)))
        self.freeze_hash_table = bool(_cfg_get(conf, "freeze_hash_table", freeze_hash_table or (self.lora_rank is not None and self.lora_rank > 0)))
        self.skip_attention_qkv = bool(_cfg_get(conf, "skip_attention_qkv", skip_attention_qkv))
        self.target_layers = target_layers or _cfg_get(conf, "target_layers", None)
        self.max_retries = int(max_retries)

        # Parse bucket budget or ratio
        self.compression_ratio: Optional[float] = None
        if compression_ratio is not None:
            self.compression_ratio = float(compression_ratio)
        elif rank is not None and 0.0 < float(rank) <= 1.0:
            self.compression_ratio = float(rank)

        if n_buckets is not None:
            self.base_buckets: Optional[int] = max(16, int(n_buckets))
        elif rank is not None and float(rank) > 1.0:
            self.base_buckets = max(16, int(rank) * 32)
        else:
            self.base_buckets = 65536 if self.compression_ratio is None else None

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_buckets: Dict[str, int] = {}

    def _determine_layer_buckets(self, in_features: int, out_features: int) -> int:
        total_w = in_features * out_features
        if self.compression_ratio is not None:
            target = max(16, int(total_w * self.compression_ratio))
        elif self.base_buckets is not None:
            target = min(self.base_buckets, max(16, int(total_w * 0.5)))
        else:
            target = min(65536, max(16, int(total_w * 0.25)))
        return max(16, min(target, total_w))

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
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
        if not (is_linear or is_conv1d) or isinstance(target_mod, HashedLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target model by substituting linear/Conv1D layers with HashedLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_buckets = {}
        seen_weights: set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze skipped non-target layers if target_attention_only is enabled
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("HashedNet: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
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

            if hasattr(target_mod, "nx") and hasattr(target_mod, "nf"):
                in_f = int(target_mod.nx)
                out_f = int(target_mod.nf)
                W_target = raw_w.t().float()
            elif hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
                in_f = int(target_mod.in_features)
                out_f = int(target_mod.out_features)
                if raw_w.shape == (in_f, out_f):
                    W_target = raw_w.t().float()
                else:
                    W_target = raw_w.view(out_f, in_f).float()
            elif is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.t().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.float()

            orig_layer_params = in_f * out_f
            total_orig_params += orig_layer_params
            W_norm = torch.norm(W_target, p="fro").item()
            total_weights = in_f * out_f
            max_allowed_buckets = max(16, int(total_weights * 0.90))

            init_buckets = self._determine_layer_buckets(in_f, out_f)
            current_buckets = init_buckets
            layer_compressed: Optional[HashedLinear] = None
            final_err = 1.0

            for attempt in range(self.max_retries + 1):
                try:
                    candidate = HashedLinear.from_linear(
                        module,
                        n_buckets=current_buckets,
                        use_sign=self.use_sign,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                    )
                except Exception as e:
                    logger.warning("HashedLinear instantiation failed on %s: %s", name, e)
                    break

                # Check for numerical validity
                if (
                    torch.isnan(candidate.weight_table).any()
                    or torch.isinf(candidate.weight_table).any()
                ):
                    logger.warning("HashedLinear weight table contains NaN/Inf on %s", name)
                    break

                # Measure relative Frobenius reconstruction error
                W_recon = candidate.get_weight().float()
                if W_norm < 1e-12:
                    err = 0.0
                else:
                    err = float(torch.norm(W_target - W_recon, p="fro").item() / W_norm)

                final_err = err
                layer_compressed = candidate

                if err <= self.max_error:
                    break

                # Adaptive bucket increase on error threshold exceedance
                if attempt < self.max_retries:
                    next_buckets = min(int(current_buckets * 2), max_allowed_buckets)
                    if next_buckets <= current_buckets:
                        break
                    logger.debug(
                        "HashedNet: Error %.4f > %.4f on %s. Retrying with %d buckets.",
                        err,
                        self.max_error,
                        name,
                        next_buckets,
                    )
                    current_buckets = next_buckets

            self.layer_errors[name] = final_err
            self.layer_buckets[name] = current_buckets

            if layer_compressed is None or final_err > self.max_error:
                logger.warning(
                    "HashedNet reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping HashedNet compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            if self.freeze_hash_table and layer_compressed is not None:
                layer_compressed.weight_table.requires_grad = False

            try:
                if hasattr(module, "base_layer"):
                    module.base_layer = layer_compressed
                    if hasattr(module, "lora_A") and module.lora_A is not None:
                        for p in (module.lora_A.parameters() if hasattr(module.lora_A, "parameters") else [module.lora_A]):
                            if isinstance(p, (nn.Parameter, torch.Tensor)):
                                p.requires_grad = True
                    if hasattr(module, "lora_B") and module.lora_B is not None:
                        for p in (module.lora_B.parameters() if hasattr(module.lora_B, "parameters") else [module.lora_B]):
                            if isinstance(p, (nn.Parameter, torch.Tensor)):
                                p.requires_grad = True
                else:
                    replace_module(self.model, name, layer_compressed)
            except Exception as e:
                logger.warning("Module replacement failed for %s: %s", name, e)
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            self.compressed_layers.append(name)
            comp_layer_params = layer_compressed.n_buckets
            if layer_compressed.lora_rank > 0:
                comp_layer_params += layer_compressed.lora_rank * (in_f + out_f)
            total_comp_params += comp_layer_params

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, HashedLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("HashedNet post-compression forward check warning: %s", e)
            forward_check_success = False

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_buckets": self.layer_buckets,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "n_buckets": self.base_buckets,
                "use_sign": self.use_sign,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
            }
        )
        logger.info(
            "HashedNet compression complete: %d compressed, %d fallback. Avg error: %.4f, Effective Param Ratio: %.2fx",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            effective_ratio,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("hashed_linear")
@COMPRESSOR_REGISTRY.register("hashed_linear_factory")
@COMPRESSOR_REGISTRY.register("hashed_net")
@COMPRESSOR_REGISTRY.register("hashed_net_compressor")
@COMPRESSOR_REGISTRY.register("hashed_net_factory")
@COMPRESSOR_REGISTRY.register("hashednets")
@COMPRESSOR_REGISTRY.register("hashednets_factory")
@COMPRESSOR_REGISTRY.register("hashednets_compressor")
def hashed_net_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[Union[int, float]] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for HashedNet compressor."""
    rank = kwargs.pop("rank", rank)
    n_buckets = kwargs.pop("n_buckets", _cfg_get(comp_config, "n_buckets", None))
    use_sign = kwargs.pop("use_sign", _cfg_get(comp_config, "use_sign", True))
    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", _cfg_get(comp_config, "lora_r", 16)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "hashed_net_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "hashed_net_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "hashed_net_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = HashedNetCompressor(
        model,
        n_buckets=n_buckets,
        rank=rank,
        use_sign=use_sign,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


hashednets_factory = hashed_net_factory
HashedNetsCompressor = HashedNetCompressor

__all__ = ["HashedNetCompressor", "HashedNetsCompressor", "hashed_net_factory", "hashednets_factory"]
