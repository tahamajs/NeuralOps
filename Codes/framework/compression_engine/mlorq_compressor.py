"""
MLoRQ: Multi-Layer Optimal Rank & Quantisation under a Total Memory Budget.

Registered as 'mlorq', 'mlorq_compressor', 'mlorq_factory'.

This version stores U, S, Vt as PACKED INT tensors (bits ∈ {2,4,8}), so the
memory budget formula `(out*r + r*in + r) * bits` is exact. The forward pass
dequantizes on the fly and never materializes the dense weight.

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
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ===========================================================================
# Bit packing helpers
# ===========================================================================
def _quantize_packed(t: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """
    Symmetric per-tensor quantization with bit packing.

    Returns:
        packed: 1-D int8 tensor (bit-packed)
        scale:  scalar float32 tensor
        shape:  original shape
    """
    orig_shape = t.shape
    flat = t.reshape(-1).float()
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.int8, device=t.device), torch.tensor(1.0, device=t.device), orig_shape

    if bits >= 8:
        max_abs = flat.abs().max().clamp_min(1e-8)
        scale = max_abs / 127.0
        q = torch.round(flat / scale).clamp(-128, 127).to(torch.int8)
        return q, scale, orig_shape

    qmax = (1 << (bits - 1)) - 1
    max_abs = flat.abs().max().clamp_min(1e-8)
    scale = max_abs / float(qmax)
    q = torch.round(flat / scale).clamp(-qmax - 1, qmax).to(torch.int32)

    # shift to unsigned
    offset = 1 << (bits - 1)
    q_unsigned = (q + offset).to(torch.int32)

    per_byte = 8 // bits
    n = q_unsigned.numel()
    pad = (-n) % per_byte
    if pad > 0:
        q_unsigned = torch.cat([q_unsigned, torch.zeros(pad, dtype=torch.int32, device=q_unsigned.device)])
    q_unsigned = q_unsigned.view(-1, per_byte)

    packed = torch.zeros(q_unsigned.shape[0], dtype=torch.int8, device=q_unsigned.device)
    for i in range(per_byte):
        packed = (packed << bits) | q_unsigned[:, i].to(torch.int8)
    return packed, scale, orig_shape


def _dequantize_packed(
    packed: torch.Tensor,
    bits: int,
    scale: torch.Tensor,
    orig_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bits >= 8:
        return (packed.float() * scale).to(device=device, dtype=dtype).reshape(orig_shape)

    per_byte = 8 // bits
    n = packed.numel() * per_byte
    mask = (1 << bits) - 1
    out = torch.zeros(n, dtype=torch.int32, device=packed.device)
    for i in range(per_byte):
        shift = bits * (per_byte - 1 - i)
        vals = (packed.to(torch.int32) >> shift) & mask
        out[i::per_byte] = vals

    offset = 1 << (bits - 1)
    q = out - offset
    # trim padding
    total = int(torch.tensor(orig_shape).prod().item())
    q = q[:total]
    return (q.float() * scale).to(device=device, dtype=dtype).reshape(orig_shape)


# ===========================================================================
# Compressed linear layer
# ===========================================================================
class MLoRQLoRALinear(nn.Module):
    """
    Low-rank quantized linear layer:

        y = (x @ Vt) * S @ U^T + bias + LoRA(x)

    where U, S, Vt are stored as PACKED int8 tensors with per-tensor scales.
    """
    is_custom_attention: bool = False  # MLR-FIX-01

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bits: int,
        U_packed: torch.Tensor,
        U_scale: torch.Tensor,
        U_shape: torch.Size,
        S_packed: torch.Tensor,
        S_scale: torch.Tensor,
        S_shape: torch.Size,
        Vt_packed: torch.Tensor,
        Vt_scale: torch.Tensor,
        Vt_shape: torch.Size,
        bias: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features
        self.rank = int(rank)
        self.bits = int(bits)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else U_packed.device
        target_dtype = dtype if dtype is not None else torch.float32

        self.register_buffer("U_packed", U_packed.to(target_device), persistent=True)
        self.register_buffer("U_scale", U_scale.to(target_device, dtype=torch.float32), persistent=True)
        self.register_buffer("S_packed", S_packed.to(target_device), persistent=True)
        self.register_buffer("S_scale", S_scale.to(target_device, dtype=torch.float32), persistent=True)
        self.register_buffer("Vt_packed", Vt_packed.to(target_device), persistent=True)
        self.register_buffer("Vt_scale", Vt_scale.to(target_device, dtype=torch.float32), persistent=True)

        self.U_shape = tuple(U_shape)
        self.S_shape = tuple(S_shape)
        self.Vt_shape = tuple(Vt_shape)

        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, self.in_features),
                            dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, self.lora_r),
                            dtype=torch.float32, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    def _dequantize(self, dtype: torch.dtype, device: torch.device):
        U = _dequantize_packed(
            self.U_packed, self.bits, self.U_scale, self.U_shape, device, dtype
        )
        S = _dequantize_packed(
            self.S_packed, self.bits, self.S_scale, self.S_shape, device, dtype
        )
        Vt = _dequantize_packed(
            self.Vt_packed, self.bits, self.Vt_scale, self.Vt_shape, device, dtype
        )
        return U, S, Vt

    @property
    def weight(self) -> torch.Tensor:
        dtype = self.U_scale.dtype
        device = self.U_scale.device
        U, S, Vt = self._dequantize(dtype, device)
        return (U * S.unsqueeze(0)) @ Vt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U, S, Vt = self._dequantize(x.dtype, x.device)
        # (B, in) @ (in, r) -> (B, r)
        x_v = F.linear(x, Vt)
        x_v = x_v * S
        out = F.linear(x_v, U)
        if self.bias is not None:
            out = out + self.bias.to(dtype=x.dtype, device=x.device)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)
        return out

    def storage_bytes(self) -> int:
        total = 0
        for name, buf in self.named_buffers():
            if buf is not None:
                total += buf.numel() * buf.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        if self.lora_A is not None:
            total += self.lora_A.numel() * self.lora_A.element_size()
        if self.lora_B is not None:
            total += self.lora_B.numel() * self.lora_B.element_size()
        return total

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        U: torch.Tensor,
        S: torch.Tensor,
        Vt: torch.Tensor,
        bits: int,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> MLoRQLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        bias = (
            base_layer.bias.detach()
            if getattr(base_layer, "bias", None) is not None
            else None
        )
        in_features = int(getattr(base_layer, "in_features", Vt.shape[1]))
        out_features = int(getattr(base_layer, "out_features", U.shape[0]))
        rank = int(min(U.shape[1], Vt.shape[0], S.numel()))

        # quantize + pack
        U_packed, U_scale, U_shape = _quantize_packed(U, bits)
        S_packed, S_scale, S_shape = _quantize_packed(S, bits)
        Vt_packed, Vt_scale, Vt_shape = _quantize_packed(Vt, bits)

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bits=bits,
            U_packed=U_packed,
            U_scale=U_scale,
            U_shape=U_shape,
            S_packed=S_packed,
            S_scale=S_scale,
            S_shape=S_shape,
            Vt_packed=Vt_packed,
            Vt_scale=Vt_scale,
            Vt_shape=Vt_shape,
            bias=bias,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=U.device,
            dtype=U.dtype,
        )

        # Preserve existing LoRA weights if present
        if (hasattr(linear, "lora_A") and hasattr(linear, "lora_B")
                and layer.lora_A is not None and layer.lora_B is not None):
            with torch.no_grad():
                src_A = (linear.lora_A.weight.data
                         if hasattr(linear.lora_A, "weight")
                         else linear.lora_A.data)
                src_B = (linear.lora_B.weight.data
                         if hasattr(linear.lora_B, "weight")
                         else linear.lora_B.data)
                if src_A.shape == layer.lora_A.shape and src_B.shape == layer.lora_B.shape:
                    layer.lora_A.copy_(src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype))
                    layer.lora_B.copy_(src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype))
            layer.lora_A.requires_grad = True
            layer.lora_B.requires_grad = True

        return layer


# ===========================================================================
# Config helper
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "self_attn",
}


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("mlorq")
@COMPRESSOR_REGISTRY.register("mlorq_compressor")
class MLoRQCompressor(BaseCompressor):
    """
    MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget.
    """
    is_custom_attention: bool = False  # MLR-FIX-01

    def __init__(
        self,
        model: nn.Module,
        total_memory_budget_mb: float = 500,
        bits_candidates: list[int] | None = None,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        min_rank: int = 4,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mlorq", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            total_memory_budget_mb = _cfg_get(
                cfg, "mlorq_budget_mb", _cfg_get(cfg, "total_memory_budget_mb", total_memory_budget_mb)
            )
            bits_candidates = _cfg_get(
                cfg, "mlorq_bits_candidates", _cfg_get(cfg, "bits_candidates", bits_candidates)
            )
            max_error = _cfg_get(cfg, "mlorq_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "mlorq_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.total_budget_bits = float(total_memory_budget_mb) * 8 * 1024 * 1024  # bits
        self.bits_candidates = sorted(set(int(b) for b in (bits_candidates or [4, 8])))
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.min_rank = max(1, int(min_rank))
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.layer_bits: dict[str, int] = {}
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn")):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in (
            "classifier", "score", "lm_head", "pooler", "embedding",
            "layernorm", "rmsnorm", "batchnorm",
        )):
            return False
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding, MLoRQLoRALinear)):
            return False

        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only and not any(k in name_l for k in _ATTN_NAMES):
            return False

        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    @staticmethod
    def _quantize_float(t: torch.Tensor, bits: int) -> torch.Tensor:
        """Non-packed quantization for error estimation."""
        if bits >= 32:
            return t.float()
        tf = t.float()
        max_abs = tf.abs().max().clamp_min(1e-8)
        if bits >= 8:
            scale = max_abs / 127.0
            return torch.round(tf / scale).clamp(-128, 127) * scale
        qmax = (1 << (bits - 1)) - 1
        scale = max_abs / float(qmax)
        return torch.round(tf / scale).clamp(-qmax - 1, qmax) * scale

    # ------------------------------------------------------------------
    def _generate_candidates(
        self, W: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Generate (rank, bits, error, cost_bits, U, S, Vt) candidates for a weight."""
        out_f, in_f = W.shape
        max_rank = min(out_f, in_f)
        device = W.device

        # Randomized SVD for large matrices (MLR-FIX-09)
        q = min(max_rank + 10, max_rank)
        if W.numel() > 1_000_000:
            try:
                U, S, V = torch.svd_lowrank(W.float(), q=q, niter=4)
                Vt = V.t().contiguous()
            except Exception:
                U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        else:
            U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)

        # Rank list (unique, <= max_rank)
        ranks = set()
        ranks.add(min(self.min_rank, max_rank))
        for frac in (8, 4, 2):
            r = max(self.min_rank, max_rank // frac)
            if r <= max_rank:
                ranks.add(r)
        ranks.add(max_rank)
        ranks = sorted(r for r in ranks if r >= 1 and r <= max_rank)

        candidates = []
        for bits in self.bits_candidates:
            for r in ranks:
                if r > U.shape[1]:
                    continue
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]

                U_q = self._quantize_float(U_r, bits)
                S_q = self._quantize_float(S_r, bits)
                Vt_q = self._quantize_float(Vt_r, bits)

                W_rec = U_q @ torch.diag(S_q) @ Vt_q
                w_norm = torch.norm(W, p="fro").clamp_min(1e-12).item()
                err = float((torch.norm(W - W_rec, p="fro") / w_norm).item())

                # cost in bits for packed storage
                cost_bits = (out_f * r + r * in_f + r) * bits
                candidates.append({
                    "rank": r,
                    "bits": bits,
                    "error": err,
                    "cost_bits": cost_bits,
                    "U": U_r,
                    "S": S_r,
                    "Vt": Vt_r,
                })
        candidates.sort(key=lambda c: c["cost_bits"])
        return candidates

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_bits = {}
        seen_weights: set[int] = set()

        # ---------- 1. Collect target layers ----------
        layer_infos: list[dict[str, Any]] = []
        total_orig_bits = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            base = getattr(module, "base_layer", module)
            w = getattr(base, "weight", None)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = w.data
            orig_bits = raw_w.numel() * raw_w.element_size() * 8
            total_orig_bits += orig_bits

            cls_name = base.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(base, "nf")
            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()

            layer_infos.append({
                "name": name,
                "module": module,
                "base": base,
                "W": W,
                "shape": W.shape,
                "is_conv1d": is_conv1d,
                "device": raw_w.device,
                "dtype": raw_w.dtype,
                "orig_bits": orig_bits,
            })

        if not layer_infos:
            logger.warning("MLoRQ: no target layers found.")
            stats = self.get_compression_stats()
            stats.update({"num_compressed": 0, "method": "mlorq"})
            self.stats = stats
            return self.model, stats

        # ---------- 2. Generate candidates ----------
        for info in layer_infos:
            cands = self._generate_candidates(info["W"])
            # Filter by max_error
            valid = [c for c in cands if c["error"] <= self.max_error]
            if not valid:
                # fallback original
                info["candidates"] = []
                info["chosen"] = None
                info["current_cost"] = info["orig_bits"]
                info["current_error"] = 0.0
            else:
                info["candidates"] = valid
                info["chosen"] = valid[0]  # cheapest valid
                info["current_cost"] = valid[0]["cost_bits"]
                info["current_error"] = valid[0]["error"]

        # ---------- 3. Greedy budget allocation (MLR-FIX-04) ----------
        total_cost = sum(info["current_cost"] for info in layer_infos)
        budget = self.total_budget_bits

        # Downgrade if over budget
        while total_cost > budget:
            best_info = None
            best_ratio = float("inf")
            best_new = None
            for info in layer_infos:
                if info["chosen"] is None:
                    continue
                # find cheaper candidate than current
                cheaper = [c for c in info["candidates"] if c["cost_bits"] < info["current_cost"]]
                if not cheaper:
                    continue
                # choose the cheapest cheaper candidate? Actually we want the one with smallest error increase per bit saved.
                # We can evaluate each cheaper candidate, but for simplicity pick the next cheaper (max cost < current).
                next_cheaper = max(cheaper, key=lambda c: c["cost_bits"])
                delta_cost = info["current_cost"] - next_cheaper["cost_bits"]
                delta_err = next_cheaper["error"] - info["current_error"]
                if delta_cost <= 0:
                    continue
                ratio = delta_err / delta_cost
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_info = info
                    best_new = next_cheaper
            if best_info is None:
                break
            # apply downgrade
            best_info["chosen"] = best_new
            best_info["current_cost"] = best_new["cost_bits"]
            best_info["current_error"] = best_new["error"]
            total_cost = sum(info["current_cost"] for info in layer_infos)

        # Upgrade if under budget
        while total_cost < budget:
            best_info = None
            best_ratio = -1.0
            best_new = None
            for info in layer_infos:
                if info["chosen"] is None:
                    continue
                # find more expensive candidate
                more = [c for c in info["candidates"] if c["cost_bits"] > info["current_cost"]]
                if not more:
                    continue
                next_more = min(more, key=lambda c: c["cost_bits"])
                delta_cost = next_more["cost_bits"] - info["current_cost"]
                delta_err = info["current_error"] - next_more["error"]
                if delta_cost <= 0:
                    continue
                if total_cost + delta_cost > budget:
                    continue
                ratio = delta_err / delta_cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_info = info
                    best_new = next_more
            if best_info is None:
                break
            best_info["chosen"] = best_new
            best_info["current_cost"] = best_new["cost_bits"]
            best_info["current_error"] = best_new["error"]
            total_cost = sum(info["current_cost"] for info in layer_infos)

        # ---------- 4. Apply chosen candidates ----------
        total_comp_bytes = 0
        for info in layer_infos:
            name = info["name"]
            module = info["module"]
            chosen = info["chosen"]

            if chosen is None:
                # fallback original
                self.fallback_layers.append(name)
                total_comp_bytes += info["orig_bits"] // 8
                continue

            try:
                new_layer = MLoRQLoRALinear.from_linear(
                    module,
                    U=chosen["U"],
                    S=chosen["S"],
                    Vt=chosen["Vt"],
                    bits=chosen["bits"],
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                # MLR-FIX-05: preserve wrapper
                if hasattr(module, "base_layer"):
                    module.base_layer = new_layer
                else:
                    replace_module(self.model, name, new_layer)

                self.compressed_layers.append(name)
                self.layer_errors[name] = float(chosen["error"])
                self.layer_ranks[name] = int(chosen["rank"])
                self.layer_bits[name] = int(chosen["bits"])
                total_comp_bytes += new_layer.storage_bytes()
            except Exception as e:
                logger.warning("MLoRQ: failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                total_comp_bytes += info["orig_bits"] // 8

        # ---------- 5. Forward sanity check (MLR-FIX-06) ----------
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        cfg = getattr(self.model, "config", None)
                        hidden = getattr(cfg, "hidden_size", None) if cfg else None
                        if hidden is None:
                            first_lin = next(
                                (m for m in self.model.modules() if isinstance(m, nn.Linear)), None
                            )
                            hidden = getattr(first_lin, "in_features", 8)
                        dt = next(self.model.parameters()).dtype
                        dummy_feat = torch.randn(1, 8, hidden, device=dev, dtype=dt)
                        self.model(dummy_feat)
        except Exception as e:
            logger.warning("MLoRQ forward sanity check failed: %s", e)
            forward_check_success = False

        # ---------- 6. Stats ----------
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors else 0.0
        )
        storage_ratio = (
            total_orig_bits / 8 / max(1, total_comp_bytes)
            if total_comp_bytes > 0 else 1.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "layer_bits": self.layer_bits,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "total_budget_bits": self.total_budget_bits,
            "total_used_bits": total_cost,
            "budget_utilization_pct": round(100.0 * total_cost / max(1, self.total_budget_bits), 2),
            "orig_storage_bytes": total_orig_bits // 8,
            "comp_storage_bytes": total_comp_bytes,
            "storage_compression_ratio": round(storage_ratio, 3),
            "storage_reduction_pct": round(100.0 * (1.0 - total_comp_bytes / max(1, total_orig_bits // 8)), 2),
            "forward_check_success": forward_check_success,
            "method": "mlorq",
        })
        self.stats = stats
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("mlorq_factory")
def mlorq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    total_budget_mb = kwargs.pop(
        "total_memory_budget_mb",
        _cfg_get(comp_config, "mlorq_budget_mb", 500),
    )
    bits_candidates = kwargs.pop(
        "bits_candidates",
        _cfg_get(comp_config, "mlorq_bits_candidates", [4, 8]),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "mlorq_max_error", _cfg_get(comp_config, "max_error", 0.50)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "mlorq_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "target_attention_only", False),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = MLoRQCompressor(
        model,
        total_memory_budget_mb=total_budget_mb,
        bits_candidates=bits_candidates,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "MLoRQLoRALinear",
    "MLoRQCompressor",
    "mlorq_factory",
]
