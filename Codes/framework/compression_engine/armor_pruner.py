"""
ARMOR: Adaptive Resilient Matrix Orthogonal Rank Pruning (2:4 Sparse Core + Low-Rank Correction).
Registered in COMPRESSOR_REGISTRY as 'armor_pruner' and 'armor'.

Mathematical Formulation:
1. Resilient First-Order Taylor Saliency (handling zero-grad gracefully):
   $$S_{ij} = \\begin{cases} |W_{ij} \\odot g_{ij}| & \\text{if } g \\text{ is valid and } \\|g\\| > 0 \\\\ |W_{ij}| \\cdot \\mathbb{E}[|x_j|] & \\text{if activations available} \\\\ |W_{ij}| \\cdot (\\sigma(W) + 1) & \\text{fallback} \\end{cases}$$
2. 2:4 Semi-Structured Pruning Mask:
   $$\\mathcal{M} = \\Pi_{2:4}(S), \\quad W_{\\text{sparse}} = W \\odot \\mathcal{M}$$
3. Residual Low-Rank Decomposition:
   $$R = W - W_{\\text{sparse}} \\approx B A, \\quad A \\in \\mathbb{R}^{r \\times d_{\\text{in}}}, \\; B \\in \\mathbb{R}^{d_{\\text{out}} \\times r}$$
4. Decoupled Dual-Branch Forward with Independent Gradient Flow:
   $$y = x (W_{\\text{sparse}} \\odot \\mathcal{M})^\\top + (x A^\\top) B^\\top + b$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


def compute_resilient_taylor_saliency(
    weight: torch.Tensor,
    grad: Optional[torch.Tensor] = None,
    activations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes Taylor parameter saliency handling zero-grad, NaN/Inf, and absent activations gracefully.
    """
    W = weight.float()

    # 1. Check if gradient is provided and non-zero
    if grad is not None and torch.is_tensor(grad):
        g = grad.float()
        if torch.isfinite(g).all() and torch.count_nonzero(g) > 0:
            return torch.abs(W * g)

    # 2. Check if activations are provided
    if activations is not None and torch.is_tensor(activations) and activations.numel() > 0:
        X = activations.reshape(-1, activations.shape[-1]).float()
        if X.shape[-1] == W.shape[1]:
            act_scale = torch.mean(torch.abs(X), dim=0, keepdim=True).to(W.device)
            act_scale = torch.nan_to_num(act_scale, nan=1.0)
            return torch.abs(W) * act_scale

    # 3. Fallback: Magnitude weighted by standard deviation
    w_std = torch.std(W).item() if W.numel() > 1 else 1.0
    if not math.isfinite(w_std) or w_std <= 0.0:
        w_std = 1.0
    return torch.abs(W) * (1.0 + w_std)


class ARMORLinear(nn.Module):
    """
    Decoupled Sparse + Low-Rank Residual Layer with optional LoRA scaling.
    Maintains clean gradient flow to low-rank adapter branches while base sparse weights are frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.rank) if self.rank > 0 else 1.0

        # Base sparse weight is frozen for parameter-efficient adaptation
        self.sparse_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.register_buffer("sparse_mask", torch.ones(out_features, in_features, dtype=dtype, device=device))

        # Trainable low-rank factorized parameters
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        mask: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        lora_alpha: float = 1.0,
    ) -> ARMORLinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            w_data = raw_w.t()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            w_data = raw_w

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=lora_A.shape[0],
            bias=getattr(linear, "bias", None) is not None and linear.bias is not None,
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )
        layer.sparse_weight.data.copy_(w_data)
        layer.sparse_mask.copy_(mask.to(dtype=raw_w.dtype, device=raw_w.device))
        layer.lora_A.data.copy_(lora_A.to(dtype=raw_w.dtype, device=raw_w.device))
        layer.lora_B.data.copy_(lora_B.to(dtype=raw_w.dtype, device=raw_w.device))
        if getattr(linear, "bias", None) is not None and linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight for compatibility with standard PyTorch / HuggingFace layers."""
        return self.sparse_weight * self.sparse_mask + self.scaling * torch.matmul(self.lora_B, self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # 1. Sparse Core Forward
        w_eff = self.sparse_weight * self.sparse_mask
        out_sparse = F.linear(x_flat, w_eff)

        # 2. Low-Rank Residual Correction Forward with LoRA scaling
        out_lr = self.scaling * F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B).to(out_sparse.dtype)

        out = out_sparse + out_lr
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], self.out_features)


class ARMORPruner(BaseCompressor):
    """
    ARMOR Pruner: Combines 2:4 structured sparsity with low-rank error compensation
    and decoupled dual-branch gradient propagation.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        rank: int = 8,
        sparsity: float = 0.5,
        structured_24: bool = True,
        decouple: bool = True,
        recalibrate_ln: bool = True,
        lora_alpha: float = 1.0,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="armor_pruner", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        self.block_size = block_size
        self.rank = max(1, int(rank if rank is not None else getattr(conf, "armor_rank", getattr(conf, "rank", 8))))
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.structured_24 = structured_24
        self.decouple = decouple
        self.recalibrate_ln = recalibrate_ln
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else getattr(conf, "lora_alpha", 1.0))
        self.max_error = float(max_error if max_error is not None else getattr(conf, "max_error", 0.35))
        self.fallback_to_original = bool(fallback_to_original if fallback_to_original is not None else getattr(conf, "fallback_to_original", True))
        self.skip_attention_qkv = bool(skip_attention_qkv if skip_attention_qkv is not None else getattr(conf, "skip_attention_qkv", True))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

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

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Capture activations across linear layers."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(x.cpu())
            return hook

        from framework.utils.compat import should_compress_linear

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if is_linear_like and should_compress_linear(self.model, name) and not self._is_gqa_asymmetric(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("ARMOR calibration pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ARMOR pruning with low-rank error correction and decoupled gradient modules."""
        from framework.utils.compat import should_compress_linear

        act_map = self._collect_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        total_correction_energy = 0.0
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, ARMORLinear):
                continue

            if not should_compress_linear(self.model, name):
                continue

            if self._is_gqa_asymmetric(name, module):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            w_id = id(module.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = module.weight.data
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            device = raw_w.device
            dtype = raw_w.dtype
            grad = getattr(module.weight, "grad", None)
            act = act_map.get(name, None)

            try:
                # 1. Resilient Saliency Computation
                saliency = compute_resilient_taylor_saliency(W, grad=grad, activations=act)

                # 2. Structured 2:4 or magnitude sparse mask
                if self.structured_24:
                    W_sparse, mask = apply_24_structured_mask(W, saliency=saliency, dim=1)
                else:
                    flat = saliency.reshape(-1)
                    k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    mask = (saliency >= thr).float()
                    W_sparse = W * mask

                # 3. Residual Low-Rank Error SVD Correction
                residual = W - W_sparse
                r = min(self.rank, residual.shape[0], residual.shape[1])
                try:
                    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                    lora_B = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
                    lora_A = (sqrt_S.unsqueeze(1) * Vh[:r, :]).to(dtype=dtype, device=device)
                    correction = torch.matmul(lora_B, lora_A)
                    total_correction_energy += torch.norm(correction).item()
                except Exception as e:
                    logger.debug("SVD failed on residual in %s: %s", name, e)
                    lora_A = torch.zeros(r, W.shape[1], dtype=dtype, device=device)
                    lora_B = torch.zeros(W.shape[0], r, dtype=dtype, device=device)

                # Measure reconstruction error
                scaling = (self.lora_alpha / r) if r > 0 else 1.0
                W_recon = W_sparse + scaling * torch.matmul(lora_B, lora_A)
                W_norm = torch.norm(W, p="fro").clamp_min(1e-8)
                err = float(torch.norm(W - W_recon, p="fro").item() / W_norm)
                self.layer_errors[name] = err

                if err > self.max_error:
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        continue

                # 4. Module Replacement: Decoupled Dual-Branch or Fused
                if self.decouple:
                    armor_layer = ARMORLinear.from_linear(
                        module,
                        mask=mask,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        lora_alpha=self.lora_alpha,
                    )
                    if name == "":
                        self.model = armor_layer
                    else:
                        replace_module(self.model, name, armor_layer)
                else:
                    if is_conv1d:
                        module.weight.data.copy_(W_recon.t().to(dtype=dtype, device=device))
                    else:
                        module.weight.data.copy_(W_recon.to(dtype=dtype, device=device))

                self.compressed_layers.append(name)
            except Exception as e:
                logger.warning("ARMOR pruning failed on %s: %s", name, e)
                self.fallback_layers.append(name)

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, ARMORLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("ARMOR post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats.update({
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "pruned_layers": len(self.compressed_layers),
            "rank": self.rank,
            "structured_24": self.structured_24,
            "decouple": self.decouple,
            "lora_alpha": self.lora_alpha,
            "total_correction_energy": total_correction_energy,
            "forward_check_success": forward_check_success,
            "method": "armor_sparse_plus_lowrank_decoupled",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("armor_pruner")
@COMPRESSOR_REGISTRY.register("armor")
@COMPRESSOR_REGISTRY.register("armor_compressor")
@COMPRESSOR_REGISTRY.register("armor_factory")
@COMPRESSOR_REGISTRY.register("armor_pruner_factory")
@COMPRESSOR_REGISTRY.register("armor_compressor_factory")
def armor_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ARMOR pruner."""
    block_size = getattr(comp_config, "armor_block_size", getattr(comp_config, "block_size", 4)) if comp_config else kwargs.get("block_size", 4)
    rank = getattr(comp_config, "armor_rank", getattr(comp_config, "rank", 8)) if comp_config else kwargs.get("rank", 8)
    structured_24 = getattr(comp_config, "armor_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)
    decouple = getattr(comp_config, "armor_decouple", kwargs.get("decouple", True))
    recalibrate_ln = getattr(comp_config, "armor_recalibrate_ln", kwargs.get("recalibrate_ln", True))
    lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 1.0))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = ARMORPruner(
        model,
        block_size=block_size,
        rank=rank,
        structured_24=structured_24,
        decouple=decouple,
        recalibrate_ln=recalibrate_ln,
        lora_alpha=lora_alpha,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model

__all__ = [
    "compute_resilient_taylor_saliency",
    "ARMORLinear",
    "ARMORPruner",
    "armor_pruner_factory",
]
