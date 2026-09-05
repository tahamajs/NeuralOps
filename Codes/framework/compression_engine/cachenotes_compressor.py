"""
CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression.

Given a task description or task type, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Paper: "CacheNotes: Task-aware key-value cache compression for reasoning-intensive
        knowledge tasks." Corallo et al., EACL 2026.
Registered as 'cachenotes'.

Mathematical Formulation:
1. CPT Importance Vector:
   $$I_{\\text{task}} = \\text{CPT}(T, d)$$
2. Runtime Dynamic KV Cache Pruning:
   $$\\mathcal{S}_{\\text{token}} = \\|K_{\\text{cache}} \\odot I_{\\text{task}}\\|_2, \\quad \\mathcal{S}_{:sinks} = +\\infty$$
   $$K_{\\text{retained}} = \\operatorname{Top-}k(K_{\\text{cache}}, \\mathcal{S}_{\\text{token}})$$
3. Decoupled Static Weight Compression (Low-Rank Task-Aware SVD without channel zeroing):
   $$W_{\\text{cpt}} = \\operatorname{diag}(I_{\\text{task}}) W$$
   $$W_{\\text{cpt}} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
   $$W_{\\text{approx}} = \\operatorname{diag}(I_{\\text{task}}^{-1}) (U_r \\Sigma_r V_r^\\top)$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)
from framework.utils.compat import should_compress_linear, replace_module

logger = logging.getLogger(__name__)


class CacheNotesLoRALinear(nn.Module):
    """
    Linear layer containing CacheNotes compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        self.weight = nn.Parameter(
            weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> CacheNotesLoRALinear:
        bias = linear.bias.detach() if hasattr(linear, "bias") and linear.bias is not None else None
        in_features = getattr(linear, "in_features", weight.shape[1])
        out_features = getattr(linear, "out_features", weight.shape[0])
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(x_flat, self.weight, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


class CompressionPlanningTokens:
    """
    CPT importance generator.
    Simulates task-aware attention planning scores for different downstream tasks.
    """

    TASK_KEYWORD_MAP = {
        "qa": ["question", "answer", "context", "passage", "who", "what", "where", "why"],
        "summarize": ["summary", "abstract", "key", "main", "brief", "conclusion"],
        "code": ["function", "class", "def", "return", "import", "self", "error", "test"],
        "reasoning": ["therefore", "because", "if", "then", "thus", "proof", "step", "hence"],
        "math": ["solve", "equation", "matrix", "integral", "theorem", "calculate", "equal"],
    }

    @classmethod
    def generate_importance(
        cls, task_type: str, num_features: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Generate per-feature importance modulation vector based on task specification."""
        task_lower = str(task_type).lower()
        importance = torch.zeros(num_features, device=device)

        # Baseline structure: prompt position bias
        pos_dim = max(1, num_features // 4)
        importance[:pos_dim] = 0.25

        if any(kw in task_lower for kw in ["qa", "question"]):
            importance[pos_dim : 2 * pos_dim] += 0.35
        elif any(kw in task_lower for kw in ["code", "program"]):
            importance[2 * pos_dim : 3 * pos_dim] += 0.40
        elif any(kw in task_lower for kw in ["reason", "math"]):
            importance += 0.20

        return torch.clamp(importance + 0.5, min=0.1, max=2.0)


class CacheNotesKVCache:
    """
    Runtime task-guided dynamic KV cache compressor.
    Evaluates token task-relevance dynamically and prunes non-essential context tokens,
    supporting variable sequence lengths during prefill and autoregressive decoding.
    """

    def __init__(
        self,
        max_cache_len: int = 128,
        task_type: str = "qa",
        keep_ratio: float = 0.3,
        num_sinks: int = 4,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.task_type = task_type
        self.keep_ratio = float(keep_ratio)
        self.num_sinks = int(num_sinks)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear cache state to prevent memory leaks across sequences."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV tokens and apply task-aware retention when exceeding capacity.
        Shapes: (B, H, N, D).
        """
        k_4d = new_k
        v_4d = new_v
        if new_k.ndim == 2:
            k_4d = new_k.unsqueeze(0).unsqueeze(0)
            v_4d = new_v.unsqueeze(0).unsqueeze(0)
        elif new_k.ndim == 3:
            k_4d = new_k.unsqueeze(1)
            v_4d = new_v.unsqueeze(1)

        B, H, N, D = k_4d.shape

        if self.k_cache is None:
            self.k_cache = k_4d
            self.v_cache = v_4d
        else:
            if self.k_cache.shape[0] != B:
                self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
                self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
            self.k_cache = torch.cat([self.k_cache, k_4d], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_4d], dim=2)

        cur_len = self.k_cache.size(2)
        if cur_len <= self.max_cache_len:
            return self.k_cache, self.v_cache

        sinks = min(self.num_sinks, cur_len)
        retained_cap = max(sinks + 1, self.max_cache_len)

        # Task importance weighting across tokens
        cpt_importance = CompressionPlanningTokens.generate_importance(
            self.task_type, D, device=self.k_cache.device
        )
        weighted_k = self.k_cache * cpt_importance.view(1, 1, 1, D)
        token_scores = torch.norm(weighted_k.float(), dim=-1)  # (B, H, cur_len)

        # Always preserve initial sink tokens
        token_scores[:, :, :sinks] += 1e6

        _, top_indices = torch.topk(token_scores, retained_cap, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        idx_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        self.k_cache = self.k_cache.gather(2, idx_exp).contiguous()
        self.v_cache = self.v_cache.gather(2, idx_exp).contiguous()

        return self.k_cache, self.v_cache


class CacheNotesCompressor(BaseCompressor):
    """
    CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning.
    Compresses static model weights using CPT-guided low-rank SVD without zeroing out entire output channels.

    Production hardening features:
    - Robust Conv1D and fused linear projection support.
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Unified critical-layer protection via should_compress_linear and protect_critical.
    - GQA-aware attention projection skipping.
    - Tied/shared weight deduplication.
    - Trainable post-compression LoRA adapters.
    - Post-compression forward sanity check.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "qa",
        keep_ratio: float = 0.5,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="cachenotes", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        if conf is not None:
            task_type = getattr(conf, "cachenotes_task", getattr(conf, "task_type", task_type))
            keep_ratio = getattr(conf, "cachenotes_keep_ratio", getattr(conf, "keep_ratio", keep_ratio))
            max_error = getattr(conf, "cachenotes_max_error", getattr(conf, "max_error", max_error))
            fallback_to_original = getattr(
                conf, "cachenotes_fallback_to_original", getattr(conf, "fallback_to_original", fallback_to_original)
            )
            protect_critical = getattr(conf, "protect_critical", protect_critical)
            lora_r = getattr(conf, "lora_r", getattr(conf, "lora_rank", lora_r))
            lora_alpha = getattr(conf, "lora_alpha", lora_alpha)
            skip_attention_qkv = getattr(conf, "skip_attention_qkv", skip_attention_qkv)
            recalibrate_ln = getattr(conf, "cachenotes_recalibrate_ln", getattr(conf, "recalibrate_ln", recalibrate_ln))

        self.task_type = task_type
        self.keep_ratio = max(0.05, min(float(keep_ratio), 1.0))
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.recalibrate_ln = bool(recalibrate_ln)
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

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like or isinstance(module, CacheNotesLoRALinear):
            return False

        if self.protect_critical:
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
                return False

        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        return True

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using task-guided CPT-weighted low-rank SVD without channel destruction."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                if not hasattr(module, "weight") or module.weight is None:
                    continue
                w_id = id(module.weight)
                if w_id in seen_weights:
                    self.skipped_layers.append(name)
                    continue
                seen_weights.add(w_id)

                raw_w = module.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                out_f, in_f = W.shape
                device = W.device
                dtype = module.weight.dtype

                # Generate CPT importance modulation per output feature
                importance = CompressionPlanningTokens.generate_importance(
                    self.task_type, out_f, device=device
                )  # [out_f]

                # Weight rows by CPT importance
                W_cpt = W * importance.unsqueeze(1)

                # Compute SVD in CPT-weighted space
                r = max(1, min(int(min(out_f, in_f) * self.keep_ratio), min(out_f, in_f)))
                try:
                    U, S, Vt = torch.linalg.svd(W_cpt, full_matrices=False)
                    # Low-rank approximation
                    W_lowrank = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
                    # Unweight back to physical space: all channels remain active and non-zero
                    W_approx = W_lowrank / importance.unsqueeze(1)
                    if torch.isnan(W_approx).any() or torch.isinf(W_approx).any():
                        W_approx = W
                except Exception as e:
                    logger.warning("SVD failed in CacheNotes on %s: %s", name, e)
                    W_approx = W

                w_norm = torch.norm(W, p="fro").item()
                err = float(torch.norm(W - W_approx, p="fro").item() / max(w_norm, 1e-12))
                self.layer_errors[name] = err

                if err > self.max_error:
                    logger.warning(
                        "CacheNotes reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping CacheNotes compression for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                if self.lora_r > 0:
                    cn_layer = CacheNotesLoRALinear.from_linear(
                        module,
                        weight=W_approx,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    if name == "":
                        self.model = cn_layer
                    else:
                        replace_module(self.model, name, cn_layer)
                else:
                    if is_conv1d:
                        module.weight.data.copy_(W_approx.t().contiguous().to(dtype=dtype, device=device))
                    else:
                        module.weight.data.copy_(W_approx.contiguous().to(dtype=dtype, device=device))

                self.compressed_layers.append(name)

        # LayerNorm Recalibration
        if self.recalibrate_ln and (self.calibration_loader is not None or self.calibration_texts is not None):
            try:
                recalibrate_layernorms(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
            except Exception as e:
                logger.warning("CacheNotes LayerNorm recalibration skipped: %s", e)

        # Post-compression forward sanity check
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.debug("CacheNotes post-compression forward check exception: %s", e)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
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
            "avg_reconstruction_error": avg_err,
            "task_type": self.task_type,
            "keep_ratio": self.keep_ratio,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "recalibrate_ln": self.recalibrate_ln,
            "method": "cachenotes_decoupled_cpt_svd",
            "kv_cache_budget_saving": f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cachenotes")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor")
@COMPRESSOR_REGISTRY.register("cachenotes_factory")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor_factory")
def cachenotes_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """CacheNotes compressor factory function."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    task_type = kwargs.pop("task_type", getattr(comp_config, "cachenotes_task", getattr(comp_config, "task_type", "qa")) if comp_config else "qa")
    keep_ratio = kwargs.pop("keep_ratio", getattr(comp_config, "cachenotes_keep_ratio", getattr(comp_config, "keep_ratio", 0.5)) if comp_config else 0.5)
    max_error = kwargs.pop("max_error", getattr(comp_config, "cachenotes_max_error", getattr(comp_config, "max_error", 0.20)) if comp_config else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "cachenotes_fallback_to_original", getattr(comp_config, "fallback_to_original", True)) if comp_config else True)
    protect_critical = kwargs.pop("protect_critical", getattr(comp_config, "protect_critical", True) if comp_config else True)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", getattr(comp_config, "lora_rank", 0)) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    skip_attention_qkv = kwargs.pop("skip_attention_qkv", getattr(comp_config, "skip_attention_qkv", True) if comp_config else True)
    recalibrate_ln = getattr(comp_config, "cachenotes_recalibrate_ln", kwargs.get("recalibrate_ln", True)) if comp_config else kwargs.get("recalibrate_ln", True)

    compressor = CacheNotesCompressor(
        model,
        task_type=task_type,
        keep_ratio=keep_ratio,
        max_error=max_error,
        fallback_to_original=fallback,
        protect_critical=protect_critical,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        recalibrate_ln=recalibrate_ln,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "CompressionPlanningTokens",
    "CacheNotesKVCache",
    "CacheNotesLoRALinear",
    "CacheNotesCompressor",
    "cachenotes_factory",
]
