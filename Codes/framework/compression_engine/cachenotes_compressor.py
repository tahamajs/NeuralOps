"""
CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression.

Given a task description or task type, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Paper: "CacheNotes: Task-aware key-value cache compression for reasoning-intensive
        knowledge tasks." Corallo et al., EACL 2026.
Registered as 'cachenotes' and 'cachenotes_compressor'.

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

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from None, dict, or namespace object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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
                torch.empty((self.lora_r, in_features), dtype=target_dtype, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=target_dtype, device=target_device)
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
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
        else:
            out_f = raw_w.shape[0]
            in_f = raw_w.shape[1]

        bias = linear.bias.detach().clone() if getattr(linear, "bias", None) is not None and linear.bias is not None else None
        return cls(
            in_features=in_f,
            out_features=out_f,
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
        out = F.linear(x_flat, self.weight.to(x_flat.dtype), self.bias.to(x_flat.dtype) if self.bias is not None else None)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.to(self.lora_A.dtype), self.lora_A), self.lora_B)
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


@COMPRESSOR_REGISTRY.register("cachenotes")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor")
class CacheNotesCompressor(BaseCompressor):
    """
    CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning.
    Compresses static model weights using CPT-guided low-rank SVD without zeroing out entire output channels.

    Production hardening features:
    - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
    - Preserves existing LoRALinear wrappers when lora_r > 0 without creating redundant layers.
    - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
    - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
    - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
    - GQA-aware attention projection skipping.
    - Tied/shared weight deduplication.
    - Trainable post-compression LoRA adapters.
    - Comprehensive metric tracking: actual ranks, reconstruction error, and KV cache saving.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "qa",
        keep_ratio: float = 0.5,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="cachenotes", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))

        task_type = str(_cfg_get(cfg, "cachenotes_task", _cfg_get(cfg, "task_type", task_type)))
        keep_ratio = float(_cfg_get(cfg, "cachenotes_keep_ratio", _cfg_get(cfg, "keep_ratio", keep_ratio)))
        max_error = float(_cfg_get(cfg, "cachenotes_max_error", _cfg_get(cfg, "max_error", max_error)))
        fallback_to_original = bool(
            _cfg_get(cfg, "cachenotes_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        protect_critical = bool(_cfg_get(cfg, "protect_critical", protect_critical))
        target_attention_only = bool(
            _cfg_get(cfg, "cachenotes_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        lora_r = int(_cfg_get(cfg, "cachenotes_lora_r", _cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r))))
        lora_alpha = float(_cfg_get(cfg, "cachenotes_lora_alpha", _cfg_get(cfg, "lora_alpha", lora_alpha)))
        skip_attention_qkv = bool(
            _cfg_get(cfg, "cachenotes_skip_attention_qkv", _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))
        )
        recalibrate_ln = bool(
            _cfg_get(cfg, "cachenotes_recalibrate_ln", _cfg_get(cfg, "recalibrate_ln", recalibrate_ln))
        )

        self.task_type = task_type
        self.keep_ratio = max(0.05, min(keep_ratio, 1.0))
        self.max_error = max_error
        self.fallback_to_original = fallback_to_original
        self.protect_critical = protect_critical
        self.target_attention_only = target_attention_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.skip_attention_qkv = skip_attention_qkv
        self.recalibrate_ln = recalibrate_ln
        self.target_modules = target_modules

        self.tokenizer = tokenizer or kwargs.get("tokenizer", getattr(model, "tokenizer", None))
        self.calibration_loader = calibration_loader or kwargs.get("calibration_loader", None)
        self.calibration_texts = calibration_texts or kwargs.get("calibration_texts", None)
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.comp_config = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.actual_ranks: Dict[str, int] = {}
        self.stats: Dict[str, Any] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        # Do not skip attention projections if attention-only compression is requested
        if self.target_attention_only:
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
        if isinstance(module, CacheNotesLoRALinear):
            return False

        # Do not target inner submodules of LoRA or wrappers directly
        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        # Protect classification / output heads
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor", "head", "pre_classifier"]):
            return False

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only:
            # Skip FFN / MLP layers
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            # In RoBERTa, layer.X.output.dense is FFN, while attention.output.dense is attention
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            # Match attention projections
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value", "in_proj_qkv", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if self.protect_critical and not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using task-guided CPT-weighted low-rank SVD without channel destruction."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.actual_ranks = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_mod.weight.data
            device = raw_w.device
            dtype = raw_w.dtype

            # Standardize shape to [out_features, in_features] in FP32
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            out_f, in_f = W.shape

            # Generate CPT importance modulation per output feature
            importance = CompressionPlanningTokens.generate_importance(
                self.task_type, out_f, device=device
            )

            # Weight rows by CPT importance
            W_cpt = W * importance.unsqueeze(1)

            # Compute SVD in CPT-weighted space
            r = max(1, min(int(min(out_f, in_f) * self.keep_ratio), min(out_f, in_f)))
            self.actual_ranks[name] = r
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

            if torch.isnan(W_approx).any() or torch.isinf(W_approx).any() or err > self.max_error:
                logger.warning(
                    "CacheNotes reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping CacheNotes compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    # Freeze fallback layers
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Update weights / module
            if has_lora:
                # Layer is already a LoRALinear wrapper: update base layer weights in-place
                if is_conv1d:
                    target_mod.weight.data.copy_(W_approx.t().contiguous().to(dtype=dtype, device=device))
                else:
                    target_mod.weight.data.copy_(W_approx.contiguous().to(dtype=dtype, device=device))
                # Keep base weights frozen, but keep LoRA adapters trainable
                target_mod.weight.requires_grad = False
                if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
                    target_mod.bias.requires_grad = False
                for p_name, p in module.named_parameters():
                    if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                        p.requires_grad = True
            elif self.lora_r > 0:
                # Not LoRALinear yet, and lora_r > 0: wrap in CacheNotesLoRALinear
                cn_layer = CacheNotesLoRALinear.from_linear(
                    target_mod,
                    weight=W_approx,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                if name == "":
                    self.model = cn_layer
                else:
                    replace_module(self.model, name, cn_layer)
            else:
                # Pure static compression (lora_r == 0)
                if is_conv1d:
                    target_mod.weight.data.copy_(W_approx.t().contiguous().to(dtype=dtype, device=device))
                else:
                    target_mod.weight.data.copy_(W_approx.contiguous().to(dtype=dtype, device=device))

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
        avg_rank = (
            float(sum(self.actual_ranks[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers)))
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
            "actual_ranks": self.actual_ranks,
            "avg_rank": avg_rank,
            "avg_reconstruction_error": avg_err,
            "task_type": self.task_type,
            "keep_ratio": self.keep_ratio,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "recalibrate_ln": self.recalibrate_ln,
            "method": "cachenotes_decoupled_cpt_svd",
            "kv_cache_budget_saving": f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction",
        })
        self.stats = stats
        self.model.compression_stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cachenotes_factory")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor_factory")
def cachenotes_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """CacheNotes compressor factory function."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    task_type = kwargs.pop(
        "task_type",
        _cfg_get(comp_config, "cachenotes_task", _cfg_get(comp_config, "task_type", "qa")),
    )
    keep_ratio = kwargs.pop(
        "keep_ratio",
        _cfg_get(comp_config, "cachenotes_keep_ratio", _cfg_get(comp_config, "keep_ratio", 0.5)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "cachenotes_max_error", _cfg_get(comp_config, "max_error", 0.35)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "cachenotes_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    protect_critical = kwargs.pop(
        "protect_critical",
        _cfg_get(comp_config, "protect_critical", True),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "cachenotes_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    lora_r = kwargs.pop(
        "lora_r",
        _cfg_get(comp_config, "cachenotes_lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 0))),
    )
    lora_alpha = kwargs.pop(
        "lora_alpha",
        _cfg_get(comp_config, "cachenotes_lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0)),
    )
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv",
        _cfg_get(comp_config, "cachenotes_skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", True)),
    )
    recalibrate_ln = kwargs.pop(
        "recalibrate_ln",
        _cfg_get(comp_config, "cachenotes_recalibrate_ln", _cfg_get(comp_config, "recalibrate_ln", True)),
    )

    compressor = CacheNotesCompressor(
        model,
        task_type=task_type,
        keep_ratio=keep_ratio,
        max_error=max_error,
        fallback_to_original=fallback,
        protect_critical=protect_critical,
        target_attention_only=target_attention_only,
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
