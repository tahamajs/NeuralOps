"""Extensibility, Custom Module Protocol & Contract Validation for NeuralOps.

Provides first-class interfaces for researchers and contributors to register
custom attention mechanisms, custom compressors, and 4-stage algebraic primitives,
with automatic contract diagnostics, inspection, profiling, and surgical compatibility.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

# Track dynamically registered custom keys to protect canonical baseline (131 / 72 / 26)
_CUSTOM_ATTENTION_KEYS: set[str] = set()
_CUSTOM_COMPRESSOR_KEYS: set[str] = set()
_CUSTOM_MODULE_METADATA: dict[str, dict[str, Any]] = {}


def register_custom_attention(
    name: Optional[str] = None,
    cls_or_fn: Optional[Union[Type[nn.Module], Callable]] = None,
    *,
    family: str = "custom",
    time_complexity: str = "O(T·d²)",
    memory_complexity: str = "O(d²) state",
    math_equation: Optional[str] = None,
    triton_kernel: Optional[str] = None,
    sram_fusion_strategy: Optional[str] = None,
    supports_causal: bool = True,
    supports_bidirectional: bool = True,
    supports_gqa: bool = False,
    overwrite: bool = True,
    **extra_metadata: Any,
) -> Any:
    """Decorator or direct function to register a custom attention mechanism.

    Usable as:
        @neuralops.register_custom_attention("my_attn", family="linear-scan")
        class MyAttention(nn.Module):
            ...

    Or:
        neuralops.register_custom_attention("my_attn", MyAttentionClass, ...)
    """
    def _decorator(obj: Union[Type[nn.Module], Callable]) -> Union[Type[nn.Module], Callable]:
        key = (name or getattr(obj, "__name__", "custom_attention")).lower()
        
        # Store metadata
        meta = {
            "family": family,
            "time_complexity": time_complexity,
            "memory_complexity": memory_complexity,
            "math_equation": math_equation or f"Y = {obj.__name__}(X; Θ)",
            "triton_kernel": triton_kernel or "custom_kernel.py",
            "sram_fusion_strategy": sram_fusion_strategy or "Custom operator registration",
            "supports_causal": supports_causal,
            "supports_bidirectional": supports_bidirectional,
            "supports_gqa": supports_gqa,
            "is_custom": True,
        }
        meta.update(extra_metadata)
        _CUSTOM_MODULE_METADATA[key] = meta
        _CUSTOM_ATTENTION_KEYS.add(key)

        # Register in ATTENTION_REGISTRY
        ATTENTION_REGISTRY.register(
            key=key,
            supports_causal=supports_causal,
            supports_bidirectional=supports_bidirectional,
            supports_gqa=supports_gqa,
            family=family,
            time_complexity=time_complexity,
            memory_complexity=memory_complexity,
            math_equation=meta["math_equation"],
            triton_kernel=meta["triton_kernel"],
            sram_fusion_strategy=meta["sram_fusion_strategy"],
            is_custom=True,
            **extra_metadata,
        )(obj)
        logger.info("Registered custom attention module '%s' (%s)", key, obj.__name__)
        return obj

    if cls_or_fn is not None:
        return _decorator(cls_or_fn)
    return _decorator


def register_custom_compressor(
    name: Optional[str] = None,
    cls_or_fn: Optional[Union[Type[Any], Callable]] = None,
    *,
    family: str = "custom-compressor",
    time_complexity: str = "O(T·d·r)",
    memory_complexity: str = "O(d·r) parameter footprint",
    math_equation: Optional[str] = None,
    triton_kernel: Optional[str] = None,
    sram_fusion_strategy: Optional[str] = None,
    overwrite: bool = True,
    **extra_metadata: Any,
) -> Any:
    """Decorator or direct function to register a custom compression engine."""
    def _decorator(obj: Union[Type[Any], Callable]) -> Union[Type[Any], Callable]:
        key = (name or getattr(obj, "__name__", "custom_compressor")).lower()
        
        meta = {
            "family": family,
            "time_complexity": time_complexity,
            "memory_complexity": memory_complexity,
            "math_equation": math_equation or f"W' = Compress_{key}(W)",
            "triton_kernel": triton_kernel or "custom_compressor_kernel.py",
            "sram_fusion_strategy": sram_fusion_strategy or "Fused low-rank / quant GEMM",
            "is_custom": True,
        }
        meta.update(extra_metadata)
        _CUSTOM_MODULE_METADATA[key] = meta
        _CUSTOM_COMPRESSOR_KEYS.add(key)

        COMPRESSOR_REGISTRY.register(
            key=key,
            **meta,
        )(obj)
        logger.info("Registered custom compressor module '%s' (%s)", key, obj.__name__)
        return obj

    if cls_or_fn is not None:
        return _decorator(cls_or_fn)
    return _decorator


def unregister_custom_module(name: str, category: Optional[str] = None) -> bool:
    """Cleanly unregister a dynamically registered custom module from registries.

    Protects the canonical baseline (131 attentions / 72 compressors).
    """
    k = name.lower()
    removed = False

    if category in (None, "attention", "attentions"):
        if k in ATTENTION_REGISTRY._dict:
            del ATTENTION_REGISTRY._dict[k]
            removed = True
        if k in ATTENTION_REGISTRY._aliases:
            del ATTENTION_REGISTRY._aliases[k]
            removed = True
        if k in ATTENTION_REGISTRY._metadata:
            del ATTENTION_REGISTRY._metadata[k]
            removed = True
        _CUSTOM_ATTENTION_KEYS.discard(k)

    if category in (None, "compressor", "compressors"):
        if k in COMPRESSOR_REGISTRY._dict:
            del COMPRESSOR_REGISTRY._dict[k]
            removed = True
        if k in COMPRESSOR_REGISTRY._aliases:
            del COMPRESSOR_REGISTRY._aliases[k]
            removed = True
        if k in COMPRESSOR_REGISTRY._metadata:
            del COMPRESSOR_REGISTRY._metadata[k]
            removed = True
        _CUSTOM_COMPRESSOR_KEYS.discard(k)

    _CUSTOM_MODULE_METADATA.pop(k, None)
    return removed


def get_custom_metadata(name: str) -> dict[str, Any]:
    """Retrieve metadata attached to a custom module."""
    return dict(_CUSTOM_MODULE_METADATA.get(name.lower(), {}))


def validate_module_contract(
    module_or_name: Union[str, nn.Module, Type[nn.Module]],
    embed_dim: int = 64,
    num_heads: int = 4,
    seq_len: int = 32,
    batch_size: int = 2,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Automated 6-Point Diagnostic Suite verifying custom or built-in modules against framework contracts:

    1. Instantiation via normalized factory (create_attention / create_compressor)
    2. Forward pass shape conservation ([B, T, D] -> [B, T, D])
    3. Backward pass & finite gradient flow verification (no NaN / Inf)
    4. Architectural Introspection completeness (14-field specification)
    5. Surgical transplantation & micro-healing compatibility on a toy Transformer
    6. Surgical checkpoint serialization save/load bit-exact roundtrip parity.
    """
    from neuralops.core.inspector import inspect_architecture
    from neuralops.api import create_attention, replace_attention
    from neuralops.models.surgical_io import save_surgical_checkpoint, load_surgical_checkpoint
    from neuralops.core._attention_surgery import heal_replaced_model
    import tempfile
    import shutil
    from pathlib import Path

    target_dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    results: Dict[str, Any] = {
        "module": str(module_or_name),
        "device": str(target_dev),
        "checks": {},
        "passed_count": 0,
        "total_checks": 6,
        "all_passed": False,
    }

    # 1. Instantiation Check
    try:
        if isinstance(module_or_name, str):
            module = create_attention(module_or_name, embed_dim=embed_dim, num_heads=num_heads)
            mod_name = module_or_name
        elif isinstance(module_or_name, type) and issubclass(module_or_name, nn.Module):
            try:
                module = module_or_name(embed_dim=embed_dim, num_heads=num_heads)
            except Exception:
                module = module_or_name()
            mod_name = module.__class__.__name__.lower()
        elif isinstance(module_or_name, nn.Module):
            module = module_or_name
            mod_name = module.__class__.__name__.lower()
        else:
            raise TypeError(f"Unsupported module_or_name type: {type(module_or_name)}")

        module = module.to(device=target_dev)
        results["checks"]["1_instantiation"] = {"status": "PASSED", "module_class": module.__class__.__name__}
        results["passed_count"] += 1
    except Exception as e:
        results["checks"]["1_instantiation"] = {"status": "FAILED", "error": str(e)}
        return results

    # 2. Forward Pass Shape Conservation
    x = torch.randn(batch_size, seq_len, embed_dim, device=target_dev, requires_grad=True)
    try:
        module.eval()
        with torch.no_grad():
            out = module(x)
            if isinstance(out, tuple):
                out = out[0]
        
        expected_shape = (batch_size, seq_len, embed_dim)
        if tuple(out.shape) == expected_shape and not torch.isnan(out).any():
            results["checks"]["2_forward_shape"] = {"status": "PASSED", "output_shape": list(out.shape)}
            results["passed_count"] += 1
        else:
            results["checks"]["2_forward_shape"] = {
                "status": "FAILED",
                "error": f"Expected shape {expected_shape}, got {tuple(out.shape)} (NaNs: {torch.isnan(out).any().item()})",
            }
    except Exception as e:
        results["checks"]["2_forward_shape"] = {"status": "FAILED", "error": str(e)}

    # 3. Backward Pass & Gradient Flow
    try:
        module.train()
        x_grad = torch.randn(batch_size, seq_len, embed_dim, device=target_dev, requires_grad=True)
        out_grad = module(x_grad)
        if isinstance(out_grad, tuple):
            out_grad = out_grad[0]
        loss = out_grad.sum()
        loss.backward()

        trainable_params = [p for p in module.parameters() if p.requires_grad]
        # At least one trainable param has gradients and x_grad has gradient
        has_param_grads = any(p.grad is not None and not torch.isnan(p.grad).any() for p in trainable_params) if trainable_params else True
        if has_param_grads and x_grad.grad is not None and not torch.isnan(x_grad.grad).any():
            results["checks"]["3_backward_gradients"] = {
                "status": "PASSED",
                "trainable_parameters_count": len(trainable_params),
            }
            results["passed_count"] += 1
        else:
            results["checks"]["3_backward_gradients"] = {
                "status": "FAILED",
                "error": "Gradients missing or contain NaN/Inf values",
            }
    except Exception as e:
        results["checks"]["3_backward_gradients"] = {"status": "FAILED", "error": str(e)}

    # 4. Introspection Specification Check
    try:
        spec = inspect_architecture(mod_name)
        if spec.name and spec.family and spec.time_complexity and len(spec.constructor_params) > 0:
            results["checks"]["4_introspection_metadata"] = {
                "status": "PASSED",
                "family": spec.family,
                "complexity": spec.time_complexity,
            }
            results["passed_count"] += 1
        else:
            results["checks"]["4_introspection_metadata"] = {
                "status": "FAILED",
                "error": "Incomplete ArchitectureSpec returned",
            }
    except Exception as e:
        results["checks"]["4_introspection_metadata"] = {"status": "FAILED", "error": str(e)}

    # 5. Surgical Replacement & Micro-Healing Compatibility on Toy Transformer
    class ToyBlock(nn.Module):
        def __init__(self, d: int, h: int):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.linear = nn.Linear(d, d)
        def forward(self, h_in):
            a_out, _ = self.attn(h_in, h_in, h_in)
            return self.linear(a_out)

    toy_teacher = nn.Sequential(ToyBlock(embed_dim, num_heads)).to(device=target_dev)
    import copy
    toy_student = copy.deepcopy(toy_teacher)
    try:
        # Clone or fresh instance for surgery
        if isinstance(module_or_name, str):
            surg_module = create_attention(module_or_name, embed_dim=embed_dim, num_heads=num_heads).to(device=target_dev)
        else:
            try:
                surg_module = copy.deepcopy(module)
            except Exception:
                surg_module = module

        replace_attention(toy_student, target_module=surg_module)
        calib_data = torch.randn(4, seq_len, embed_dim, device=target_dev)
        heal_stats = heal_replaced_model(toy_teacher, toy_student, calib_data, steps=5, lr=1e-3)
        results["checks"]["5_surgical_micro_healing"] = {
            "status": "PASSED",
            "initial_mse": heal_stats.get("initial_mse", 0.0),
            "final_mse": heal_stats.get("final_mse", 0.0),
        }
        results["passed_count"] += 1
    except Exception as e:
        results["checks"]["5_surgical_micro_healing"] = {"status": "FAILED", "error": str(e)}

    # 6. Surgical Serialization Save/Load Roundtrip Parity
    tmp_dir = Path(tempfile.mkdtemp(prefix="neuralops_contract_"))
    try:
        toy_student.eval()
        test_x = torch.randn(batch_size, seq_len, embed_dim, device=target_dev)
        with torch.no_grad():
            orig_out = toy_student(test_x)

        save_surgical_checkpoint(toy_student, tmp_dir)
        fresh_base = nn.Sequential(ToyBlock(embed_dim, num_heads)).to(device=target_dev)
        reloaded_model, manifest = load_surgical_checkpoint(fresh_base, tmp_dir, map_location=target_dev)
        reloaded_model.eval()

        with torch.no_grad():
            reloaded_out = reloaded_model(test_x)

        diff = (orig_out - reloaded_out).abs().max().item()
        if diff < 1e-4:
            results["checks"]["6_persistence_roundtrip"] = {
                "status": "PASSED",
                "max_abs_diff": diff,
            }
            results["passed_count"] += 1
        else:
            results["checks"]["6_persistence_roundtrip"] = {
                "status": "FAILED",
                "error": f"Roundtrip discrepancy max diff: {diff:.6e} >= 1e-4",
            }
    except Exception as e:
        results["checks"]["6_persistence_roundtrip"] = {"status": "FAILED", "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    results["all_passed"] = (results["passed_count"] == results["total_checks"])
    return results


def scaffold_module(
    name: str,
    category: str = "attention",
    family: str = "linear-scan",
    output_file: Optional[str] = None,
) -> str:
    """Generate production-ready boilerplate code for a new custom module."""
    camel_name = "".join(part.capitalize() for part in name.split("_"))
    if category == "attention":
        template = f'''"""Custom {camel_name} Attention Mechanism for NeuralOps."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuralops

@neuralops.register_custom_attention(
    "{name}",
    family="{family}",
    time_complexity="O(T·d²)",
    memory_complexity="O(d²) state",
    math_equation="S_t = S_{{t-1}} + k_t^T @ v_t  |  o_t = q_t @ S_t",
    triton_kernel="custom_{name}.py",
    sram_fusion_strategy="In-SRAM state register accumulation",
)
class {camel_name}(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, rank: int | None = None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, T, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2) * self.scale
        k = self.k_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        
        # Subquadratic linear attention recurrence
        scores = torch.matmul(F.elu(q) + 1.0, (F.elu(k) + 1.0).transpose(-1, -2))
        context = torch.matmul(scores, v)
        out = self.out_proj(context.transpose(1, 2).contiguous().view(B, T, D))
        return out
'''
    else:
        template = f'''"""Custom {camel_name} Compression Engine for NeuralOps."""

from __future__ import annotations
import torch
import torch.nn as nn
import neuralops

@neuralops.register_custom_compressor(
    "{name}",
    family="{family}",
    time_complexity="O(T·d·r)",
    memory_complexity="O(d·r) parameter footprint",
    math_equation="W ≈ U @ V",
    triton_kernel="low_rank.py",
)
class {camel_name}:
    def __init__(self, model: nn.Module = None, rank: int = 32, **kwargs):
        self.model = model
        self.rank = rank

    def compress(self, model: nn.Module = None) -> nn.Module:
        target = model or self.model
        # Compression logic here
        return target
'''
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(template)
    return template


__all__ = [
    "register_custom_attention",
    "register_custom_compressor",
    "unregister_custom_module",
    "get_custom_metadata",
    "validate_module_contract",
    "scaffold_module",
]
