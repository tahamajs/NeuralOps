#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""FLA-style operation registry for correctness-gated microbenchmarks.

The registry is deliberately metadata-first.  An operation is benchmarkable
only when both its implementation and its verified input contract are known;
unknown CUDA/Triton contracts are represented as unavailable instead of being
measured with guessed tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


TensorFactory = Callable[[dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]
Operation = Callable[..., Any]


@dataclass(frozen=True)
class OpConfig:
    """Description of one correctness-gated benchmark operation."""

    name: str
    category: str
    func: Operation | None
    tensor_specs: dict[str, Any]
    ref_func: Operation | None
    input_factory: TensorFactory | None = None
    unavailable_reason: str | None = None


def _act_scale_inputs(spec: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Exact contract copied from the archived isolated CUDA harness."""
    import torch

    dtype = spec["dtype"]
    device = spec["device"]
    x = torch.randn(64, 256, device=device, dtype=dtype) * 0.05
    s_inv = torch.randn(256, device=device, dtype=dtype) * 0.1
    weight = torch.randn(128, 256, device=device, dtype=dtype) * 0.05
    return (x, s_inv, weight), {}


def _kernel_inputs(name: str) -> TensorFactory:
    """Return an input factory using the archived CUDA harness contracts."""
    def factory(spec: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        import torch

        device, dtype = spec["device"], spec["dtype"]
        if name == "causal_conv1d":
            return (
                torch.randn(4, 64, 512, device=device, dtype=dtype) * 0.1,
                torch.randn(64, 4, device=device, dtype=dtype) * 0.1,
            ), {}
        if name == "rmsnorm":
            return (
                torch.randn(32, 512, device=device, dtype=dtype) * 0.1,
                torch.randn(512, device=device, dtype=dtype) * 0.1,
            ), {}
        if name == "swiglu":
            return (torch.randn(32, 512, device=device, dtype=dtype) * 0.1,), {}
        if name == "fused_cross_entropy":
            return (
                torch.randn(2048, 1024, device=device, dtype=dtype) * 0.1,
                torch.randint(0, 1024, (2048,), device=device, dtype=torch.long),
            ), {}
        raise KeyError(name)
    return factory


def _attention_inputs(spec: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    import torch

    return (torch.randn(
        spec["batch"], spec["sequence"], spec["hidden"],
        device=spec["device"], dtype=spec["dtype"],
    ),), {}


def _attention_operation(
    name: str,
    *,
    force_pytorch: bool,
    state: dict[str, Any] | None = None,
) -> Operation:
    """Build an apples-to-apples invocation of one attention module."""
    state = state if state is not None else {}

    def invoke(x: Any) -> Any:
        import torch
        from neuralops.core.registry import create_attention
        from neuralops.core.cuda_kernels import _fallback as fallback

        # Seed before construction so both paths share deterministic weights.
        torch.manual_seed(42)
        module = state.get("module")
        if module is None:
            module = create_attention(name, embed_dim=x.shape[-1], num_heads=12)
            module = module.to(device=x.device, dtype=x.dtype)
            module.eval()
            state["module"] = module
        previous = fallback.FORCE_PYTORCH_FALLBACK
        fallback.FORCE_PYTORCH_FALLBACK = force_pytorch
        try:
            return module(x)
        finally:
            fallback.FORCE_PYTORCH_FALLBACK = previous

    return invoke


def _triton_norm_inputs(spec: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    import torch

    device = spec["device"]
    dtype = spec["dtype"]
    x = torch.randn(4, 64, 512, device=device, dtype=dtype)
    weight = torch.randn(512, 512, device=device, dtype=dtype)
    bias = torch.randn(512, device=device, dtype=dtype)
    return (x, weight, bias), {"eps": 1e-6}


def _compressor_inputs(spec: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build calibration activations for model-level compressor contracts."""
    import torch

    return (
        torch.randn(
            spec.get("batch", 2), spec.get("sequence", 16), spec.get("hidden", 128),
            device=spec["device"], dtype=torch.float32,
        ),
    ), {}


def _compressor_operation(name: str, *, compressed: bool, state: dict[str, Any]) -> Operation:
    """Run a registered compressor against a real model-level contract."""
    import copy
    import inspect
    import torch
    import torch.nn as nn
    from neuralops.core.registry import create_compressor

    def invoke(x: Any) -> Any:
        if "base" not in state:
            hidden = int(x.shape[-1])
            state["base"] = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            ).to(device=x.device, dtype=torch.float32)
        if not compressed:
            return state["base"](x.float())
        if "compressed" not in state:
            candidate = copy.deepcopy(state["base"])
            rank = min(32, max(4, x.shape[-1] // 2))
            try:
                engine = create_compressor(
                    name, model=candidate, rank=rank,
                    calibration_data=x.float(), calib_data=x.float(),
                    skip_should_compress_filter=True,
                )
            except TypeError:
                engine = create_compressor(
                    name, model=candidate, rank=rank,
                    skip_should_compress_filter=True,
                )
            compressed_model = engine if isinstance(engine, nn.Module) else None
            if compressed_model is None and hasattr(engine, "compress"):
                method = engine.compress
                parameters = inspect.signature(method).parameters
                kwargs: dict[str, Any] = {}
                if "calibration_data" in parameters:
                    kwargs["calibration_data"] = x.float()
                if "calib_data" in parameters:
                    kwargs["calib_data"] = x.float()
                if "model" in parameters:
                    kwargs["model"] = candidate
                outcome = method(**kwargs)
                compressed_model = outcome[0] if isinstance(outcome, tuple) else outcome
            if not isinstance(compressed_model, nn.Module):
                raise TypeError(
                    f"{name} did not return a transformed nn.Module; "
                    "use its model-specific/KV-cache benchmark contract"
                )
            state["compressed"] = compressed_model.to(device=x.device)
            before = sum(p.numel() for p in state["base"].parameters())
            after = sum(p.numel() for p in state["compressed"].parameters())
            state["reduction_pct"] = (1.0 - after / before) * 100.0 if before else 0.0
        return state["compressed"](x.float())

    invoke.state = state  # type: ignore[attr-defined]
    return invoke



def _register_registry_modules() -> dict[str, OpConfig]:
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

    result: dict[str, OpConfig] = {}
    attention_specs = {
        "batch": 4, "sequence": 512, "hidden": 768,
        "heads": 12, "dtype": "float16", "device": "cuda",
    }
    for name in ATTENTION_REGISTRY.keys():
        attention_state: dict[str, Any] = {}
        result[f"attention:{name}"] = OpConfig(
            name=name,
            category="attention",
            func=_attention_operation(name, force_pytorch=False, state=attention_state),
            tensor_specs=attention_specs,
            ref_func=_attention_operation(name, force_pytorch=True, state=attention_state),
            input_factory=_attention_inputs,
            unavailable_reason=None,
        )
    for name in COMPRESSOR_REGISTRY.keys():
        fast_state: dict[str, Any] = {}
        reference_state: dict[str, Any] = {}
        result[f"compressor:{name}"] = OpConfig(
            name=name,
            category="compressor",
            func=_compressor_operation(name, compressed=True, state=fast_state),
            tensor_specs={
                "device": "cuda", "batch": 2, "sequence": 16, "hidden": 128,
            },
            ref_func=_compressor_operation(name, compressed=False, state=reference_state),
            input_factory=_compressor_inputs,
        )


    try:
        from neuralops.core.cuda_kernels.act_scale_gemm import (
            act_scale_quant_gemm,
            act_scale_quant_gemm_pytorch_ref,
        )
        result["cuda:act_scale_gemm"] = OpConfig(
            name="act_scale_gemm",
            category="cuda",
            func=act_scale_quant_gemm,
            tensor_specs={"device": "cuda", "dtype": "float16", "shape": (64, 128, 256)},
            ref_func=act_scale_quant_gemm_pytorch_ref,
            input_factory=_act_scale_inputs,
        )
    except Exception as exc:
        result["cuda:act_scale_gemm"] = OpConfig(
            name="act_scale_gemm", category="cuda", func=None,
            tensor_specs={}, ref_func=None,
            unavailable_reason=f"CUDA operator import failed: {exc}",
        )

    cuda_specs = {"device": "cuda", "dtype": "float16"}
    try:
        from neuralops.core.cuda_kernels import (
            causal_conv1d, causal_conv1d_pytorch_ref,
            rmsnorm, rmsnorm_pytorch_ref,
            swiglu, swiglu_pytorch_ref,
            fused_cross_entropy, fused_cross_entropy_pytorch_ref,
        )
        for name, fast, ref in (
            ("causal_conv1d", causal_conv1d, causal_conv1d_pytorch_ref),
            ("rmsnorm", rmsnorm, rmsnorm_pytorch_ref),
            ("swiglu", swiglu, swiglu_pytorch_ref),
            ("fused_cross_entropy", fused_cross_entropy, fused_cross_entropy_pytorch_ref),
        ):
            result[f"cuda:{name}"] = OpConfig(
                name=name, category="cuda", func=fast, tensor_specs=cuda_specs,
                ref_func=ref, input_factory=_kernel_inputs(name),
            )
    except Exception as exc:
        for name in ("causal_conv1d", "rmsnorm", "swiglu", "fused_cross_entropy"):
            result[f"cuda:{name}"] = OpConfig(
                name=name, category="cuda", func=None, tensor_specs={}, ref_func=None,
                unavailable_reason=f"Kernel import failed: {exc}",
            )

    try:
        from neuralops.core.triton_kernels import fused_norm_linear, fused_norm_linear_pytorch
        result["triton:fused_norm_linear"] = OpConfig(
            name="fused_norm_linear",
            category="triton",
            func=fused_norm_linear,
            tensor_specs={"device": "cuda", "dtype": "float16", "shape": (4, 64, 512)},
            ref_func=fused_norm_linear_pytorch,
            input_factory=_triton_norm_inputs,
        )
    except Exception as exc:
        result["triton:fused_norm_linear"] = OpConfig(
            name="fused_norm_linear",
            category="triton",
            func=None,
            tensor_specs={},
            ref_func=None,
            unavailable_reason=f"Triton operator import failed: {exc}",
        )

    # These kernels exceeded the bounded release budget even at S=128. Keep
    # them explicitly unavailable instead of measuring an unrelated operation.
    for name in ("hla_attention", "samba_moe"):
        result[f"attention:{name}"] = OpConfig(
            name=name,
            category="attention",
            func=None,
            tensor_specs={
                "device": "cuda", "fallback_only": True,
                "fallback_reason": "Kernel exceeded the 180-second release budget at S=128",
            },
            ref_func=None,
            input_factory=None,
            unavailable_reason="FALLBACK_ONLY: kernel exceeded the 180-second release budget",
        )

    return result


OPS_REGISTRY = _register_registry_modules()


def get_op(name: str) -> OpConfig:
    """Return a registered operation or raise a descriptive error."""
    if name == "triton:fused_LayerNorm":
        name = "triton:fused_norm_linear"
    try:
        return OPS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark operation: {name}") from exc


def list_ops(category: str = "all") -> list[str]:
    """List operation keys, optionally filtered by category."""
    if category == "all":
        return sorted(OPS_REGISTRY)
    return sorted(key for key, value in OPS_REGISTRY.items() if value.category == category)


__all__ = ["OpConfig", "OPS_REGISTRY", "get_op", "list_ops"]
