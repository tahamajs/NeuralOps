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
    import torch

    return (torch.randn(4, 64, 512, device=spec["device"], dtype=torch.float32),), {}


def _kmeans_operation(*, compressed: bool) -> Operation:
    """Run k-means on a toy model while preserving the uncompressed reference."""
    import copy
    import torch.nn as nn
    from neuralops.core.registry import create_compressor

    state: dict[str, Any] = {}

    def invoke(x: Any) -> Any:
        if "base" not in state:
            state["base"] = nn.Sequential(nn.Linear(512, 512)).to(x.device)
        if not compressed:
            return state["base"](x)
        if "compressed" not in state:
            candidate = copy.deepcopy(state["base"])
            state["compressed"] = create_compressor(
                "kmeans_quantize", model=candidate, rank=32,
                skip_should_compress_filter=True,
            )
            before = sum(p.numel() for p in state["base"].parameters())
            after = sum(p.numel() for p in state["compressed"].parameters())
            state["reduction_pct"] = (1.0 - after / before) * 100.0 if before else 0.0
        return state["compressed"](x)

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
        result[f"compressor:{name}"] = OpConfig(
            name=name,
            category="compressor",
            func=None,
            tensor_specs={"model": "benchmark_transformer_block", "device": "cuda"},
            ref_func=None,
            unavailable_reason="Compressor correctness requires its model-level contract",
        )

    if "compressor:kmeans_quantize" in result:
        fast = _kmeans_operation(compressed=True)
        reference = _kmeans_operation(compressed=False)
        result["compressor:kmeans_quantize"] = OpConfig(
            name="kmeans_quantize",
            category="compressor",
            func=fast,
            tensor_specs={"device": "cuda", "param_reduction_pct": 0.0},
            ref_func=reference,
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
