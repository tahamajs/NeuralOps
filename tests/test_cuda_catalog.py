#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying single source of truth for the CUDA kernel catalog and symbol resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralops.kernels.cuda import CATALOG as KERNELS_CATALOG
from neuralops.ops.cuda._map import _KERNEL_SYMBOL_MAP
from neuralops.ops.cuda._map import CUDA_KERNEL_CATALOG as OPS_CATALOG


def test_catalog_matches_disk():
    """Verify every ops/cuda/ kernel folder is present in CUDA_KERNEL_CATALOG."""
    root_dir = Path(__file__).resolve().parent.parent
    cuda_dir = root_dir / "neuralops" / "ops" / "cuda"
    disk_folders = sorted(
        [p.name for p in cuda_dir.iterdir() if p.is_dir() and p.name not in ("__pycache__", "plugins")]
    )
    assert len(disk_folders) == 139, f"Expected 139 on-disk folders, found {len(disk_folders)}"

    catalog_set = set(OPS_CATALOG)
    missing = [f for f in disk_folders if f not in catalog_set]
    assert not missing, f"On-disk folders missing from CUDA_KERNEL_CATALOG: {missing}"


def test_every_catalog_entry_has_symbol_map():
    """Verify every catalog entry has at least one symbol-map key."""
    missing = [entry for entry in OPS_CATALOG if entry not in _KERNEL_SYMBOL_MAP]
    assert not missing, f"Catalog entries without a symbol-map key: {missing}"


def test_no_phantom_quantized_fp8_gemm():
    """Verify quantized_fp8_gemm is absent unless a .cu file exists."""
    root_dir = Path(__file__).resolve().parent.parent
    cu_matches = list(root_dir.glob("**/quantized_fp8_gemm.cu"))
    assert not cu_matches, "Unexpected quantized_fp8_gemm.cu found"
    assert "quantized_fp8_gemm" not in OPS_CATALOG, "Phantom entry quantized_fp8_gemm must be removed"
    assert "quantized_fp8_gemm" not in KERNELS_CATALOG, "Phantom entry quantized_fp8_gemm must be removed"


def test_kernels_catalog_equals_ops_catalog():
    """Verify neuralops.kernels.cuda.CATALOG equals neuralops.ops.cuda._map.CUDA_KERNEL_CATALOG."""
    assert KERNELS_CATALOG is OPS_CATALOG
    assert KERNELS_CATALOG == OPS_CATALOG


def test_29_missing_modules_now_visible():
    """Verify the 29 previously missing modules resolve via neuralops.ops.cuda.<name>."""
    import neuralops.ops.cuda as ops_cuda

    missing_29 = [
        "adaptive_sparse",
        "admm",
        "asvd",
        "awq",
        "based_taylor_attention",
        "chela_slay",
        "chela_slay_wide",
        "direct_svd",
        "dynamic_conv",
        "fp8_compressor",
        "fused_svd_projection",
        "fwsvd",
        "gated_deltanet",
        "gla_chunk_attn",
        "hla_attention",
        "hnc_svd",
        "hybrid_local_kernel",
        "kmeans_quantize",
        "lightning_attention",
        "lsh_attention",
        "rala",
        "retnet_chunk_retention",
        "rwkv6_state_scan",
        "sparsegpt",
        "svd_linear_decay",
        "svd_mamba_fused",
        "token_compressed",
        "ttt_linear_step",
        "wanda",
    ]

    for name in missing_29:
        assert hasattr(ops_cuda, name), f"Module {name} not found in ops_cuda"
        val = getattr(ops_cuda, name)
        assert val is not None, f"Module {name} resolved to None"


def test_extended_kernels_jit_count_matches_docstring() -> None:
    """Verify that @triton.jit count in extended_kernels.py matches the count stated in its docstring."""
    from neuralops.kernels import extended_kernels

    file_path = Path(extended_kernels.__file__).resolve()
    content = file_path.read_text(encoding="utf-8")

    jit_count = content.count("@triton.jit")
    assert jit_count == 2, f"Expected 2 @triton.jit decorators, found {jit_count}"

    doc = extended_kernels.__doc__ or ""
    assert f"{jit_count} Triton kernels" in doc or f"{jit_count} Triton JIT kernels" in doc, (
        f"Docstring missing '{jit_count} Triton kernels'"
    )
    assert "18 PyTorch references" in doc, "Docstring missing '18 PyTorch references'"


def test_no_duplicate_cu_files() -> None:
    """Verify no duplicate .cu files exist and all catalog folders exist on disk (F-41)."""
    import hashlib

    root_dir = Path(__file__).resolve().parent.parent
    cuda_dir = root_dir / "neuralops" / "ops" / "cuda"

    cu_files = list(cuda_dir.glob("**/*.cu"))
    assert len(cu_files) > 0, "No .cu files found"

    hashes: dict[str, list[Path]] = {}
    for cu in cu_files:
        content = cu.read_bytes()
        # Skip small compat shims that only include another .cu file
        if b"#include" in content and len(content) < 512:
            continue
        sha = hashlib.sha256(content).hexdigest()
        hashes.setdefault(sha, []).append(cu)

    duplicates = {sha: paths for sha, paths in hashes.items() if len(paths) > 1}
    assert not duplicates, f"Duplicate .cu files found: {duplicates}"

    # Verify every non-special catalog folder exists on disk
    for entry in OPS_CATALOG:
        folder = cuda_dir / entry
        if entry in _KERNEL_SYMBOL_MAP:
            target_sub = _KERNEL_SYMBOL_MAP[entry][0]
            if not target_sub.startswith("_"):
                assert (cuda_dir / target_sub).exists(), f"Target folder {target_sub} for catalog entry {entry} missing"
        else:
            assert folder.exists(), f"Catalog folder {folder} does not exist on disk"


def test_trt_plugin_referenced_in_build() -> None:
    """Verify that TensorRT linear attention plugin is packaged in pyproject.cuda.toml and exists on disk (F-46)."""
    import tomllib

    root_dir = Path(__file__).resolve().parent.parent
    plugin_file = root_dir / "neuralops" / "ops" / "cuda" / "plugins" / "trt_linear_attention_plugin.cpp"
    assert plugin_file.exists(), f"TensorRT plugin source missing: {plugin_file}"

    pyproject_cuda_path = root_dir / "pyproject.cuda.toml"
    assert pyproject_cuda_path.exists(), "pyproject.cuda.toml not found"

    data = tomllib.loads(pyproject_cuda_path.read_text(encoding="utf-8"))
    pkg_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})

    plugins_included = any(
        "trt_linear_attention_plugin.cpp" in entries or "*.cpp" in entries
        for entries in pkg_data.values()
    )
    assert plugins_included, f"trt_linear_attention_plugin.cpp not referenced in pyproject.cuda.toml package-data: {pkg_data}"
