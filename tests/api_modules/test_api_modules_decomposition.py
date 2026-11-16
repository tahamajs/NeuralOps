# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for decomposed api_modules subpackage and modular entrypoints.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest
import torch
import torch.nn as nn

import neuralops
import neuralops.api as napi
import neuralops.api_modules.attention_surgery as m_attn
import neuralops.api_modules.benchmarking as m_bench
import neuralops.api_modules.compression as m_comp
import neuralops.api_modules.export as m_exp
import neuralops.api_modules.inference as m_inf
import neuralops.api_modules.registry_ops as m_reg
import neuralops.api_modules.shadow_twin as m_twin
import neuralops.api_modules.training as m_train


def test_api_modules_imports() -> None:
    """Verify that all 8 api_modules can be imported cleanly and provide their respective functions."""
    assert callable(m_attn.replace_attention)
    assert callable(m_attn.transplant_attention)
    assert callable(m_attn.compose_attention)

    assert callable(m_comp.compress)
    assert callable(m_comp.compress_model)

    assert callable(m_twin.collect_surgery_distillation_loss)
    assert callable(m_twin.set_surgery_alpha)
    assert callable(m_twin.set_surgery_stage)
    assert callable(m_twin.finalize_surgery)

    assert callable(m_exp.export_model)
    assert callable(m_exp.export)
    assert callable(m_exp.push_to_hub)
    assert callable(m_exp.load_model)
    assert callable(m_exp.compile_to_so)

    assert callable(m_inf.pipeline)
    assert callable(m_inf.serve)
    assert hasattr(m_inf, "TextGenerationPipeline")

    assert callable(m_reg.list_attentions)
    assert callable(m_reg.list_compressors)
    assert callable(m_reg.get_attention_class)
    assert callable(m_reg.get_compressor_class)
    assert callable(m_reg.list_families)

    assert callable(m_train.train)
    assert callable(m_train.distill)
    assert callable(m_train.evaluate)
    assert callable(m_train.load_and_prepare_dataset)
    assert callable(m_train.train_grpo)

    assert callable(m_bench.benchmark)
    assert callable(m_bench.benchmark_all)
    assert callable(m_bench.get_version)
    assert callable(m_bench.has_triton)
    assert callable(m_bench.available_backends)
    assert callable(m_bench.validate)
    assert callable(m_bench.certify)
    assert callable(m_bench.info)


def test_identity_parity_with_napi() -> None:
    """Verify that neuralops.api re-exports the exact modular objects."""
    assert napi.replace_attention is m_attn.replace_attention
    assert napi.transplant_attention is m_attn.transplant_attention
    assert napi.compose_attention is m_attn.compose_attention
    assert napi.compress is m_comp.compress
    assert napi.compress_model is m_comp.compress_model
    assert napi.export_model is m_exp.export_model
    assert napi.export is m_exp.export
    assert napi.pipeline is m_inf.pipeline
    assert napi.serve is m_inf.serve
    assert napi.TextGenerationPipeline is m_inf.TextGenerationPipeline
    assert napi.list_attentions is m_reg.list_attentions
    assert napi.list_compressors is m_reg.list_compressors
    assert napi.train is m_train.train
    assert napi.distill is m_train.distill
    assert napi.evaluate is m_train.evaluate
    assert napi.benchmark is m_bench.benchmark
    assert napi.validate is m_bench.validate
    assert napi.certify is m_bench.certify
    assert napi.info is m_bench.info


def test_file_size_constraints() -> None:
    """Verify that all api files and api_modules stay strictly under the 800 lines limit (AGENTS.md Rule 6.3)."""
    repo_root = Path(neuralops.__file__).resolve().parent
    api_file = repo_root / "api.py"
    init_file = repo_root / "__init__.py"
    api_modules_dir = repo_root / "api_modules"

    files_to_check = [api_file, init_file] + list(api_modules_dir.glob("*.py"))
    violations = []
    for f in files_to_check:
        lines = len(f.read_text(encoding="utf-8").splitlines())
        if lines > 800:
            violations.append(f"{f.name}: {lines} lines (>800)")

    assert not violations, f"File size rule violated: {violations}"


def test_no_torch_cuda_amp() -> None:
    """Verify that no torch.cuda.amp references exist in api_modules or core compat shims (AGENTS.md Rule 2.1)."""
    repo_root = Path(neuralops.__file__).resolve().parent
    targets = [
        repo_root / "api.py",
        repo_root / "__init__.py",
        repo_root / "core" / "_torch_compat.py",
        repo_root / "core" / "_module_alias_finder.py",
    ] + list((repo_root / "api_modules").glob("*.py"))

    violations = []
    for f in targets:
        text = f.read_text(encoding="utf-8")
        if "torch.cuda.amp" in text:
            violations.append(f"{f.name} contains deprecated torch.cuda.amp")

    assert not violations, f"Rule 2.1 violations: {violations}"
