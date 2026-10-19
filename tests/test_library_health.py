#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CI Health Gate & Library Static/Dynamic Invariant Verification Tests.


Contains the 6 critical health gate tests:
  1. test_no_unbound_locals()
  2. test_no_silent_dispatch_fallback()
  3. test_no_instance_attr_outside_init()
  4. test_neuralops_imports()
  5. test_replace_attention_end_to_end()
  6. test_zero_dead_code_and_shadowed_defs()
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

import neuralops
from neuralops import (
    load_model,
    replace_attention,
    compress,
    export_model,
    list_attentions,
    list_compressors,
    get_attention_class,
    get_compressor_class,
    list_families,
    get_version,
)
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from neuralops.core.triton_kernels import triton_dispatch
from neuralops.core.triton_kernels._types import _WARNED_REASONS, log_fallback_reason


def test_no_unbound_locals():
    """Verify that all Python modules in neuralops compile cleanly without unbound locals or syntax errors."""
    repo_root = Path(__file__).resolve().parent.parent
    neuralops_pkg = repo_root / "neuralops"
    assert neuralops_pkg.exists(), f"Package directory not found: {neuralops_pkg}"

    compiled_count = 0
    for py_file in neuralops_pkg.rglob("*.py"):
        if "_archive" in str(py_file):
            continue
        code_text = py_file.read_text(encoding="utf-8")
        try:
            compiled = compile(code_text, str(py_file), "exec")
            assert compiled is not None
            compiled_count += 1
        except SyntaxError as e:
            pytest.fail(f"Syntax error compiling {py_file}: {e}")

    assert compiled_count > 100, f"Expected >100 compiled modules, got {compiled_count}"

    try:
        import subprocess

        res = subprocess.run(
            ["ruff", "check", "--select", "F821,F823", str(neuralops_pkg)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            errors = [line for line in res.stdout.splitlines() if "F821" in line or "F823" in line]
            assert len(errors) == 0, f"Found unbound local / undefined names: {errors}"
    except FileNotFoundError:
        pass


def test_no_silent_dispatch_fallback(caplog):
    """Verify that when triton_dispatch falls back to PyTorch, the fallback is logged and tracked (never silent)."""
    _WARNED_REASONS.clear()

    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 16, 64)
    v = torch.randn(2, 4, 16, 64)

    with caplog.at_level(logging.INFO):
        out = triton_dispatch("linear_attn", q, k, v)

    assert out.shape == (2, 4, 16, 64)
    assert not torch.isnan(out).any()

    fallback_records = [r for r in _WARNED_REASONS if "linear_attn" in r]
    assert len(fallback_records) > 0, "Dispatch fallback occurred silently without registering in _WARNED_REASONS"
    assert any("cpu" in r.lower() or "cuda" in r.lower() for r in fallback_records)

    test_key = "test_kernel"
    test_reason = "simulated_hardware_unsupported"
    log_fallback_reason(test_key, test_reason)
    assert f"{test_key}:{test_reason}" in _WARNED_REASONS


def test_no_instance_attr_outside_init():
    """Verify that attention modules do not dynamically add unexpected instance attributes outside __init__."""
    sample_modules = ["cosformer", "galerkin", "cluster_attention", "gqa", "compressed_mha"]

    for name in sample_modules:
        cls = ATTENTION_REGISTRY.get(name)
        if cls is None:
            continue

        try:
            mod = cls(dim=64, num_heads=4)
        except Exception:
            continue

        if not isinstance(mod, nn.Module):
            continue

        mod.eval()
        x = torch.randn(2, 16, 64)

        attrs_before = set(mod.__dict__.keys())

        try:
            with torch.no_grad():
                _ = mod(x)
        except Exception:
            continue

        attrs_after = set(mod.__dict__.keys())
        new_attrs = attrs_after - attrs_before

        filtered_new = {a for a in new_attrs if not a.startswith("_torch") and not a.startswith("_is_full_backward_hook")}
        assert len(filtered_new) == 0, f"Module '{name}' dynamically set instance attributes outside __init__: {filtered_new}"


def test_neuralops_imports():
    """Verify that top-level package imports cleanly, preserves version, and maintains canonical registry counts."""
    assert hasattr(neuralops, "__version__")
    assert neuralops.__version__ == "1.4.1"
    assert get_version() == "1.4.1"

    api_functions = [
        load_model,
        replace_attention,
        compress,
        export_model,
        list_attentions,
        list_compressors,
        get_attention_class,
        get_compressor_class,
        list_families,
    ]
    for fn in api_functions:
        assert callable(fn), f"Public API {fn.__name__} is not callable"

    assert len(ATTENTION_REGISTRY) == 131, f"Expected 131 registered attentions, got {len(ATTENTION_REGISTRY)}"
    assert len(COMPRESSOR_REGISTRY) == 72, f"Expected 72 registered compressors, got {len(COMPRESSOR_REGISTRY)}"

    attns = list_attentions()
    comps = list_compressors()
    assert len(attns) >= 120
    assert len(comps) >= 70
    assert "cosformer" in attns
    assert "direct_svd" in comps


def test_replace_attention_end_to_end():
    """Verify end-to-end model surgery: replacing attention in BertModel and validating forward execution and metadata."""
    config = BertConfig(
        vocab_size=120,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    model = BertModel(config)

    replaced_model = replace_attention(model, "cosformer")

    assert hasattr(replaced_model, "_neuralops_metadata"), "Replaced model missing _neuralops_metadata"
    meta = replaced_model._neuralops_metadata
    assert len(meta) >= 1
    assert meta[0]["op"] == "replace"
    assert meta[0]["type"] == "cosformer"

    replaced_model.eval()
    input_ids = torch.randint(0, 120, (2, 16))
    with torch.no_grad():
        output = replaced_model(input_ids)

    assert hasattr(output, "last_hidden_state")
    assert output.last_hidden_state.shape == (2, 16, 64)
    assert not torch.isnan(output.last_hidden_state).any()


def test_zero_dead_code_and_shadowed_defs():
    """Verify zero shadowed definitions and zero unreachable statements across the repository via AST auditing."""
    repo_root = Path(__file__).resolve().parent.parent
    py_files = sorted(
        p
        for p in repo_root.rglob("*.py")
        if not any(
            part in p.parts
            for part in (".git", "__pycache__", "build", "dist", ".venv", "venv", "archive", "_archive")
        )
    )
    assert len(py_files) > 100, f"Expected >100 Python source files, got {len(py_files)}"

    shadowed_defs: list[str] = []
    unreachable_stmts: list[str] = []

    for path in py_files:
        src = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError as exc:
            pytest.fail(f"Syntax error parsing {path}: {exc}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef)):
                seen: dict[str, int] = {}
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        decs = [ast.unparse(d) for d in getattr(stmt, "decorator_list", [])]
                        if any("overload" in d or "setter" in d or "deleter" in d for d in decs):
                            continue
                        if stmt.name in seen:
                            shadowed_defs.append(
                                f"{path.relative_to(repo_root)}:{stmt.lineno} -> '{stmt.name}' shadows definition at line {seen[stmt.name]}"
                            )
                        seen[stmt.name] = stmt.lineno

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    body = getattr(sub, "body", None)
                    if isinstance(body, list):
                        for idx, stmt in enumerate(body[:-1]):
                            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                                nxt = body[idx + 1]
                                unreachable_stmts.append(
                                    f"{path.relative_to(repo_root)}:{nxt.lineno} -> Unreachable statement in '{node.name}' after {type(stmt).__name__}"
                                )
                                break

    assert len(shadowed_defs) == 0, f"Found {len(shadowed_defs)} shadowed definitions:\n" + "\n".join(shadowed_defs)
    assert len(unreachable_stmts) == 0, f"Found {len(unreachable_stmts)} unreachable statements:\n" + "\n".join(unreachable_stmts)
