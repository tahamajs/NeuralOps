# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration and Governance Test Suite for Modularized Surgery Subsystems.

Verifies:
1. Complete backward compatibility of facades (neuralops.surgery.director, neuralops.surgery.fx_surgery).
2. Direct submodule importability for director_modules and fx submodules.
3. Functional parity and execution across all extracted components.
4. Strict compliance with AGENTS.md rules:
   - Line counts <= 800 (AGENTS.md §6.3)
   - Zero naked print() statements (AGENTS.md §2, §6)
   - Zero hardcoded 'cuda:0' strings (AGENTS.md §10)
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

# Director facade & submodule imports
import neuralops.surgery.director as dir_facade
from neuralops.surgery.director_modules import (
    bridges,
    certification,
    embedding,
    invariants,
    profiler,
)

# FX facade & submodule imports
import neuralops.surgery.fx_surgery as fx_facade
from neuralops.surgery.fx import (
    compiler,
    kernels,
    patterns,
    rewriter,
    tracer,
)


class TestDirectorModularization:
    """Tests director decomposition, submodule imports, and invariant execution."""

    def test_director_facade_symbols(self) -> None:
        """Verify director facade re-exports all critical symbols."""
        required = [
            "ArchitectureProfile",
            "profile_model_topology",
            "detect_projection_layout",
            "extract_qkv_weights",
            "inspect_positional_topology",
            "verify_tied_embeddings",
            "remediate_tied_embeddings",
            "re_tie_embeddings",
            "get_input_embedding_module",
            "get_output_embedding_module",
            "sanitize_kv_for_padding_mask",
            "sanitize_output_for_padding_mask",
            "apply_rotary_pos_emb",
            "apply_rope_pre_feature_map",
            "decouple_rope_transform",
            "initialize_identity_adapters",
            "compute_state_decay_bound",
            "compute_decayed_state_bound",
            "GatheredParametersContext",
            "InvariantPreservingLinearBridge",
            "CrossAttentionLinearBridge",
            "UniversalSurgeryDirector",
        ]
        for sym in required:
            assert hasattr(dir_facade, sym), f"director facade missing {sym}"

    def test_director_submodule_direct_usage(self) -> None:
        """Verify direct calls to extracted submodules work as expected."""

        # 1. Embedding submodule
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 32)
                self.lm_head = nn.Linear(32, 100, bias=False)

        model = DummyModel()
        in_emb = embedding.get_input_embedding_module(model)
        out_emb = embedding.get_output_embedding_module(model)
        assert in_emb is model.embed_tokens
        assert out_emb is model.lm_head
        assert not embedding.verify_tied_embeddings(model)

        # Tie embeddings
        embedding.remediate_tied_embeddings(model)
        assert embedding.verify_tied_embeddings(model)

        # 2. Invariants submodule
        k = torch.ones(2, 4, 8, 16)
        v = torch.ones(2, 4, 8, 16)
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0.0
        k_san, v_san = invariants.sanitize_kv_for_padding_mask(k, v, mask)
        assert torch.all(k_san[:, :, 4:, :] == 0.0)
        assert torch.all(v_san[:, :, 4:, :] == 0.0)

        # 3. Bridges submodule
        bridge = bridges.InvariantPreservingLinearBridge(
            has_qk_norm=True,
            head_dim=16,
        )
        q_norm, k_norm = bridge.apply_qk_norm(k, k)
        assert q_norm.shape == k.shape
        assert bridge.last_accum_dtype == torch.float32

        # 4. Profiler submodule
        prof = profiler.profile_model_topology(model)
        assert isinstance(prof, profiler.ArchitectureProfile)

        # 5. Certification submodule
        report = certification.certify_model(model, prof, strict=False)
        assert "laws_certified" in report
        assert "Law_8_Tied_Embeddings" in report["laws_certified"]
        assert report["laws_certified"]["Law_8_Tied_Embeddings"] is True


class TestFXModularization:
    """Tests FX decomposition, submodule imports, and graph rewrites."""

    def test_fx_facade_symbols(self) -> None:
        """Verify fx facade re-exports all critical symbols."""
        required = [
            "AttentionSubgraph",
            "BASE_LEAF_MODULE_TYPES",
            "LEAF_MODULE_NAMES",
            "LEAF_MODULE_TYPES",
            "NeuralOpsAttentionTracer",
            "_find_attention_candidates",
            "_match_decomposed_qk_matmul",
            "_sanitize_graph_inputs",
            "SoftmaxProfile",
            "_isolate_softmax_pattern",
            "AVMatchResult",
            "_match_av_matmul",
            "SDPAContext",
            "_match_sdpa_atomic_node",
            "_insert_feature_map",
            "_build_causal_prefix_scan_subgraph",
            "_build_bidirectional_linear_subgraph",
            "_rewire_output_projection",
            "_reanchor_residual_connection",
            "_eliminate_dead_code",
            "_verify_graph_integrity",
            "FXAttentionGraphRewriter",
            "benchmark_dispatch_overhead",
            "bidirectional_associative_scan",
            "causal_prefix_scan",
            "compile_linear_surgery",
            "fast_linear_attention_replacement",
            "linear_feature_map",
            "rewriter_linear_attention_fx",
            "make_linear_surgery_compiler",
            "verify_zero_graph_breaks",
            "configure_inductor_fusion",
            "assert_pointwise_fusion",
            "linear_feature_map_derivative",
            "analytical_linear_attention_backward",
            "LinearAttentionBackwardGraphModule",
            "_build_aot_autograd_backward_subgraph",
            "verify_aot_autograd_backward_gradients",
        ]
        for sym in required:
            assert hasattr(fx_facade, sym), f"fx_surgery facade missing {sym}"

    def test_fx_submodule_direct_usage(self) -> None:
        """Verify direct execution of extracted FX functions and classes."""
        # 1. Kernels submodule
        x = torch.randn(2, 4, 8, 16)
        phi_x = kernels.linear_feature_map(x)
        assert torch.all(phi_x >= 0.0)

        d_phi = kernels.linear_feature_map_derivative(x)
        assert d_phi.shape == x.shape

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = kernels.causal_prefix_scan(q, k, v)
        assert out.shape == (2, 4, 8, 16)
        assert not torch.isnan(out).any()

        # 2. Tracer submodule
        class SimpleAttentionToy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(32, 32, bias=False)
                self.k_proj = nn.Linear(32, 32, bias=False)
                self.v_proj = nn.Linear(32, 32, bias=False)
                self.out_proj = nn.Linear(32, 32, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                q = self.q_proj(x).view(x.shape[0], x.shape[1], 2, 16).transpose(1, 2)
                k = self.k_proj(x).view(x.shape[0], x.shape[1], 2, 16).transpose(1, 2)
                v = self.v_proj(x).view(x.shape[0], x.shape[1], 2, 16).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-2, -1)) / 4.0
                probs = torch.softmax(scores, dim=-1)
                context = torch.matmul(probs, v)
                context = context.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], 32)
                return self.out_proj(context)

        toy = SimpleAttentionToy()
        trc = tracer.NeuralOpsAttentionTracer()
        graph = trc.trace(toy)
        gm = torch.fx.GraphModule(toy, graph)
        assert len(gm.graph.nodes) > 0

        # 3. Patterns submodule
        candidates = patterns._find_attention_candidates(gm.graph)
        assert len(candidates) >= 1

        # 4. Rewriter submodule
        rew = rewriter.FXAttentionGraphRewriter(mode="causal")
        rewritten_gm = rew.rewrite(gm)
        assert isinstance(rewritten_gm, torch.fx.GraphModule)

        # Execute rewritten graph
        inp = torch.randn(2, 8, 32)
        out_rewritten = rewritten_gm(inp)
        assert out_rewritten.shape == (2, 8, 32)
        assert not torch.isnan(out_rewritten).any()


class TestGovernanceAndInvariants:
    """Verifies that AGENTS.md rules are strictly satisfied."""

    def test_zero_hardcoded_device_in_surgery(self) -> None:
        """Verify no hardcoded 'cuda:0' strings exist in surgery code."""
        modules = [
            dir_facade,
            bridges,
            certification,
            embedding,
            invariants,
            profiler,
            fx_facade,
            compiler,
            kernels,
            patterns,
            rewriter,
            tracer,
        ]
        for mod in modules:
            source = inspect.getsource(mod)
            assert "cuda:0" not in source, f"Hardcoded cuda:0 detected in {mod.__name__}"

    def test_zero_naked_prints_in_surgery(self) -> None:
        """Verify no naked print() calls exist in surgery code."""
        paths = [
            "neuralops/surgery/director.py",
            "neuralops/surgery/director_modules/bridges.py",
            "neuralops/surgery/director_modules/certification.py",
            "neuralops/surgery/director_modules/embedding.py",
            "neuralops/surgery/director_modules/invariants.py",
            "neuralops/surgery/director_modules/profiler.py",
            "neuralops/surgery/fx_surgery.py",
            "neuralops/surgery/fx/compiler.py",
            "neuralops/surgery/fx/kernels.py",
            "neuralops/surgery/fx/patterns.py",
            "neuralops/surgery/fx/rewriter.py",
            "neuralops/surgery/fx/tracer.py",
        ]
        for path in paths:
            with open(path) as f:
                content = f.read()
            # Check for naked print( not preceded by def or within comments
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped:
                    assert False, f"Naked print detected in {path}:{idx + 1}: {stripped}"

    def test_file_size_limits_compliance(self) -> None:
        """Verify modularized files strictly obey the hard limit (<= 800 lines)."""
        paths = [
            "neuralops/surgery/director.py",
            "neuralops/surgery/director_modules/bridges.py",
            "neuralops/surgery/director_modules/certification.py",
            "neuralops/surgery/director_modules/embedding.py",
            "neuralops/surgery/director_modules/invariants.py",
            "neuralops/surgery/director_modules/profiler.py",
            "neuralops/surgery/fx_surgery.py",
            "neuralops/surgery/fx/compiler.py",
            "neuralops/surgery/fx/kernels.py",
            "neuralops/surgery/fx/patterns.py",
            "neuralops/surgery/fx/rewriter.py",
            "neuralops/surgery/fx/tracer.py",
        ]
        for path in paths:
            assert os.path.exists(path), f"File {path} does not exist"
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) <= 800, f"File {path} exceeds 800 lines ({len(lines)} lines)"
