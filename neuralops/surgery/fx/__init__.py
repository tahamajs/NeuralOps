# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FX Graph Surgery Subpackage.

Modularized components for AST pattern matching, graph rewriting,
and Inductor compiler integration.
"""

from __future__ import annotations

from neuralops.surgery.fx.compiler import (
    assert_pointwise_fusion,
    benchmark_dispatch_overhead,
    compile_linear_surgery,
    configure_inductor_fusion,
    make_linear_surgery_compiler,
    rewriter_linear_attention_fx,
    verify_zero_graph_breaks,
)
from neuralops.surgery.fx.kernels import (
    LinearAttentionBackwardGraphModule,
    _build_aot_autograd_backward_subgraph,
    analytical_linear_attention_backward,
    bidirectional_associative_scan,
    causal_prefix_scan,
    fast_linear_attention_replacement,
    linear_feature_map,
    linear_feature_map_derivative,
    verify_aot_autograd_backward_gradients,
)
from neuralops.surgery.fx.patterns import (
    AVMatchResult,
    AttentionSubgraph,
    SDPAContext,
    SoftmaxProfile,
    _find_attention_candidates,
    _find_root_proj,
    _isolate_softmax_pattern,
    _match_av_matmul,
    _match_decomposed_qk_matmul,
    _match_sdpa_atomic_node,
)
from neuralops.surgery.fx.rewriter import (
    FXAttentionGraphRewriter,
    _build_bidirectional_linear_subgraph,
    _build_causal_prefix_scan_subgraph,
    _eliminate_dead_code,
    _insert_feature_map,
    _reanchor_residual_connection,
    _rewire_output_projection,
    _verify_graph_integrity,
)
from neuralops.surgery.fx.tracer import (
    BASE_LEAF_MODULE_TYPES,
    LEAF_MODULE_NAMES,
    LEAF_MODULE_TYPES,
    NeuralOpsAttentionTracer,
    _SDPA_TARGETS,
    _sanitize_graph_inputs,
)

__all__ = [
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
