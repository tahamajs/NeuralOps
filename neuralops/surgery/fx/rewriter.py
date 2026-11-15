# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FX Attention Graph Rewriter Module.
"""

from __future__ import annotations

import logging
import operator
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.fx.tracer import NeuralOpsAttentionTracer
from neuralops.surgery.fx.patterns import (
    AttentionSubgraph,
    SoftmaxProfile,
    AVMatchResult,
    SDPAContext,
    _find_attention_candidates,
    _isolate_softmax_pattern,
    _match_av_matmul,
    _match_sdpa_atomic_node,
)
from neuralops.surgery.fx.kernels import (
    fast_linear_attention_replacement,
    linear_feature_map,
)

logger = logging.getLogger("neuralops.surgery.fx.rewriter")


# Phase 009: Non-Negative Feature Map Injector (phi(x) = elu(x) + 1.0)
# ---------------------------------------------------------------------------
def _insert_feature_map(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    enforce_fp32: bool = False,
) -> torch.fx.Node:
    """Inserts non-negative linear attention feature map phi(x) = elu(x) + 1.0 into graph.

    Guarantees phi(x) >= 0.0 for all inputs x in R. Enforces FP32 accumulator cast if
    enforce_fp32 is True or if node is recognized as FP16/BF16 to prevent precision underflow.

    Returns:
        Output proxy node of the inserted feature map.
    """
    with graph.inserting_after(node):
        elu_node = graph.call_function(F.elu, args=(node,))
        phi_node = graph.call_function(operator.add, args=(elu_node, 1.0))

        needs_fp32 = enforce_fp32
        if not needs_fp32 and hasattr(node, "meta"):
            ex_val = node.meta.get("example_value", None)
            if hasattr(ex_val, "dtype") and ex_val.dtype in (torch.float16, torch.bfloat16):
                needs_fp32 = True

        if needs_fp32:
            phi_node = graph.call_method("to", args=(phi_node, torch.float32))

        return phi_node


# ---------------------------------------------------------------------------
# Phase 010: Causal Linear Prefix Scan Graph Node Builder
# ---------------------------------------------------------------------------
def _build_causal_prefix_scan_subgraph(
    graph: torch.fx.Graph,
    phi_q: torch.fx.Node,
    phi_k: torch.fx.Node,
    v: torch.fx.Node,
    eps: float = 1e-6,
    enforce_fp32: bool = False,
) -> torch.fx.Node:
    """Builds causal linear prefix recurrence via vectorized cumulative sum nodes.

    Constructs the exact associative prefix scan dataflow:
        S_t = sum_{s=1}^t phi(K_s)^T V_s in R^{... x t x D_k x D_v}
        z_t = sum_{s=1}^t phi(K_s)^T in R^{... x t x D_k}
        O_t = (phi(Q_t) S_t) / (phi(Q_t) z_t + eps) in R^{... x t x D_v}

    Uses dimension-agnostic ellipsis einsums ("...td,...te->...tde") and prefix cumsums
    with dim=-3 for state tensor and dim=-2 for normalizer vector.

    Returns:
        Output proxy node O representing causal linear attention output.
    """
    orig_dtype = None
    if hasattr(phi_q, "meta"):
        ex = phi_q.meta.get("example_value", None)
        if hasattr(ex, "dtype"):
            orig_dtype = ex.dtype

    cast_fp32 = enforce_fp32 or (orig_dtype in (torch.float16, torch.bfloat16))

    q_in = phi_q
    k_in = phi_k
    v_in = v

    if cast_fp32:
        q_in = graph.call_method("to", args=(q_in, torch.float32))
        k_in = graph.call_method("to", args=(k_in, torch.float32))
        v_in = graph.call_method("to", args=(v_in, torch.float32))

    kv_outer = graph.call_function(torch.einsum, args=("...td,...te->...tde", k_in, v_in))
    s_cum = graph.call_function(torch.cumsum, args=(kv_outer, -3))
    z_cum = graph.call_function(torch.cumsum, args=(k_in, -2))
    num = graph.call_function(torch.einsum, args=("...td,...tde->...te", q_in, s_cum))
    den = graph.call_function(torch.einsum, args=("...td,...td->...t", q_in, z_cum))
    den_unsq = graph.call_method("unsqueeze", args=(den, -1))
    den_guarded = graph.call_function(operator.add, args=(den_unsq, eps))
    out = graph.call_function(operator.truediv, args=(num, den_guarded))

    if cast_fp32 and orig_dtype is not None:
        out = graph.call_method("to", args=(out, orig_dtype))

    return out


def _build_bidirectional_linear_subgraph(
    graph: torch.fx.Graph,
    phi_q: Union[torch.fx.Node, torch.fx.Proxy],
    phi_k: Union[torch.fx.Node, torch.fx.Proxy],
    v: Union[torch.fx.Node, torch.fx.Proxy],
    eps: float = 1e-6,
    enforce_fp32: bool = False,
) -> torch.fx.Node:
    """Constructs bidirectional linear attention subgraph directly into an FX Graph.

    For non-causal encoder models (BERT, RoBERTa, ViT), unmasked linear attention:
        S = sum_{s=1}^T phi(K_s)^T V_s in R^{B x H x D_k x D_v}
        z = sum_{s=1}^T phi(K_s)^T in R^{B x H x D_k}
        O_t = (phi(Q_t) S) / (phi(Q_t) z + eps) in R^{B x H x D_v}

    Complexity collapses from O(T^2 * D) to O(T * D^2), yielding strictly linear memory
    scaling across sequence lengths T in [512, 8192].
    """
    node_q = phi_q.node if isinstance(phi_q, torch.fx.Proxy) else phi_q
    node_k = phi_k.node if isinstance(phi_k, torch.fx.Proxy) else phi_k
    node_v = v.node if isinstance(v, torch.fx.Proxy) else v

    orig_dtype = None
    if hasattr(node_q, "meta"):
        ex = node_q.meta.get("example_value", None)
        if hasattr(ex, "dtype"):
            orig_dtype = ex.dtype

    cast_fp32 = enforce_fp32 or (orig_dtype in (torch.float16, torch.bfloat16))

    q_in = node_q
    k_in = node_k
    v_in = node_v

    if cast_fp32:
        q_in = graph.call_method("to", args=(q_in, torch.float32))
        k_in = graph.call_method("to", args=(k_in, torch.float32))
        v_in = graph.call_method("to", args=(v_in, torch.float32))

    # 1. Total associative KV state: S = sum_{s=1}^T phi(K_s)^T V_s
    kv_total = graph.call_function(torch.einsum, args=("...sd,...se->...de", k_in, v_in))

    # 2. Key sum normalizer vector: z = sum_{s=1}^T phi(K_s)^T
    k_sum = graph.call_function(torch.sum, args=(k_in, -2, True))

    # 3. Numerator projection: phi(Q_t) S
    num = graph.call_function(torch.einsum, args=("...td,...de->...te", q_in, kv_total))

    # 4. Denominator projection: phi(Q_t) z
    den = graph.call_function(torch.einsum, args=("...td,...1d->...t", q_in, k_sum))

    # 5. Stabilized normalizer division: O_t = num / (den.unsqueeze(-1) + eps)
    den_unsqueezed = graph.call_method("unsqueeze", args=(den, -1))
    den_guarded = graph.call_function(operator.add, args=(den_unsqueezed, eps))
    out = graph.call_function(operator.truediv, args=(num, den_guarded))

    if cast_fp32 and orig_dtype is not None:
        out = graph.call_method("to", args=(out, orig_dtype))

    return out


def _rewire_output_projection(
    graph: torch.fx.Graph,
    out_linear: torch.fx.Node,
    out_proj_node: torch.fx.Node,
    batch_size: Optional[Union[int, torch.fx.Node]] = None,
    seq_len: Optional[Union[int, torch.fx.Node]] = None,
) -> torch.fx.Node:
    """Rewires linear attention output to downstream projection with shape contiguity restoration.

    Converts multi-head output (B, H, T, D_h) to contiguous (B, T, H * D_h) = (B, T, D_model)
    before feeding downstream projection W_O in R^{D_model x D_model}.
    """
    with graph.inserting_before(out_proj_node):
        perm = graph.call_method("transpose", args=(out_linear, 1, 2))
        cont = graph.call_method("contiguous", args=(perm,))
        b = batch_size if batch_size is not None else graph.call_method("size", args=(cont, 0))
        t = seq_len if seq_len is not None else graph.call_method("size", args=(cont, 1))
        view = graph.call_method("view", args=(cont, b, t, -1))

    # Replace upstream input to out_proj_node with view
    if len(out_proj_node.args) > 0:
        new_args = list(out_proj_node.args)
        replaced = False
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.fx.Node):
                new_args[i] = view
                replaced = True
                break
        if not replaced:
            new_args[0] = view
        out_proj_node.args = tuple(new_args)
    else:
        out_proj_node.args = (view,)

    return view


def _reanchor_residual_connection(
    graph: torch.fx.Graph,
    residual_input: torch.fx.Node,
    out_proj_node: torch.fx.Node,
    norm_node: Optional[torch.fx.Node] = None,
) -> torch.fx.Node:
    """Identifies or constructs the Pre-LN residual stream connection and preserves gradient highway.

    Pre-LN architecture: x_{l+1} = x_l + W_O * Attn(RMSNorm(x_l)).
    Preserves exact identity Jacobian d(x_{l+1}) / d(x_l) = I + d(Attn) / d(x_l).
    """
    if norm_node is not None:
        if len(norm_node.args) >= 1 and norm_node.args[0] != residual_input:
            norm_args = list(norm_node.args)
            norm_args[0] = residual_input
            norm_node.args = tuple(norm_args)

    res_add_node: Optional[torch.fx.Node] = None
    for user in list(out_proj_node.users):
        target_name = getattr(user.target, "__name__", str(user.target))
        is_add = (
            user.op == "call_function"
            and (user.target in (operator.add, operator.__add__, torch.add) or target_name == "add")
        ) or (user.op == "call_method" and user.target == "add")
        if is_add:
            res_add_node = user
            break

    if res_add_node is None:
        for node in graph.nodes:
            target_name = getattr(node.target, "__name__", str(node.target))
            is_add = (
                node.op == "call_function"
                and (node.target in (operator.add, operator.__add__, torch.add) or target_name == "add")
            ) or (node.op == "call_method" and node.target == "add")
            if is_add and len(node.args) >= 2:
                if (node.args[0] == residual_input and node.args[1] == out_proj_node) or (
                    node.args[0] == out_proj_node and node.args[1] == residual_input
                ):
                    res_add_node = node
                    break

    if res_add_node is not None:
        res_add_node.args = (residual_input, out_proj_node)
    else:
        with graph.inserting_after(out_proj_node):
            res_add_node = graph.call_function(operator.add, args=(residual_input, out_proj_node))

    return res_add_node


def _eliminate_dead_code(graph: torch.fx.Graph) -> int:
    """Iteratively eliminates orphaned dead nodes until a complete sweep produces 0 deletions.

    Sweeps backward over the FX graph, erasing nodes with zero users (excluding output nodes).
    Thoroughly prunes orphaned quadratic attention calculations: Q K^T matmuls, scaling, masking,
    softmax, and dropout nodes.

    Returns:
        int: Total number of dead nodes eliminated across all sweeps.
    """
    total_pruned = 0
    while True:
        pruned_in_sweep = 0
        for node in reversed(list(graph.nodes)):
            if len(node.users) == 0 and node.op != "output":
                graph.erase_node(node)
                pruned_in_sweep += 1
        total_pruned += pruned_in_sweep
        if pruned_in_sweep == 0:
            break
    return total_pruned


def _verify_graph_integrity(
    graph_or_gm: Union[torch.fx.Graph, torch.fx.GraphModule],
    gm: Optional[torch.fx.GraphModule] = None,
) -> bool:
    """Verifies graph integrity via FX linting, Kahn's topological sort, and recompilation.

    Validates:
    1. graph.lint(): ensures all use-def chains are strictly ordered with zero dangling edges.
    2. Kahn's algorithm: asserts graph is a strict DAG, visiting 100% of nodes with zero cycles.
    3. Recompiles the GraphModule to synchronize Python bytecode.

    Returns:
        bool: True if verification succeeds.
    """
    if isinstance(graph_or_gm, torch.fx.GraphModule):
        graph = graph_or_gm.graph
        graph_module: Optional[torch.fx.GraphModule] = graph_or_gm
    else:
        graph = graph_or_gm
        graph_module = gm if gm is not None else getattr(graph, "owning_module", None)

    # 1. Graph linting (validates use-def integrity)
    graph.lint()

    # 2. Kahn's algorithm topological sort verification
    nodes_list = list(graph.nodes)
    total_nodes = len(nodes_list)
    nodes_set = set(nodes_list)

    # Compute in-degree for each node within this graph
    in_degree: Dict[torch.fx.Node, int] = {}
    for node in nodes_list:
        in_nodes = [inp for inp in node.all_input_nodes if inp in nodes_set]
        in_degree[node] = len(in_nodes)

    queue = deque([node for node in nodes_list if in_degree[node] == 0])
    visited_count = 0

    while queue:
        curr = queue.popleft()
        visited_count += 1
        for user in curr.users:
            if user in in_degree:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)

    if visited_count != total_nodes:
        unvisited = [n.name for n in nodes_list if in_degree.get(n, 0) > 0]
        raise RuntimeError(
            f"Graph topological sort failed: visited {visited_count}/{total_nodes} nodes. "
            f"Possible cycle or disconnected dependency in nodes: {unvisited[:10]}"
        )

    # 3. Recompile graph
    if graph_module is not None:
        graph_module.recompile()
    elif hasattr(graph, "recompile") and callable(graph.recompile):
        graph.recompile()

    return True


class FXAttentionGraphRewriter:
    """Orchestrates symbolic tracing, dual subgraph pattern matching,

    dead-code elimination, and graph recompilation for linear attention transplants.
    """

    def __init__(
        self,
        target_attention: str = "cosformer",
        head_dim: int = 64,
        is_causal: bool = True,
        leaf_modules: Optional[List[Type[nn.Module]]] = None,
        decay: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.target_attention = target_attention
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.decay = decay
        self.extra_kwargs = kwargs
        self.tracer = NeuralOpsAttentionTracer(leaf_modules=leaf_modules)

    def _find_attention_candidates(self, graph: torch.fx.Graph) -> List[torch.fx.Node]:
        """Identifies attention clusters via topological scan and reverse search."""
        return _find_attention_candidates(graph)

    def find_attention_candidates(
        self,
        gm_or_graph: Union[torch.fx.GraphModule, torch.fx.Graph],
    ) -> List[torch.fx.Node]:
        """Convenience helper to retrieve all attention candidate anchors from a graph or GraphModule."""
        graph = gm_or_graph.graph if isinstance(gm_or_graph, torch.fx.GraphModule) else gm_or_graph
        return _find_attention_candidates(graph)

    def replace_sdpa_nodes(self, gm: torch.fx.GraphModule) -> int:
        """Finds and replaces all aten.scaled_dot_product_attention nodes with linear kernels."""
        graph = gm.graph
        count = 0
        for node in list(graph.nodes):
            sdpa_ctx = _match_sdpa_atomic_node(node)
            if sdpa_ctx is not None:
                q = sdpa_ctx.q
                k = sdpa_ctx.k
                v = sdpa_ctx.v
                mask = sdpa_ctx.mask
                scale = sdpa_ctx.scale
                causal = sdpa_ctx.is_causal if sdpa_ctx.is_causal is not None else self.is_causal

                with graph.inserting_before(node):
                    new_node = graph.call_function(
                        fast_linear_attention_replacement,
                        args=(q, k, v),
                        kwargs={
                            "mask": mask,
                            "scale": scale,
                            "is_causal": causal,
                            "target_attention": self.target_attention,
                            "decay": self.decay,
                        },
                    )
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
                    count += 1
        return count

    def replace_decomposed_patterns(self, gm: torch.fx.GraphModule) -> int:
        """Matches decomposed [matmul -> scale -> mask -> softmax -> matmul] sequences."""
        graph = gm.graph
        count = 0
        for node in list(graph.nodes):
            # Target the final matmul(P, V)
            target_str = str(node.target)
            target_name = getattr(node.target, "__name__", target_str)
            is_matmul = (
                node.op == "call_function"
                and (
                    node.target in (torch.matmul, operator.matmul, torch.bmm)
                    or target_name in ("matmul", "bmm")
                    or "matmul" in target_str
                    or "bmm" in target_str
                )
            ) or (node.op == "call_method" and node.target in ("matmul", "bmm"))

            if is_matmul and len(node.args) >= 2:
                p_candidate = node.args[0]
                v_candidate = node.args[1]

                # Check if p_candidate is dropout preceding softmax
                if isinstance(p_candidate, torch.fx.Node):
                    if (
                        p_candidate.op == "call_function"
                        and (p_candidate.target in (torch.dropout, F.dropout) or "dropout" in str(p_candidate.target))
                    ) or (p_candidate.op == "call_module" and "dropout" in str(p_candidate.target).lower()):
                        if len(p_candidate.args) >= 1 and isinstance(p_candidate.args[0], torch.fx.Node):
                            p_candidate = p_candidate.args[0]

                    p_target_str = str(p_candidate.target)
                    p_target_name = getattr(p_candidate.target, "__name__", p_target_str)
                    is_softmax = (
                        p_candidate.op == "call_function"
                        and (
                            p_candidate.target in (torch.softmax, F.softmax)
                            or "softmax" in p_target_str
                            or p_target_name == "softmax"
                        )
                    ) or (p_candidate.op == "call_method" and p_candidate.target == "softmax")

                    if is_softmax and len(p_candidate.args) >= 1:
                        scores_node = p_candidate.args[0]
                        curr = scores_node
                        mask_node = None
                        scale_val = None

                        # Strip additions, masked_fill, and divisions/multiplications
                        while isinstance(curr, torch.fx.Node) and curr.op in ("call_function", "call_method"):
                            c_target = curr.target
                            c_name = getattr(c_target, "__name__", str(c_target))

                            if c_target in (torch.add, operator.add, operator.__add__) or c_name == "add":
                                if isinstance(curr.args[0], torch.fx.Node) and curr.args[0].op in (
                                    "call_function",
                                    "call_method",
                                ):
                                    mask_node = curr.args[1]
                                    curr = curr.args[0]
                                elif len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node):
                                    mask_node = curr.args[0]
                                    curr = curr.args[1]
                                else:
                                    break
                            elif (
                                c_target is torch.masked_fill
                                or c_name == "masked_fill"
                                or (curr.op == "call_method" and curr.target == "masked_fill")
                            ):
                                mask_node = curr.args[1] if len(curr.args) > 1 else None
                                curr = curr.args[0]
                            elif c_target in (torch.div, operator.truediv, operator.__truediv__) or c_name in (
                                "div",
                                "truediv",
                            ):
                                scale_val = curr.args[1] if len(curr.args) > 1 else None
                                curr = curr.args[0]
                            elif c_target in (torch.mul, operator.mul, operator.__mul__) or c_name == "mul":
                                if isinstance(curr.args[0], torch.fx.Node) and curr.args[0].op in (
                                    "call_function",
                                    "call_method",
                                ):
                                    scale_val = curr.args[1] if len(curr.args) > 1 else None
                                    curr = curr.args[0]
                                elif len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node):
                                    scale_val = curr.args[0]
                                    curr = curr.args[1]
                                else:
                                    break
                            elif (
                                c_target in (torch.matmul, operator.matmul, torch.bmm)
                                or c_name in ("matmul", "bmm")
                                or "matmul" in str(c_target)
                                or "bmm" in str(c_target)
                            ):
                                break
                            else:
                                break

                        # Verify we arrived at matmul(Q, K^T)
                        if (
                            isinstance(curr, torch.fx.Node)
                            and (
                                curr.target in (torch.matmul, operator.matmul, torch.bmm)
                                or getattr(curr.target, "__name__", "") in ("matmul", "bmm")
                                or (curr.op == "call_method" and curr.target in ("matmul", "bmm"))
                            )
                            and len(curr.args) >= 2
                        ):
                            q_candidate = curr.args[0]
                            k_candidate_trans = curr.args[1]

                            # Detect transpose or permute on K
                            k_candidate = k_candidate_trans
                            if isinstance(k_candidate_trans, torch.fx.Node):
                                if k_candidate_trans.op == "call_method" and k_candidate_trans.target in (
                                    "transpose",
                                    "permute",
                                ):
                                    k_candidate = k_candidate_trans.args[0]
                                elif k_candidate_trans.op == "call_function" and (
                                    k_candidate_trans.target in (torch.transpose, torch.permute)
                                    or getattr(k_candidate_trans.target, "__name__", "") in ("transpose", "permute")
                                ):
                                    k_candidate = k_candidate_trans.args[0]

                            with graph.inserting_before(node):
                                new_node = graph.call_function(
                                    fast_linear_attention_replacement,
                                    args=(q_candidate, k_candidate, v_candidate),
                                    kwargs={
                                        "mask": mask_node,
                                        "scale": scale_val,
                                        "is_causal": self.is_causal,
                                        "target_attention": self.target_attention,
                                        "decay": self.decay,
                                    },
                                )
                                node.replace_all_uses_with(new_node)
                                graph.erase_node(node)
                                count += 1
        return count

    def eliminate_dead_nodes(self, gm: torch.fx.GraphModule) -> int:
        """Iteratively eliminates orphaned nodes until graph node count stabilizes."""
        return _eliminate_dead_code(gm.graph)

    def clean_and_recompile(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Eliminates dead nodes, validates graph topology, and recompiles bytecode."""
        removed = self.eliminate_dead_nodes(gm)
        _verify_graph_integrity(gm)
        logger.info("FX Graph sanitized: %d dead nodes eliminated.", removed)
        return gm

    def _build_bidirectional_linear_subgraph(
        self,
        graph: torch.fx.Graph,
        phi_q: Union[torch.fx.Node, torch.fx.Proxy],
        phi_k: Union[torch.fx.Node, torch.fx.Proxy],
        v: Union[torch.fx.Node, torch.fx.Proxy],
        eps: float = 1e-6,
        enforce_fp32: bool = False,
    ) -> torch.fx.Node:
        """Constructs bidirectional linear attention subgraph directly into an FX Graph."""
        return _build_bidirectional_linear_subgraph(graph, phi_q, phi_k, v, eps=eps, enforce_fp32=enforce_fp32)

    def _rewire_output_projection(
        self,
        graph: torch.fx.Graph,
        out_linear: torch.fx.Node,
        out_proj_node: torch.fx.Node,
        batch_size: Optional[Union[int, torch.fx.Node]] = None,
        seq_len: Optional[Union[int, torch.fx.Node]] = None,
    ) -> torch.fx.Node:
        """Rewires linear attention output to downstream projection with shape contiguity restoration."""
        return _rewire_output_projection(graph, out_linear, out_proj_node, batch_size, seq_len)

    def _reanchor_residual_connection(
        self,
        graph: torch.fx.Graph,
        residual_input: torch.fx.Node,
        out_proj_node: torch.fx.Node,
        norm_node: Optional[torch.fx.Node] = None,
    ) -> torch.fx.Node:
        """Identifies or constructs the Pre-LN residual stream connection and preserves gradient highway."""
        return _reanchor_residual_connection(graph, residual_input, out_proj_node, norm_node)

    def _eliminate_dead_code(self, graph: torch.fx.Graph) -> int:
        """Iteratively eliminates orphaned dead nodes until a complete sweep produces 0 deletions."""
        return _eliminate_dead_code(graph)

    def _verify_graph_integrity(
        self,
        graph_or_gm: Union[torch.fx.Graph, torch.fx.GraphModule],
        gm: Optional[torch.fx.GraphModule] = None,
    ) -> bool:
        """Verifies graph integrity via FX linting, Kahn's topological sort, and recompilation."""
        return _verify_graph_integrity(graph_or_gm, gm)

    def rewrite(
        self,
        model: nn.Module,
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> torch.fx.GraphModule:
        """Traces the model, performs dual subgraph replacement, purges dead code, and recompiles."""
        logger.info("Symbolically tracing model with NeuralOpsAttentionTracer...")
        actual_model = getattr(model, "model", model)
        if isinstance(actual_model, torch.fx.GraphModule):
            gm = actual_model
        else:
            graph = self.tracer.trace(actual_model, concrete_args=concrete_args)
            gm = torch.fx.GraphModule(actual_model, graph)

        sdpa_count = self.replace_sdpa_nodes(gm)
        decomp_count = self.replace_decomposed_patterns(gm)
        total_replaced = sdpa_count + decomp_count

        logger.info("Replaced %d attention nodes (SDPA: %d, Decomposed: %d).", total_replaced, sdpa_count, decomp_count)

        self.clean_and_recompile(gm)
        return gm
