# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Pattern Matchers and Subgraph Detectors for FX Attention Surgery.
"""

from __future__ import annotations

import logging
import math
import operator
from collections import deque
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.fx.tracer import _SDPA_TARGETS
from neuralops.surgery.fx.kernels import fast_linear_attention_replacement

logger = logging.getLogger("neuralops.surgery.fx.patterns")


class AttentionSubgraph:
    """Structured container holding AST node handles for an attention cluster."""

    def __init__(
        self,
        anchor_node: torch.fx.Node,
        out_proj_node: Optional[torch.fx.Node] = None,
        q_node: Optional[torch.fx.Node] = None,
        k_node: Optional[torch.fx.Node] = None,
        v_node: Optional[torch.fx.Node] = None,
        attn_weights_node: Optional[torch.fx.Node] = None,
        scale: Optional[float] = None,
    ) -> None:
        self.anchor_node = anchor_node
        self.out_proj_node = out_proj_node
        self.q_node = q_node
        self.k_node = k_node
        self.v_node = v_node
        self.attn_weights_node = attn_weights_node
        self.scale = scale

    def __repr__(self) -> str:
        anchor_name = getattr(self.anchor_node, "name", str(self.anchor_node))
        out_name = getattr(self.out_proj_node, "name", None)
        q_name = getattr(self.q_node, "name", None)
        k_name = getattr(self.k_node, "name", None)
        v_name = getattr(self.v_node, "name", None)
        return (
            f"AttentionSubgraph(anchor={anchor_name}, out_proj={out_name}, "
            f"q={q_name}, k={k_name}, v={v_name}, scale={self.scale})"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.anchor_node, name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttentionSubgraph):
            return self.anchor_node == other.anchor_node
        return self.anchor_node == other

    def __hash__(self) -> int:
        return hash(self.anchor_node)


def _find_root_proj(node: Any) -> Any:
    """Traverses backward through view, transpose, permute, and reshape to find root projection module."""
    curr = node
    while isinstance(curr, torch.fx.Node):
        if curr.op == "call_module" and any(
            k in str(curr.target).lower() for k in ("q_", "k_", "v_", "query", "key", "value", "proj", "dense", "attn")
        ):
            return curr
        if curr.op in ("call_method", "call_function") and any(
            op_name in str(curr.target).lower() for op_name in ("view", "reshape", "transpose", "permute", "contiguous")
        ):
            if len(curr.args) >= 1 and isinstance(curr.args[0], torch.fx.Node):
                curr = curr.args[0]
            else:
                break
        else:
            break
    return node


def _match_decomposed_qk_matmul(
    node: torch.fx.Node,
) -> Optional[Tuple[torch.fx.Node, torch.fx.Node, Optional[float]]]:
    """Matches quadratic attention score pattern A_raw = alpha * (Q K^T).

    Traverses matmul, transpose, permute, and scaling operations, extracting
    handles to the Query (Q) node, Key (K) node, and numerical scale factor alpha = 1 / sqrt(d_k).
    """
    if not isinstance(node, torch.fx.Node):
        return None

    curr: Optional[torch.fx.Node] = node
    scale_val: Optional[float] = None
    visited: Set[torch.fx.Node] = set()

    # Traverse backward through softmax, dropout, addition, masked_fill, and division/multiplication
    while isinstance(curr, torch.fx.Node) and curr not in visited:
        visited.add(curr)
        c_target = curr.target
        c_name = getattr(c_target, "__name__", str(c_target))

        # Check if curr is already the QK matmul
        is_matmul = (
            curr.op == "call_function"
            and (
                curr.target in (torch.matmul, operator.matmul, torch.bmm)
                or c_name in ("matmul", "bmm")
                or "matmul" in str(c_target)
                or "bmm" in str(c_target)
            )
        ) or (curr.op == "call_method" and curr.target in ("matmul", "bmm"))

        if is_matmul and len(curr.args) >= 2:
            arg1 = curr.args[1]
            if isinstance(arg1, torch.fx.Node):
                arg1_target = arg1.target
                arg1_name = getattr(arg1_target, "__name__", str(arg1_target))
                if arg1.op in ("call_method", "call_function") and (
                    arg1_target in ("transpose", "permute", torch.transpose, torch.permute)
                    or arg1_name in ("transpose", "permute")
                ):
                    break

        if curr.op in ("call_function", "call_method"):
            if c_target in (torch.softmax, F.softmax) or c_name == "softmax":
                curr = curr.args[0] if len(curr.args) >= 1 and isinstance(curr.args[0], torch.fx.Node) else None
            elif c_target in (torch.dropout, F.dropout) or "dropout" in c_name:
                curr = curr.args[0] if len(curr.args) >= 1 and isinstance(curr.args[0], torch.fx.Node) else None
            elif c_target in (torch.div, operator.truediv, operator.__truediv__) or c_name in ("div", "truediv"):
                if len(curr.args) > 1 and isinstance(curr.args[1], (int, float)):
                    scale_val = 1.0 / float(curr.args[1])
                curr = curr.args[0] if len(curr.args) >= 1 and isinstance(curr.args[0], torch.fx.Node) else None
            elif c_target in (torch.mul, operator.mul, operator.__mul__) or c_name in ("mul", "__mul__"):
                if len(curr.args) > 1 and isinstance(curr.args[1], (int, float)):
                    scale_val = float(curr.args[1])
                elif len(curr.args) > 0 and isinstance(curr.args[0], (int, float)):
                    scale_val = float(curr.args[0])
                if len(curr.args) > 0 and isinstance(curr.args[0], torch.fx.Node):
                    curr = curr.args[0]
                elif len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node):
                    curr = curr.args[1]
                else:
                    curr = None
            elif c_target in (torch.add, operator.add, operator.__add__) or c_name in ("add", "__add__"):
                curr = curr.args[0] if len(curr.args) >= 1 and isinstance(curr.args[0], torch.fx.Node) else None
            elif c_target is torch.masked_fill or c_name == "masked_fill":
                curr = curr.args[0] if len(curr.args) >= 1 and isinstance(curr.args[0], torch.fx.Node) else None
            else:
                break
        else:
            break

    if not isinstance(curr, torch.fx.Node):
        return None

    c_target = curr.target
    c_name = getattr(c_target, "__name__", str(c_target))
    is_matmul = (
        curr.op == "call_function"
        and (
            curr.target in (torch.matmul, operator.matmul, torch.bmm)
            or c_name in ("matmul", "bmm")
            or "matmul" in str(c_target)
            or "bmm" in str(c_target)
        )
    ) or (curr.op == "call_method" and curr.target in ("matmul", "bmm"))

    if not is_matmul or len(curr.args) < 2:
        return None

    q_candidate = curr.args[0]
    k_trans_candidate = curr.args[1]
    if not isinstance(q_candidate, torch.fx.Node) or not isinstance(k_trans_candidate, torch.fx.Node):
        return None

    # Traverse transpose, permute, view, reshape on K
    k_candidate = k_trans_candidate
    if k_trans_candidate.op in ("call_method", "call_function"):
        kt_target = k_trans_candidate.target
        kt_name = getattr(kt_target, "__name__", str(kt_target))
        if kt_target in ("transpose", "permute", torch.transpose, torch.permute) or kt_name in ("transpose", "permute"):
            if len(k_trans_candidate.args) >= 1 and isinstance(k_trans_candidate.args[0], torch.fx.Node):
                k_candidate = k_trans_candidate.args[0]

    # If scale_val was not captured from backward traversal, scan forward users of curr
    if scale_val is None:
        for user in curr.users:
            u_target = user.target
            u_name = getattr(u_target, "__name__", str(u_target))
            if u_target in (torch.div, operator.truediv, operator.__truediv__) or u_name in ("div", "truediv"):
                if len(user.args) > 1 and isinstance(user.args[1], (int, float)):
                    scale_val = 1.0 / float(user.args[1])
                    break
            elif u_target in (torch.mul, operator.mul, operator.__mul__) or u_name in ("mul", "__mul__"):
                if len(user.args) > 1 and isinstance(user.args[1], (int, float)):
                    scale_val = float(user.args[1])
                    break
                elif len(user.args) > 0 and isinstance(user.args[0], (int, float)):
                    scale_val = float(user.args[0])
                    break

    q_root = _find_root_proj(q_candidate)
    k_root = _find_root_proj(k_candidate)

    return (q_root, k_root, scale_val)


def _find_attention_candidates(graph: torch.fx.Graph) -> List[AttentionSubgraph]:
    """Identifies attention clusters via reverse topological search from projection layers.

    Detects both native SDPA nodes and decomposed attention [matmul -> scale -> mask -> softmax -> matmul]
    aggregation anchors, returning structured AttentionSubgraph instances in topological order.
    """
    subgraphs: List[AttentionSubgraph] = []
    seen_anchors: Set[torch.fx.Node] = set()
    node_to_index: Dict[torch.fx.Node, int] = {n: i for i, n in enumerate(graph.nodes)}

    # Reverse search from layer out-projections
    for node in reversed(graph.nodes):
        if node.op == "call_module":
            target_str = str(node.target).lower()
            if any(proj_key in target_str for proj_key in ("out_proj", "o_proj", "c_proj", "dense")):
                out_proj_node = node
                curr_cand = node.args[0] if len(node.args) >= 1 else None

                while isinstance(curr_cand, torch.fx.Node):
                    # Check if anchor is SDPA or fast_linear_attention_replacement
                    is_sdpa = curr_cand.op == "call_function" and (
                        curr_cand.target is fast_linear_attention_replacement
                        or (_SDPA_TARGETS and curr_cand.target in _SDPA_TARGETS)
                        or "scaled_dot_product_attention" in str(curr_cand.target)
                        or getattr(curr_cand.target, "__name__", "") == "scaled_dot_product_attention"
                    )
                    if is_sdpa:
                        if curr_cand not in seen_anchors:
                            seen_anchors.add(curr_cand)
                            q_node = _find_root_proj(curr_cand.args[0]) if len(curr_cand.args) > 0 else None
                            k_node = _find_root_proj(curr_cand.args[1]) if len(curr_cand.args) > 1 else None
                            v_node = _find_root_proj(curr_cand.args[2]) if len(curr_cand.args) > 2 else None
                            scale = curr_cand.kwargs.get("scale", None)
                            if scale is None and len(curr_cand.args) > 6:
                                scale = curr_cand.args[6]
                            subgraphs.append(
                                AttentionSubgraph(
                                    anchor_node=curr_cand,
                                    out_proj_node=out_proj_node,
                                    q_node=q_node,
                                    k_node=k_node,
                                    v_node=v_node,
                                    attn_weights_node=None,
                                    scale=scale,
                                )
                            )
                        break

                    # Check if anchor is decomposed matmul(P, V)
                    target_name = getattr(curr_cand.target, "__name__", str(curr_cand.target))
                    is_matmul = (
                        curr_cand.op == "call_function"
                        and (
                            curr_cand.target in (torch.matmul, operator.matmul, torch.bmm)
                            or target_name in ("matmul", "bmm")
                            or "matmul" in str(curr_cand.target)
                        )
                    ) or (curr_cand.op == "call_method" and curr_cand.target in ("matmul", "bmm"))

                    if is_matmul and len(curr_cand.args) >= 2:
                        p_cand = curr_cand.args[0]
                        v_cand = curr_cand.args[1]
                        if isinstance(p_cand, torch.fx.Node):
                            qk_res = _match_decomposed_qk_matmul(p_cand)
                            if qk_res is not None:
                                if curr_cand not in seen_anchors:
                                    seen_anchors.add(curr_cand)
                                    q_node, k_node, scale = qk_res
                                    v_node = _find_root_proj(v_cand)
                                    subgraphs.append(
                                        AttentionSubgraph(
                                            anchor_node=curr_cand,
                                            out_proj_node=out_proj_node,
                                            q_node=q_node,
                                            k_node=k_node,
                                            v_node=v_node,
                                            attn_weights_node=p_cand,
                                            scale=scale,
                                        )
                                    )
                                break

                    # Step backward through reshape, view, transpose, permute, contiguous
                    if curr_cand.op in ("call_method", "call_function") and any(
                        op_name in str(curr_cand.target)
                        for op_name in ("view", "reshape", "transpose", "permute", "contiguous")
                    ):
                        curr_cand = (
                            curr_cand.args[0]
                            if len(curr_cand.args) >= 1 and isinstance(curr_cand.args[0], torch.fx.Node)
                            else None
                        )
                    else:
                        break

    # Forward scan fallback for anchors not reachable from standard out_proj naming
    for node in graph.nodes:
        if node in seen_anchors:
            continue

        is_sdpa = node.op == "call_function" and (
            node.target is fast_linear_attention_replacement
            or (_SDPA_TARGETS and node.target in _SDPA_TARGETS)
            or "scaled_dot_product_attention" in str(node.target)
            or getattr(node.target, "__name__", "") == "scaled_dot_product_attention"
        )
        if is_sdpa:
            seen_anchors.add(node)
            q_node = _find_root_proj(node.args[0]) if len(node.args) > 0 else None
            k_node = _find_root_proj(node.args[1]) if len(node.args) > 1 else None
            v_node = _find_root_proj(node.args[2]) if len(node.args) > 2 else None
            scale = node.kwargs.get("scale", None)
            if scale is None and len(node.args) > 6:
                scale = node.args[6]
            subgraphs.append(
                AttentionSubgraph(
                    anchor_node=node,
                    out_proj_node=None,
                    q_node=q_node,
                    k_node=k_node,
                    v_node=v_node,
                    attn_weights_node=None,
                    scale=scale,
                )
            )
            continue

        target_name = getattr(node.target, "__name__", str(node.target))
        is_matmul = (
            node.op == "call_function"
            and (
                node.target in (torch.matmul, operator.matmul, torch.bmm)
                or target_name in ("matmul", "bmm")
                or "matmul" in str(node.target)
            )
        ) or (node.op == "call_method" and node.target in ("matmul", "bmm"))

        if is_matmul and len(node.args) >= 2:
            p_cand = node.args[0]
            v_cand = node.args[1]
            if isinstance(p_cand, torch.fx.Node):
                qk_res = _match_decomposed_qk_matmul(p_cand)
                if qk_res is not None:
                    seen_anchors.add(node)
                    q_node, k_node, scale = qk_res
                    v_node = _find_root_proj(v_cand)
                    subgraphs.append(
                        AttentionSubgraph(
                            anchor_node=node,
                            out_proj_node=None,
                            q_node=q_node,
                            k_node=k_node,
                            v_node=v_node,
                            attn_weights_node=p_cand,
                            scale=scale,
                        )
                    )

    # Sort subgraphs by topological order in the graph
    subgraphs.sort(key=lambda sg: node_to_index.get(sg.anchor_node, 0))
    return subgraphs


# ---------------------------------------------------------------------------
# Phase 006: Softmax Operator Isolator & Temperature Scaler Extraction
# ---------------------------------------------------------------------------
class SoftmaxProfile(tuple):
    """Profile container holding isolated softmax node and extracted scaling parameters."""

    node: torch.fx.Node
    scale: float
    softcap: Optional[float]
    mask_node: Optional[torch.fx.Node]
    qk_node: Optional[torch.fx.Node]

    def __new__(
        cls,
        node: torch.fx.Node,
        scale: float,
        softcap: Optional[float] = None,
        mask_node: Optional[torch.fx.Node] = None,
        qk_node: Optional[torch.fx.Node] = None,
    ) -> "SoftmaxProfile":
        instance = super().__new__(cls, (node, scale, softcap))
        instance.node = node
        instance.scale = scale
        instance.softcap = softcap
        instance.mask_node = mask_node
        instance.qk_node = qk_node
        return instance

    def __repr__(self) -> str:
        return (
            f"SoftmaxProfile(node={self.node}, scale={self.scale}, "
            f"softcap={self.softcap}, mask_node={self.mask_node}, qk_node={self.qk_node})"
        )


def _isolate_softmax_pattern(node: torch.fx.Node) -> SoftmaxProfile:
    """Isolates softmax operation over dim=-1 and extracts temperature scale alpha and soft-capping factor C.

    Traverses backwards through graph dependencies starting from candidate or softmax node,
    unpacking optional dropout, masking, and logit soft-capping layers (e.g. Gemma-2).

    Raises:
        ValueError: If softmax dimension is not -1.
    """
    softmax_node = node
    # If the candidate is a dropout node, inspect its input
    if (
        softmax_node.op == "call_function"
        and (softmax_node.target in (torch.dropout, F.dropout) or "dropout" in str(softmax_node.target).lower())
    ) or (softmax_node.op == "call_module" and "dropout" in str(softmax_node.target).lower()):
        if len(softmax_node.args) >= 1 and isinstance(softmax_node.args[0], torch.fx.Node):
            softmax_node = softmax_node.args[0]

    # If the candidate is a matmul(P, V), check if P is softmax or dropout(softmax)
    is_matmul = (
        softmax_node.op == "call_function"
        and (
            softmax_node.target in (torch.matmul, operator.matmul, torch.bmm)
            or getattr(softmax_node.target, "__name__", "") in ("matmul", "bmm")
            or "matmul" in str(softmax_node.target)
            or "bmm" in str(softmax_node.target)
        )
    ) or (softmax_node.op == "call_method" and softmax_node.target in ("matmul", "bmm"))

    if is_matmul and len(softmax_node.args) >= 1 and isinstance(softmax_node.args[0], torch.fx.Node):
        p_cand = softmax_node.args[0]
        if (
            p_cand.op == "call_function"
            and (p_cand.target in (torch.dropout, F.dropout) or "dropout" in str(p_cand.target).lower())
        ) or (p_cand.op == "call_module" and "dropout" in str(p_cand.target).lower()):
            if len(p_cand.args) >= 1 and isinstance(p_cand.args[0], torch.fx.Node):
                p_cand = p_cand.args[0]
        softmax_node = p_cand

    is_sm = (
        softmax_node.op == "call_function"
        and (
            softmax_node.target in (torch.softmax, F.softmax)
            or "softmax" in str(softmax_node.target)
            or getattr(softmax_node.target, "__name__", "") == "softmax"
        )
    ) or (softmax_node.op == "call_method" and softmax_node.target == "softmax")

    if not is_sm:
        raise ValueError(f"Could not isolate softmax pattern from node: {node}")

    # Verify softmax dimension is -1
    dim = softmax_node.kwargs.get("dim", None)
    if dim is None and len(softmax_node.args) > 1:
        dim = softmax_node.args[1]
    if dim is None:
        dim = -1

    if dim != -1:
        raise ValueError(f"Invalid softmax dimension: expected dim=-1, got {dim}")

    scores_node = softmax_node.args[0] if len(softmax_node.args) >= 1 else None
    curr = scores_node
    scale_val: Optional[float] = None
    softcap_val: Optional[float] = None
    mask_node: Optional[torch.fx.Node] = None
    qk_node: Optional[torch.fx.Node] = None

    while isinstance(curr, torch.fx.Node) and curr.op in ("call_function", "call_method"):
        c_target = curr.target
        c_name = getattr(c_target, "__name__", str(c_target))

        # Check for logit soft-capping: C * tanh(scores / C)
        is_mul = (
            c_target in (torch.mul, operator.mul, operator.__mul__)
            or c_name in ("mul", "__mul__")
            or "mul" in str(c_target)
        )
        if is_mul and len(curr.args) >= 2:
            arg0, arg1 = curr.args[0], curr.args[1]
            tanh_node = None
            cand_c = None
            if isinstance(arg0, torch.fx.Node) and (
                arg0.target in (torch.tanh, F.tanh) or getattr(arg0.target, "__name__", "") == "tanh"
            ):
                tanh_node = arg0
                cand_c = arg1
            elif isinstance(arg1, torch.fx.Node) and (
                arg1.target in (torch.tanh, F.tanh) or getattr(arg1.target, "__name__", "") == "tanh"
            ):
                tanh_node = arg1
                cand_c = arg0

            if tanh_node is not None:
                if isinstance(cand_c, (int, float)):
                    softcap_val = float(cand_c)
                inner = tanh_node.args[0] if len(tanh_node.args) >= 1 else None
                if isinstance(inner, torch.fx.Node):
                    if (
                        inner.target in (torch.div, operator.truediv, operator.__truediv__)
                        or getattr(inner.target, "__name__", "") in ("div", "truediv")
                    ) and len(inner.args) >= 1:
                        if softcap_val is None and len(inner.args) >= 2 and isinstance(inner.args[1], (int, float)):
                            softcap_val = float(inner.args[1])
                        curr = inner.args[0]
                        continue
                    elif (
                        inner.target in (torch.mul, operator.mul, operator.__mul__)
                        or getattr(inner.target, "__name__", "") in ("mul", "__mul__")
                    ) and len(inner.args) >= 2:
                        curr = inner.args[0] if isinstance(inner.args[0], torch.fx.Node) else inner.args[1]
                        continue
                    else:
                        curr = inner
                        continue
                break

        # Check for additive mask
        if c_target in (torch.add, operator.add, operator.__add__) or c_name in ("add", "__add__"):
            if isinstance(curr.args[0], torch.fx.Node) and curr.args[0].op in ("call_function", "call_method"):
                mask_node = curr.args[1] if len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node) else None
                curr = curr.args[0]
            elif len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node):
                mask_node = curr.args[0] if isinstance(curr.args[0], torch.fx.Node) else None
                curr = curr.args[1]
            else:
                break
        elif (
            c_target is torch.masked_fill
            or c_name == "masked_fill"
            or (curr.op == "call_method" and curr.target == "masked_fill")
        ):
            mask_node = curr.args[1] if len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node) else None
            curr = curr.args[0]
        elif c_target in (torch.div, operator.truediv, operator.__truediv__) or c_name in ("div", "truediv"):
            if len(curr.args) > 1 and isinstance(curr.args[1], (int, float)):
                scale_val = 1.0 / float(curr.args[1])
            curr = curr.args[0]
        elif is_mul:
            if isinstance(curr.args[0], torch.fx.Node) and curr.args[0].op in ("call_function", "call_method"):
                if len(curr.args) > 1 and isinstance(curr.args[1], (int, float)):
                    scale_val = float(curr.args[1])
                curr = curr.args[0]
            elif len(curr.args) > 1 and isinstance(curr.args[1], torch.fx.Node):
                if isinstance(curr.args[0], (int, float)):
                    scale_val = float(curr.args[0])
                curr = curr.args[1]
            else:
                break
        elif (
            c_target in (torch.matmul, operator.matmul, torch.bmm)
            or c_name in ("matmul", "bmm")
            or "matmul" in str(c_target)
            or "bmm" in str(c_target)
        ):
            qk_node = curr
            break
        else:
            break

    if (
        qk_node is None
        and isinstance(curr, torch.fx.Node)
        and (
            curr.target in (torch.matmul, operator.matmul, torch.bmm)
            or getattr(curr.target, "__name__", "") in ("matmul", "bmm")
            or (curr.op == "call_method" and curr.target in ("matmul", "bmm"))
        )
    ):
        qk_node = curr

    return SoftmaxProfile(
        node=softmax_node,
        scale=scale_val if scale_val is not None else 1.0,
        softcap=softcap_val,
        mask_node=mask_node,
        qk_node=qk_node,
    )


# ---------------------------------------------------------------------------
# Phase 007: AV Matmul Pattern Matcher & Value Collector
# ---------------------------------------------------------------------------
class AVMatchResult(tuple):
    """Result container for matched attention-value contraction O = A @ V."""

    av_node: torch.fx.Node
    v_node: torch.fx.Node
    q_node: Optional[torch.fx.Node]
    k_node: Optional[torch.fx.Node]
    scale: Optional[float]
    softcap: Optional[float]

    def __new__(
        cls,
        av_node: torch.fx.Node,
        v_node: torch.fx.Node,
        q_node: Optional[torch.fx.Node] = None,
        k_node: Optional[torch.fx.Node] = None,
        scale: Optional[float] = None,
        softcap: Optional[float] = None,
    ) -> "AVMatchResult":
        instance = super().__new__(cls, (av_node, v_node))
        instance.av_node = av_node
        instance.v_node = v_node
        instance.q_node = q_node
        instance.k_node = k_node
        instance.scale = scale
        instance.softcap = softcap
        return instance

    @property
    def qkvs(self) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node], torch.fx.Node, Optional[float]]:
        """Returns complete (Q, K, V, scale) core tuple for surgical replacement."""
        return (self.q_node, self.k_node, self.v_node, self.scale)

    def __repr__(self) -> str:
        return (
            f"AVMatchResult(av_node={self.av_node}, v_node={self.v_node}, "
            f"q_node={self.q_node}, k_node={self.k_node}, scale={self.scale}, softcap={self.softcap})"
        )


def _match_av_matmul(
    softmax_node: torch.fx.Node,
    return_qkvs: bool = False,
) -> Union[AVMatchResult, Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node], torch.fx.Node, Optional[float]]]:
    """Matches attention-value context contraction O = A @ V, bypassing intermediate dropout nodes.

    Inspects consumer graph nodes of softmax_node, traverses any intermediate dropout operations,
    and isolates value tensor proxy V as well as upstream query and key representations.

    Returns:
        AVMatchResult unpackable as (av_node, v_node) with .qkvs property providing (Q, K, V, scale).
        If return_qkvs is True, directly returns (Q, K, V, scale).
    """
    profile = _isolate_softmax_pattern(softmax_node)
    scale = profile.scale
    softcap = profile.softcap
    qk_node = profile.qk_node

    q_node: Optional[torch.fx.Node] = None
    k_node: Optional[torch.fx.Node] = None
    if qk_node is not None and len(qk_node.args) >= 2:
        q_node = qk_node.args[0]
        k_cand = qk_node.args[1]
        if isinstance(k_cand, torch.fx.Node):
            if k_cand.op == "call_method" and k_cand.target in ("transpose", "permute"):
                k_node = k_cand.args[0]
            elif k_cand.op == "call_function" and (
                k_cand.target in (torch.transpose, torch.permute)
                or getattr(k_cand.target, "__name__", "") in ("transpose", "permute")
            ):
                k_node = k_cand.args[0]
            else:
                k_node = k_cand

    actual_sm = profile.node
    av_node: Optional[torch.fx.Node] = None
    v_node: Optional[torch.fx.Node] = None

    users = list(actual_sm.users.keys())
    for user in users:
        is_dropout = (
            user.op == "call_function"
            and (user.target in (torch.dropout, F.dropout) or "dropout" in str(user.target).lower())
        ) or (user.op == "call_module" and "dropout" in str(user.target).lower())

        candidate_matmuls = list(user.users.keys()) if is_dropout else [user]

        for cand in candidate_matmuls:
            is_matmul = (
                cand.op == "call_function"
                and (
                    cand.target in (torch.matmul, operator.matmul, torch.bmm)
                    or getattr(cand.target, "__name__", "") in ("matmul", "bmm")
                    or "matmul" in str(cand.target)
                    or "bmm" in str(cand.target)
                )
            ) or (cand.op == "call_method" and cand.target in ("matmul", "bmm"))

            if is_matmul and len(cand.args) >= 2:
                if cand.args[0] == actual_sm or cand.args[0] == user:
                    av_node = cand
                    v_node = cand.args[1]
                    break
        if av_node is not None:
            break

    if av_node is None or v_node is None:
        raise ValueError(f"Could not locate downstream AV matmul for softmax node: {softmax_node}")

    result = AVMatchResult(
        av_node=av_node,
        v_node=v_node,
        q_node=q_node,
        k_node=k_node,
        scale=scale,
        softcap=softcap,
    )
    if return_qkvs:
        return result.qkvs
    return result


# ---------------------------------------------------------------------------
# Phase 008: PyTorch 2.0 SDPA Pattern Matcher (F.scaled_dot_product_attention)
# ---------------------------------------------------------------------------
class SDPAContext(NamedTuple):
    """Context container for matched F.scaled_dot_product_attention nodes."""

    q: torch.fx.Node
    k: torch.fx.Node
    v: torch.fx.Node
    mask: Optional[Union[torch.fx.Node, torch.Tensor]] = None
    is_causal: bool = False
    scale: Optional[float] = None
    dropout_p: float = 0.0
    node: Optional[torch.fx.Node] = None


def _match_sdpa_atomic_node(node: torch.fx.Node) -> Optional[SDPAContext]:
    """Matches PyTorch 2.x native scaled_dot_product_attention call nodes.

    Extracts positional and keyword arguments (q, k, v, mask, is_causal, scale, dropout_p)
    from torch.nn.functional.scaled_dot_product_attention or aten.scaled_dot_product_attention.

    Returns:
        SDPAContext tuple containing extracted tensor proxies and configuration, or None if node is not SDPA.
    """
    if node.op != "call_function":
        return None

    target = node.target
    target_str = str(target)
    target_name = getattr(target, "__name__", "")
    is_sdpa = (
        (_SDPA_TARGETS and target in _SDPA_TARGETS)
        or "scaled_dot_product_attention" in target_str
        or target_name == "scaled_dot_product_attention"
    )
    if not is_sdpa:
        return None

    args = node.args
    kwargs = node.kwargs
    if len(args) < 3:
        return None

    q = args[0]
    k = args[1]
    v = args[2]

    mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    dropout_p = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
    is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
    scale = kwargs.get("scale", args[6] if len(args) > 6 else None)

    return SDPAContext(
        q=q,
        k=k,
        v=v,
        mask=mask,
        is_causal=bool(is_causal),
        scale=scale,
        dropout_p=float(dropout_p) if isinstance(dropout_p, (int, float)) else 0.0,
        node=node,
    )


# ---------------------------------------------------------------------------
