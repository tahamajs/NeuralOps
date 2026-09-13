"""
model_to_3d.py
==============
Multi-Model & Multi-Paradigm Exporter for Live 3D Neural Architecture Visualizer.
Supports:
  Models:
    - distilbert: distilbert-base-uncased
    - roberta:    roberta-base
    - gpt2:       gpt2
    - qwen:       Qwen/Qwen2.5-0.5B (proxy for Qwen3.5-0.8B)

  Selectable Compression Engines / Modules:
    - svd:      Low-Rank SVD Decomposition (W ~= U @ V, r=64) [AASVD]
    - moe:      Sparse Mixture-of-Experts (Router + 4 Parallel Expert Blocks) [ToMoE]
    - mla:      Multi-Head Latent Attention (Latent KV projection bottleneck) [MLA]
    - slicegpt: Structured Slicing & Channel Pruning (45% sliced dimension) [SliceGPT]

  Hierarchical Level of Detail (LOD):
    - Macro Nodes: Consolidated Layer-level bounding volumes for Far / Overview zoom.
    - Granular Nodes: Fine-grained tensor and factor matrices for Close / Detailed zoom.
"""

import os
import re
import json
import math
import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


# Non-linear / structural layers to skip in spatial rendering
SKIP_TYPES = (
    nn.Embedding,
    nn.LayerNorm,
    nn.Dropout,
    nn.GELU,
    nn.ReLU,
    nn.SiLU
)


def instantiate_model_structure(model_key: str):
    """
    Fast structural model loading using AutoConfig.from_pretrained and AutoModel.from_config.
    Caps hidden layers at 12 for high-speed export while preserving visual proportions.
    Returns (model, true_num_layers).
    """
    cfg_map = {
        "distilbert": "distilbert-base-uncased",
        "roberta":    "roberta-base",
        "gpt2":       "gpt2",
        "qwen":       "Qwen/Qwen2.5-0.5B",
    }

    config = AutoConfig.from_pretrained(cfg_map[model_key])

    true_layers = (
        getattr(config, "num_hidden_layers", None) or
        getattr(config, "n_layers", None) or
        getattr(config, "n_layer", None) or
        12
    )

    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(getattr(config, "num_hidden_layers", 12), 12)
    if hasattr(config, "n_layers"):
        config.n_layers = min(config.n_layers, 12)
    if hasattr(config, "n_layer"):
        config.n_layer = min(config.n_layer, 12)

    model = AutoModel.from_config(config)
    model.eval()
    return model, true_layers


def calculate_dim(val, base=768, target=1.5, min_val=0.20):
    """
    Computes spatial 3D dimension strictly proportional to the underlying tensor rank/dimension.
    Area (width x depth) is mathematically proportional to parameter count and memory footprint:
      - 768 x 768 (Dense Attention): 1.5 x 1.5 = 2.25 units
      - 3072 x 768 (Dense FFN Up):    6.0 x 1.5 = 9.00 units (exactly 4x volume!)
      - 64 x 768 (SVD V-Factor):      0.2 x 1.5 = 0.30 units (compressed 1/7.5x volume!)
    """
    if val <= 0:
        return min_val
    return max(min_val, round((val / base) * target, 3))


def format_bytes(num_bytes):
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.1f} KB"
    return f"{num_bytes} B"


def format_params(num_p):
    if num_p >= 1e6:
        return f"{num_p / 1e6:.2f}M"
    elif num_p >= 1e3:
        return f"{num_p / 1e3:.1f}K"
    return str(num_p)


def parse_layer_info(node_id: str):
    m = re.search(r'(?:layer|layers|h)\.(\d+)', node_id)
    if m:
        idx = int(m.group(1))
        return idx, f"Layer {idx}"
    return None, "Global"


LAYER_STRIDE = 3.2


def get_layer_y_base(parent_layer_idx):
    if parent_layer_idx is None:
        return 1.0
    return 1.0 + parent_layer_idx * LAYER_STRIDE


def get_sublayer_offset(node_id: str):
    nid = node_id.lower()
    # Attention Q, K, V
    if any(k in nid for k in ['q_lin', 'query', 'q_proj']):
        return 0.15
    elif any(k in nid for k in ['k_lin', 'key', 'k_proj']):
        return 0.55
    elif any(k in nid for k in ['v_lin', 'value', 'v_proj']):
        return 0.95
    elif 'c_attn' in nid:  # GPT-2 combined QKV
        return 0.55
    # Attention output
    elif any(k in nid for k in ['out_lin', 'output.dense', 'o_proj']) and ('intermediate' not in nid and 'mlp' not in nid and 'ffn' not in nid):
        return 1.45
    elif 'attn.c_proj' in nid:
        return 1.45
    # FFN inputs / intermediate
    elif any(k in nid for k in ['lin1', 'intermediate', 'c_fc', 'gate_proj']):
        return 2.05
    elif 'up_proj' in nid:
        return 2.35
    # FFN output
    elif any(k in nid for k in ['lin2', 'output', 'down_proj', 'mlp.c_proj']):
        return 2.70
    return 1.50


# ---------------------------------------------------------------------------
# 1. Base Dense Spatial Graph Parser
# ---------------------------------------------------------------------------

def parse_dense_baseline_nodes(model):
    nodes = []
    y_pos = 1.0
    seq_idx = 0

    for name, module in model.named_modules():
        if name == "":
            continue

        mod_type_name = module.__class__.__name__.lower()
        if isinstance(module, SKIP_TYPES) or "norm" in mod_type_name or "embed" in mod_type_name:
            continue

        is_linear = isinstance(module, nn.Linear)
        is_conv1d = (module.__class__.__name__ == "Conv1D")

        if is_linear or is_conv1d:
            parent_layer_idx, parent_layer_name = parse_layer_info(name)

            if is_linear:
                in_f = module.in_features
                out_f = module.out_features
                p_count = in_f * out_f + (out_f if module.bias is not None else 0)
            else:
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    in_f = module.weight.shape[0]
                    out_f = module.weight.shape[1]
                else:
                    in_f = getattr(module, "nx", 768)
                    out_f = getattr(module, "nf", 768)
                p_count = in_f * out_f + (out_f if getattr(module, "bias", None) is not None else 0)

            y_base = get_layer_y_base(parent_layer_idx)
            y_val = round(y_base + get_sublayer_offset(name), 2)

            nodes.append({
                "id": name,
                "type": "dense",
                "sub_type": "dense_linear",
                "parent_layer_index": parent_layer_idx,
                "parent_layer_name": parent_layer_name,
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": y_val,
                "z": 0.0,
                "width": calculate_dim(out_f),
                "height": 0.45,
                "depth": calculate_dim(in_f),
                "label": name,
                "metadata": {
                    "shape": [out_f, in_f],
                    "params": format_params(p_count),
                    "params_raw": p_count,
                    "storage": format_bytes(p_count * 4),
                    "storage_bytes": p_count * 4,
                    "dense_equivalent_params": p_count,
                    "dense_equivalent_params_formatted": format_params(p_count),
                    "compression_ratio": 1.0,
                    "compression": "1.0x (Baseline)",
                    "type": "Dense Linear Projection"
                }
            })
            seq_idx += 1

    return nodes


# ---------------------------------------------------------------------------
# 2. Compression Engine Generators
# ---------------------------------------------------------------------------

def generate_svd_module(b_nodes, rank=64):
    """Decomposes each dense projection into V_factor and U_factor."""
    c_nodes = []
    edges = []
    seq_idx = 0

    for b in b_nodes:
        out_f, in_f = b["metadata"]["shape"]
        actual_rank = min(rank, in_f // 4, out_f // 4)
        actual_rank = max(32, actual_rank)

        v_p = in_f * actual_rank
        u_p = actual_rank * out_f + out_f
        total_svd_p = v_p + u_p
        dense_equiv_p = b["metadata"]["params_raw"]
        ratio = round(dense_equiv_p / max(1, total_svd_p), 2)

        v_y = round(b["y"] - 0.14, 2)
        u_y = round(b["y"] + 0.18, 2)

        v_id = f"{b['id']}_v"
        c_nodes.append({
            "id": v_id,
            "type": "svd",
            "sub_type": "v_factor",
            "parent_layer_index": b["parent_layer_index"],
            "parent_layer_name": b["parent_layer_name"],
            "sequence_index": seq_idx,
            "x": 0.0,
            "y": v_y,
            "z": 0.0,
            "width": calculate_dim(actual_rank),
            "height": 0.38,
            "depth": calculate_dim(in_f),
            "label": f"{b['id']}.V_factor",
            "metadata": {
                "shape": [actual_rank, in_f],
                "rank": actual_rank,
                "params": format_params(v_p),
                "params_raw": v_p,
                "storage": format_bytes(v_p * 4),
                "storage_bytes": v_p * 4,
                "dense_equivalent_params": dense_equiv_p,
                "dense_equivalent_params_formatted": format_params(dense_equiv_p),
                "compression_ratio": ratio,
                "compression": f"{ratio:.1f}x",
                "type": "SVD Factor V (Projection)"
            }
        })
        seq_idx += 1

        u_id = f"{b['id']}_u"
        c_nodes.append({
            "id": u_id,
            "type": "svd",
            "sub_type": "u_factor",
            "parent_layer_index": b["parent_layer_index"],
            "parent_layer_name": b["parent_layer_name"],
            "sequence_index": seq_idx,
            "x": 0.0,
            "y": u_y,
            "z": 0.0,
            "width": calculate_dim(out_f),
            "height": 0.38,
            "depth": calculate_dim(actual_rank),
            "label": f"{b['id']}.U_factor",
            "metadata": {
                "shape": [out_f, actual_rank],
                "rank": actual_rank,
                "params": format_params(u_p),
                "params_raw": u_p,
                "storage": format_bytes(u_p * 4),
                "storage_bytes": u_p * 4,
                "dense_equivalent_params": dense_equiv_p,
                "dense_equivalent_params_formatted": format_params(dense_equiv_p),
                "compression_ratio": ratio,
                "compression": f"{ratio:.1f}x",
                "type": "SVD Factor U (Expansion)"
            }
        })
        seq_idx += 1

        edges.append({"from": v_id, "to": u_id, "role": "svd_neck"})

    return c_nodes, edges


def generate_moe_module(b_nodes, num_experts=4):
    """Replaces dense FFN with Gating Router + 4 parallel Expert blocks; SVD on attention."""
    c_nodes = []
    edges = []
    seq_idx = 0

    expert_offsets = [-2.4, -0.8, 0.8, 2.4]

    for b in b_nodes:
        out_f, in_f = b["metadata"]["shape"]
        dense_p = b["metadata"]["params_raw"]
        is_ffn = any(kw in b["id"].lower() for kw in ["ffn", "mlp", "lin1", "lin2", "c_fc", "c_proj", "intermediate"])

        if is_ffn:
            y_base = get_layer_y_base(b["parent_layer_index"])
            router_y = round(y_base + 1.85, 2)
            expert_y = round(y_base + 2.35, 2)

            # Router
            router_p = in_f * num_experts
            router_id = f"{b['id']}_router"
            c_nodes.append({
                "id": router_id,
                "type": "moe",
                "sub_type": "router",
                "parent_layer_index": b["parent_layer_index"],
                "parent_layer_name": b["parent_layer_name"],
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": router_y,
                "z": 0.0,
                "width": calculate_dim(num_experts * 32),
                "height": 0.35,
                "depth": calculate_dim(in_f),
                "label": f"{b['id']}.MoE_Router",
                "metadata": {
                    "shape": [num_experts, in_f],
                    "params": format_params(router_p),
                    "params_raw": router_p,
                    "storage": format_bytes(router_p * 4),
                    "storage_bytes": router_p * 4,
                    "dense_equivalent_params": dense_p,
                    "dense_equivalent_params_formatted": format_params(dense_p),
                    "compression_ratio": round(dense_p / max(1, router_p), 2),
                    "compression": "Dynamic Top-1 Router",
                    "type": "MoE Gating Router"
                }
            })
            seq_idx += 1

            # 4 parallel experts
            expert_p = (in_f * out_f) // 2
            total_active_p = router_p + expert_p
            ratio = round(dense_p / max(1, total_active_p), 2)

            for e_idx in range(num_experts):
                expert_id = f"{b['id']}_expert_{e_idx}"
                c_nodes.append({
                    "id": expert_id,
                    "type": "moe",
                    "sub_type": "expert",
                    "parent_layer_index": b["parent_layer_index"],
                    "parent_layer_name": b["parent_layer_name"],
                    "sequence_index": seq_idx,
                    "x": expert_offsets[e_idx],
                    "y": expert_y,
                    "z": 0.0,
                    "width": calculate_dim(out_f // 2),
                    "height": 0.40,
                    "depth": calculate_dim(in_f),
                    "label": f"{b['id']}.Expert_{e_idx}",
                    "metadata": {
                        "shape": [out_f // 2, in_f],
                        "params": format_params(expert_p),
                        "params_raw": expert_p,
                        "storage": format_bytes(expert_p * 4),
                        "storage_bytes": expert_p * 4,
                        "dense_equivalent_params": dense_p,
                        "dense_equivalent_params_formatted": format_params(dense_p),
                        "compression_ratio": ratio,
                        "compression": f"{ratio:.1f}x Active (Top-1)",
                        "type": f"MoE Sparse Expert #{e_idx}"
                    }
                })
                seq_idx += 1
                edges.append({"from": router_id, "to": expert_id, "role": "moe_branch"})

        else:
            # Attention linear: SVD compression
            actual_rank = min(64, in_f // 4, out_f // 4)
            actual_rank = max(32, actual_rank)
            v_p = in_f * actual_rank
            u_p = actual_rank * out_f + out_f
            total_svd_p = v_p + u_p
            ratio = round(dense_p / max(1, total_svd_p), 2)

            v_y = round(b["y"] - 0.14, 2)
            u_y = round(b["y"] + 0.18, 2)

            v_id = f"{b['id']}_v"
            c_nodes.append({
                "id": v_id,
                "type": "svd",
                "sub_type": "v_factor",
                "parent_layer_index": b["parent_layer_index"],
                "parent_layer_name": b["parent_layer_name"],
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": v_y,
                "z": 0.0,
                "width": calculate_dim(actual_rank),
                "height": 0.38,
                "depth": calculate_dim(in_f),
                "label": f"{b['id']}.V_factor",
                "metadata": {
                    "shape": [actual_rank, in_f],
                    "rank": actual_rank,
                    "params": format_params(v_p),
                    "params_raw": v_p,
                    "storage": format_bytes(v_p * 4),
                    "storage_bytes": v_p * 4,
                    "dense_equivalent_params": dense_p,
                    "dense_equivalent_params_formatted": format_params(dense_p),
                    "compression_ratio": ratio,
                    "compression": f"{ratio:.1f}x",
                    "type": "SVD Factor V (Projection)"
                }
            })
            seq_idx += 1

            u_id = f"{b['id']}_u"
            c_nodes.append({
                "id": u_id,
                "type": "svd",
                "sub_type": "u_factor",
                "parent_layer_index": b["parent_layer_index"],
                "parent_layer_name": b["parent_layer_name"],
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": u_y,
                "z": 0.0,
                "width": calculate_dim(out_f),
                "height": 0.38,
                "depth": calculate_dim(actual_rank),
                "label": f"{b['id']}.U_factor",
                "metadata": {
                    "shape": [out_f, actual_rank],
                    "rank": actual_rank,
                    "params": format_params(u_p),
                    "params_raw": u_p,
                    "storage": format_bytes(u_p * 4),
                    "storage_bytes": u_p * 4,
                    "dense_equivalent_params": dense_p,
                    "dense_equivalent_params_formatted": format_params(dense_p),
                    "compression_ratio": ratio,
                    "compression": f"{ratio:.1f}x",
                    "type": "SVD Factor U (Expansion)"
                }
            })
            seq_idx += 1
            edges.append({"from": v_id, "to": u_id, "role": "svd_neck"})

    return c_nodes, edges


def generate_mla_module(b_nodes, latent_dim=128):
    """Multi-Head Latent Attention compression: compresses KV matrices into low-rank latent vector c_KV."""
    c_nodes = []
    edges = []
    seq_idx = 0

    for b in b_nodes:
        out_f, in_f = b["metadata"]["shape"]
        dense_p = b["metadata"]["params_raw"]
        is_attn = any(kw in b["id"].lower() for kw in ["attn", "query", "key", "value", "q_lin", "k_lin", "v_lin", "q_proj", "k_proj", "v_proj"])

        if is_attn and any(kw in b["id"].lower() for kw in ["k_lin", "v_lin", "key", "value", "k_proj", "v_proj"]):
            # Latent KV compression projection
            latent_p = in_f * latent_dim
            ratio = round(dense_p / max(1, latent_p), 2)
            c_nodes.append({
                "id": f"{b['id']}_latent",
                "type": "mla",
                "sub_type": "latent_kv",
                "parent_layer_index": b["parent_layer_index"],
                "parent_layer_name": b["parent_layer_name"],
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": b["y"],
                "z": 0.0,
                "width": calculate_dim(latent_dim),
                "height": 0.42,
                "depth": calculate_dim(in_f),
                "label": f"{b['id']}.Latent_KV",
                "metadata": {
                    "shape": [latent_dim, in_f],
                    "rank": latent_dim,
                    "params": format_params(latent_p),
                    "params_raw": latent_p,
                    "storage": format_bytes(latent_p * 4),
                    "storage_bytes": latent_p * 4,
                    "dense_equivalent_params": dense_p,
                    "dense_equivalent_params_formatted": format_params(dense_p),
                    "compression_ratio": ratio,
                    "compression": f"{ratio:.1f}x KV Cache Reduction",
                    "type": "MLA Latent KV Projection"
                }
            })
            seq_idx += 1
        else:
            # SVD Low rank
            actual_rank = min(64, in_f // 4, out_f // 4)
            actual_rank = max(32, actual_rank)
            v_p = in_f * actual_rank
            u_p = actual_rank * out_f + out_f
            total_p = v_p + u_p
            ratio = round(dense_p / max(1, total_p), 2)

            v_y = round(b["y"] - 0.14, 2)
            u_y = round(b["y"] + 0.18, 2)

            v_id = f"{b['id']}_v"
            c_nodes.append({
                "id": v_id,
                "type": "svd",
                "sub_type": "v_factor",
                "parent_layer_index": b["parent_layer_index"],
                "parent_layer_name": b["parent_layer_name"],
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": v_y,
                "z": 0.0,
                "width": calculate_dim(actual_rank),
                "height": 0.38,
                "depth": calculate_dim(in_f),
                "label": f"{b['id']}.V_factor",
                "metadata": {
                    "shape": [actual_rank, in_f],
                    "rank": actual_rank,
                    "params": format_params(v_p),
                    "params_raw": v_p,
                    "storage": format_bytes(v_p * 4),
                    "storage_bytes": v_p * 4,
                    "dense_equivalent_params": dense_p,
                    "dense_equivalent_params_formatted": format_params(dense_p),
                    "compression_ratio": ratio,
                    "compression": f"{ratio:.1f}x",
                    "type": "SVD Factor V (Projection)"
                }
            })
            seq_idx += 1

            u_id = f"{b['id']}_u"
            c_nodes.append({
                "id": u_id,
                "type": "svd",
                "sub_type": "u_factor",
                "parent_layer_index": b["parent_layer_index"],
                "parent_layer_name": b["parent_layer_name"],
                "sequence_index": seq_idx,
                "x": 0.0,
                "y": u_y,
                "z": 0.0,
                "width": calculate_dim(out_f),
                "height": 0.38,
                "depth": calculate_dim(actual_rank),
                "label": f"{b['id']}.U_factor",
                "metadata": {
                    "shape": [out_f, actual_rank],
                    "rank": actual_rank,
                    "params": format_params(u_p),
                    "params_raw": u_p,
                    "storage": format_bytes(u_p * 4),
                    "storage_bytes": u_p * 4,
                    "dense_equivalent_params": dense_p,
                    "dense_equivalent_params_formatted": format_params(dense_p),
                    "compression_ratio": ratio,
                    "compression": f"{ratio:.1f}x",
                    "type": "SVD Factor U (Expansion)"
                }
            })
            seq_idx += 1
            edges.append({"from": v_id, "to": u_id, "role": "svd_neck"})

    return c_nodes, edges


def generate_slicegpt_module(b_nodes, sparsity=0.45):
    """Structured Slicing / Pruning: removes 45% of channels along orthogonal rotation basis."""
    c_nodes = []
    edges = []
    seq_idx = 0

    retain = (1.0 - sparsity)

    for b in b_nodes:
        out_f, in_f = b["metadata"]["shape"]
        dense_p = b["metadata"]["params_raw"]

        new_out = int(out_f * retain)
        new_in = int(in_f * retain)
        sliced_p = new_out * new_in + new_out
        ratio = round(dense_p / max(1, sliced_p), 2)

        c_nodes.append({
            "id": f"{b['id']}_sliced",
            "type": "dense",
            "sub_type": "sliced_linear",
            "parent_layer_index": b["parent_layer_index"],
            "parent_layer_name": b["parent_layer_name"],
            "sequence_index": seq_idx,
            "x": 0.0,
            "y": b["y"],
            "z": 0.0,
            "width": calculate_dim(new_out),
            "height": 0.40,
            "depth": calculate_dim(new_in),
            "label": f"{b['id']}.Sliced",
            "metadata": {
                "shape": [new_out, new_in],
                "params": format_params(sliced_p),
                "params_raw": sliced_p,
                "storage": format_bytes(sliced_p * 4),
                "storage_bytes": sliced_p * 4,
                "dense_equivalent_params": dense_p,
                "dense_equivalent_params_formatted": format_params(dense_p),
                "compression_ratio": ratio,
                "compression": f"{ratio:.1f}x (45% Sliced)",
                "type": "SliceGPT Structured Pruning"
            }
        })
        seq_idx += 1

    return c_nodes, edges


# ---------------------------------------------------------------------------
# 3. Macro Aggregated Block Generator (Level of Detail LOD)
# ---------------------------------------------------------------------------

def generate_macro_nodes(nodes, stream_prefix, module_label):
    """
    Groups detailed nodes by parent_layer_index to build consolidated macro bounding boxes.
    Enables smooth Semantic Zooming: macro overview when far away, expanding to fine details up close.
    """
    layers_dict = {}
    for n in nodes:
        l_idx = n.get("parent_layer_index")
        if l_idx is None:
            l_idx = 999
        if l_idx not in layers_dict:
            layers_dict[l_idx] = []
        layers_dict[l_idx].append(n)

    macro_nodes = []

    for l_idx, l_nodes in sorted(layers_dict.items(), key=lambda x: x[0]):
        y_min = min(n["y"] - n["height"] / 2.0 for n in l_nodes)
        y_max = max(n["y"] + n["height"] / 2.0 for n in l_nodes)
        y_center = (y_min + y_max) / 2.0
        h_span = max(1.2, (y_max - y_min) + 0.35)

        max_w = max(n["width"] for n in l_nodes)
        max_d = max(n["depth"] for n in l_nodes)

        tot_p = sum(n["metadata"]["params_raw"] for n in l_nodes)
        tot_b = sum(n["metadata"]["storage_bytes"] for n in l_nodes)
        dense_eq = sum(n["metadata"].get("dense_equivalent_params", n["metadata"]["params_raw"]) for n in l_nodes)
        ratio = round(dense_eq / max(1, tot_p), 2)

        layer_name = l_nodes[0].get("parent_layer_name") or f"Layer {l_idx}"

        macro_nodes.append({
            "id": f"macro_{stream_prefix}_layer_{l_idx}",
            "type": "macro",
            "parent_layer_index": l_idx,
            "parent_layer_name": layer_name,
            "y": round(y_center, 2),
            "width": round(max_w + 0.6, 2),
            "height": round(h_span, 2),
            "depth": round(max_d + 0.6, 2),
            "label": f"{layer_name} [{module_label}]",
            "metadata": {
                "shape": [len(l_nodes), int(tot_p)],
                "params": format_params(tot_p),
                "params_raw": tot_p,
                "storage": format_bytes(tot_b),
                "storage_bytes": tot_b,
                "dense_equivalent_params": dense_eq,
                "dense_equivalent_params_formatted": format_params(dense_eq),
                "compression_ratio": ratio,
                "compression": f"{ratio:.2f}x Block",
                "type": f"Macro {module_label} Block ({len(l_nodes)} Tensors)"
            }
        })

    return macro_nodes


# ---------------------------------------------------------------------------
# 4. Strict Validation
# ---------------------------------------------------------------------------

def validate_nodes(model_name: str, stream_name: str, nodes: list):
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            raise ValueError(f"Validation error in {model_name}/{stream_name}: Node missing 'id'.")

        meta = node.get("metadata")
        if not meta or not isinstance(meta, dict):
            raise ValueError(f"Validation error: Node '{node_id}' in {model_name} missing 'metadata' dict.")

        shape = meta.get("shape")
        if not shape or not isinstance(shape, list) or len(shape) < 2 or any(s <= 0 for s in shape):
            raise ValueError(f"Validation error: Node '{node_id}' in {model_name} has invalid shape: {shape}")

        params = meta.get("params")
        params_raw = meta.get("params_raw")
        if not params or not isinstance(params, str) or params_raw is None or params_raw <= 0:
            raise ValueError(f"Validation error: Node '{node_id}' in {model_name} has invalid params: {params} ({params_raw})")

        storage = meta.get("storage")
        storage_bytes = meta.get("storage_bytes")
        if not storage or not isinstance(storage, str) or storage_bytes is None or storage_bytes <= 0:
            raise ValueError(f"Validation error: Node '{node_id}' in {model_name} has invalid storage: {storage} ({storage_bytes})")

        dense_eq = meta.get("dense_equivalent_params")
        if dense_eq is None or dense_eq <= 0:
            raise ValueError(f"Validation error: Node '{node_id}' in {model_name} has invalid dense_equivalent_params: {dense_eq}")

        ratio = meta.get("compression_ratio")
        if ratio is None or ratio <= 0:
            raise ValueError(f"Validation error: Node '{node_id}' in {model_name} has invalid compression_ratio: {ratio}")


# ---------------------------------------------------------------------------
# 5. Export Execution
# ---------------------------------------------------------------------------

def export_all_models(output_path="spatial-data.json"):
    payload = {
        "title": "Multi-Architecture Neural Compression Atlas",
        "generated_at": "2026-09-12",
        "models": {}
    }

    model_keys = ["distilbert", "roberta", "gpt2", "qwen"]
    summary_rows = []

    for key in model_keys:
        print(f"\n[Processing {key.upper()}]")
        baseline_model, true_layers = instantiate_model_structure(key)
        b_nodes = parse_dense_baseline_nodes(baseline_model)
        b_macro = generate_macro_nodes(b_nodes, "baseline", "Dense Baseline")
        validate_nodes(key, "baseline_nodes", b_nodes)
        validate_nodes(key, "baseline_macro_nodes", b_macro)

        total_b_params = sum(n["metadata"]["params_raw"] for n in b_nodes)

        # Generate each compression module
        modules_data = {}

        module_gens = {
            "svd": ("SVD Low-Rank (AASVD)", generate_svd_module(b_nodes, rank=64)),
            "moe": ("Sparse MoE (ToMoE)", generate_moe_module(b_nodes, num_experts=4)),
            "mla": ("Latent Attention (MLA)", generate_mla_module(b_nodes, latent_dim=128)),
            "slicegpt": ("Structured Pruning (SliceGPT)", generate_slicegpt_module(b_nodes, sparsity=0.45)),
        }

        for mod_key, (mod_label, (c_nodes, c_edges)) in module_gens.items():
            c_macro = generate_macro_nodes(c_nodes, f"comp_{mod_key}", mod_label)
            validate_nodes(key, f"compressed_nodes_{mod_key}", c_nodes)
            validate_nodes(key, f"macro_nodes_{mod_key}", c_macro)

            total_c_params = sum(n["metadata"]["params_raw"] for n in c_nodes)
            overall_ratio = round(total_b_params / max(1, total_c_params), 2)
            storage_saved = max(0, (total_b_params - total_c_params) * 4)

            modules_data[mod_key] = {
                "name": mod_label,
                "compressed_nodes": c_nodes,
                "edges": c_edges,
                "macro_nodes": c_macro,
                "num_compressed": len(c_nodes),
                "total_compressed_params": total_c_params,
                "total_compressed_params_formatted": format_params(total_c_params),
                "overall_compression_ratio": overall_ratio,
                "storage_saved_bytes": storage_saved,
                "storage_saved_formatted": format_bytes(storage_saved),
            }

            summary_rows.append({
                "model": key.upper(),
                "module": mod_key.upper(),
                "b_nodes": len(b_nodes),
                "c_nodes": len(c_nodes),
                "b_params": format_params(total_b_params),
                "c_params": format_params(total_c_params),
                "ratio": f"{overall_ratio:.2f}x"
            })

        default_mod = modules_data["svd"]

        all_y = [n["y"] for n in (b_nodes + default_mod["compressed_nodes"])]
        max_y = (max(all_y) + 2.5) if all_y else 14.0

        payload["models"][key] = {
            "true_num_layers": true_layers,
            "capped_num_layers": min(true_layers, 12),
            "baseline_nodes": b_nodes,
            "baseline_macro_nodes": b_macro,
            "compressed_nodes": default_mod["compressed_nodes"],
            "macro_nodes": default_mod["macro_nodes"],
            "edges": default_mod["edges"],
            "max_y": round(max_y, 2),
            "num_baseline": len(b_nodes),
            "num_compressed": default_mod["num_compressed"],
            "total_baseline_params": total_b_params,
            "total_baseline_params_formatted": format_params(total_b_params),
            "total_compressed_params": default_mod["total_compressed_params"],
            "total_compressed_params_formatted": default_mod["total_compressed_params_formatted"],
            "overall_compression_ratio": default_mod["overall_compression_ratio"],
            "storage_saved_bytes": default_mod["storage_saved_bytes"],
            "storage_saved_formatted": default_mod["storage_saved_formatted"],
            "modules": modules_data
        }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] Exported {len(payload['models'])} models with 4 compression paradigms to {output_path}")

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'Model':<12} | {'Module':<10} | {'Base Nodes':<10} | {'Comp Nodes':<10} | {'Base Params':<14} | {'Comp Params':<14} | {'Overall Ratio':<12}")
    print("-" * 105)
    for row in summary_rows:
        print(f"{row['model']:<12} | {row['module']:<10} | {row['b_nodes']:<10} | {row['c_nodes']:<10} | {row['b_params']:<14} | {row['c_params']:<14} | {row['ratio']:<12}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "spatial-data.json")
    export_all_models(out_file)
