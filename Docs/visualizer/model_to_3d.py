"""
model_to_3d.py
==============
Fast Multi-Model Exporter for Live 3D Neural Architecture Visualizer.
Supports 4 Transformer Families:
  - distilbert: distilbert-base-uncased
  - roberta:    roberta-base
  - gpt2:       gpt2
  - qwen:       Qwen/Qwen2.5-0.5B (proxy for Qwen3.5-0.8B)

Key Features:
1. Fast config-based structural loading (No weight downloads).
2. Explicit filtering of non-linear layers (Embedding, LayerNorm, RMSNorm, Dropout, activations).
3. Real deepcopy compression transformation (SVDLinear replacement).
4. Handles Conv1D (GPT-2) and nn.Linear.
5. Grouped export format into spatial-data.json with bounding metrics.
"""

import os
import json
import math
import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# ---------------------------------------------------------------------------
# SVD Factorized Module Definition
# ---------------------------------------------------------------------------

class SVDLinear(nn.Module):
    """Low-rank SVD linear factorization: W ~= U @ V"""
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.v = nn.Linear(in_features, self.rank, bias=False)
        self.u = nn.Linear(self.rank, out_features, bias=True)

    def forward(self, x):
        return self.u(self.v(x))


# Non-linear / structural layers to skip in spatial rendering
SKIP_TYPES = (
    nn.Embedding,
    nn.LayerNorm,
    nn.Dropout,
    nn.GELU,
    nn.ReLU,
    nn.SiLU
)

# ---------------------------------------------------------------------------
# A1. Fast Config-Based Model Loading (Zero Weight Downloads)
# ---------------------------------------------------------------------------

def instantiate_model_structure(model_key: str):
    """
    Fast structural model loading using AutoConfig.from_pretrained and AutoModel.from_config.
    Caps hidden layers at 12 for high-speed export while preserving visual proportions.
    """
    cfg_map = {
        "distilbert": "distilbert-base-uncased",
        "roberta":    "roberta-base",
        "gpt2":       "gpt2",
        "qwen":       "Qwen/Qwen2.5-0.5B",
    }

    print(f" -> Loading config for {model_key} ({cfg_map[model_key]})...")
    config = AutoConfig.from_pretrained(cfg_map[model_key])

    # Cap depth for fast, uniform rendering while keeping depth differences visible
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(getattr(config, "num_hidden_layers", 12), 12)
    if hasattr(config, "n_layers"):
        config.n_layers = min(config.n_layers, 12)
    if hasattr(config, "n_layer"):
        config.n_layer = min(config.n_layer, 12)

    model = AutoModel.from_config(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helper: Get Parent Module
# ---------------------------------------------------------------------------

def _get_parent(model, dotted_name):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p, None)
        if parent is None:
            return None, ""
    return parent, parts[-1]


# ---------------------------------------------------------------------------
# A3. Deep Copy & Real Compression Transformation
# ---------------------------------------------------------------------------

def make_compressed_copy(baseline_model, rank=64):
    """
    Deep copy the baseline model, then replace eligible nn.Linear and Conv1D layers
    with SVDLinear modules.
    """
    compressed = copy.deepcopy(baseline_model)

    for name, module in list(compressed.named_modules()):
        if name == "" or any(skip in name.lower() for skip in [
            "classifier", "score", "lm_head", "pooler", "embed", "wte", "wpe"
        ]):
            continue

        in_f = None
        out_f = None

        # Standard PyTorch nn.Linear
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
        # HuggingFace GPT-2 Conv1D
        elif module.__class__.__name__ == "Conv1D":
            if hasattr(module, "weight") and module.weight.ndim == 2:
                in_f = module.weight.shape[0]
                out_f = module.weight.shape[1]
            elif hasattr(module, "nx") and hasattr(module, "nf"):
                in_f = module.nx
                out_f = module.nf

        if in_f is not None and out_f is not None:
            if min(in_f, out_f) >= 256:
                parent, leaf = _get_parent(compressed, name)
                if parent is not None and leaf:
                    actual_rank = min(rank, in_f // 4, out_f // 4)
                    actual_rank = max(32, actual_rank)
                    new_layer = SVDLinear(in_f, out_f, rank=actual_rank)
                    setattr(parent, leaf, new_layer)

    return compressed


# ---------------------------------------------------------------------------
# 3D Geometry Dimension Scalers
# ---------------------------------------------------------------------------

def calculate_scale(val):
    if val <= 0:
        return 0.5
    return max(0.4, round(math.sqrt(val) / 12.0, 3))


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


# ---------------------------------------------------------------------------
# Spatial Graph Parser
# ---------------------------------------------------------------------------

def parse_model_to_3d_nodes(model, side_offset=0.0):
    nodes = []
    edges = []
    y_pos = 1.0

    visited = set()

    for name, module in model.named_modules():
        if name == "" or name in visited:
            continue

        # Check for RMSNorm or other normalization subclasses dynamically
        mod_type_name = module.__class__.__name__.lower()
        if isinstance(module, SKIP_TYPES) or "norm" in mod_type_name or "embed" in mod_type_name:
            continue

        # 1. SVDLinear (Compressed representation)
        if isinstance(module, SVDLinear):
            visited.update([f"{name}.v", f"{name}.u"])
            rank = module.rank
            in_f = module.in_features
            out_f = module.out_features

            v_p = in_f * rank
            u_p = rank * out_f + out_f
            dense_p = in_f * out_f
            ratio = dense_p / max(1, (v_p + u_p))

            v_id = f"{name}_v"
            nodes.append({
                "id": v_id,
                "type": "svd",
                "sub_type": "v_factor",
                "x": side_offset,
                "y": round(y_pos, 2),
                "z": 0.0,
                "width": calculate_scale(rank),
                "height": 0.38,
                "depth": calculate_scale(in_f),
                "label": f"{name}.V_factor",
                "metadata": {
                    "shape": [rank, in_f],
                    "rank": rank,
                    "params": format_params(v_p),
                    "storage": format_bytes(v_p * 4),
                    "compression": f"{ratio:.1f}x",
                    "type": "SVD Factor V (Projection)"
                }
            })

            y_pos += 0.55
            u_id = f"{name}_u"
            nodes.append({
                "id": u_id,
                "type": "svd",
                "sub_type": "u_factor",
                "x": side_offset,
                "y": round(y_pos, 2),
                "z": 0.0,
                "width": calculate_scale(out_f),
                "height": 0.38,
                "depth": calculate_scale(rank),
                "label": f"{name}.U_factor",
                "metadata": {
                    "shape": [out_f, rank],
                    "rank": rank,
                    "params": format_params(u_p),
                    "storage": format_bytes(u_p * 4),
                    "compression": f"{ratio:.1f}x",
                    "type": "SVD Factor U (Expansion)"
                }
            })

            edges.append({"from": v_id, "to": u_id, "role": "svd_neck"})
            y_pos += 1.3
            continue

        # 2. Linear / Conv1D (Dense baseline representation)
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = (module.__class__.__name__ == "Conv1D")

        if is_linear or is_conv1d:
            parts = name.split(".")
            is_child_of_handled = False
            for p_len in range(1, len(parts)):
                parent_path = ".".join(parts[:p_len])
                if parent_path in visited:
                    is_child_of_handled = True
                    break

            if is_child_of_handled:
                continue

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

            nodes.append({
                "id": name,
                "type": "dense",
                "sub_type": "dense_linear",
                "x": side_offset,
                "y": round(y_pos, 2),
                "z": 0.0,
                "width": calculate_scale(out_f),
                "height": 0.45,
                "depth": calculate_scale(in_f),
                "label": name,
                "metadata": {
                    "shape": [out_f, in_f],
                    "params": format_params(p_count),
                    "storage": format_bytes(p_count * 4),
                    "type": "Dense Linear Projection"
                }
            })
            y_pos += 1.25

    return nodes, edges


# ---------------------------------------------------------------------------
# A4. Export Grouped JSON Structure
# ---------------------------------------------------------------------------

def export_all_models(output_path="spatial-data.json"):
    payload = {
        "title": "Multi-Architecture Neural Compression Atlas",
        "generated_at": "2026-09-11",
        "models": {}
    }

    model_keys = ["distilbert", "roberta", "gpt2", "qwen"]

    for key in model_keys:
        print(f"\n[Processing {key.upper()}]")
        baseline = instantiate_model_structure(key)
        compressed = make_compressed_copy(baseline, rank=64)

        b_nodes, b_edges = parse_model_to_3d_nodes(baseline, side_offset=0.0)
        c_nodes, c_edges = parse_model_to_3d_nodes(compressed, side_offset=0.0)

        all_y = [n["y"] for n in (b_nodes + c_nodes)]
        max_y = (max(all_y) + 2.5) if all_y else 14.0

        payload["models"][key] = {
            "baseline_nodes":   b_nodes,
            "compressed_nodes": c_nodes,
            "edges":            b_edges + c_edges,
            "max_y":            round(max_y, 2),
            "num_baseline":     len(b_nodes),
            "num_compressed":   len(c_nodes),
        }
        print(f" -> {key.upper()}: {len(b_nodes)} baseline nodes, {len(c_nodes)} compressed nodes, max_y={max_y:.1f}")

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] Exported {len(payload['models'])} models to {output_path}")


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "spatial-data.json")
    export_all_models(out_file)
