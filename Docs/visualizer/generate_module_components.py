#!/usr/bin/env python3
"""
generate_module_components.py
=============================
Parses every attention module and compression engine in the codebase,
designs their detailed 3D architectural components and tensor topology across
the 4 transformer models (DistilBERT, RoBERTa, GPT-2, Qwen3.5), and writes:
  1. Individual component files: Docs/visualizer/components/<module_id>.json
  2. Consolidated component registry: Docs/visualizer/modules_components.json
"""

import os
import glob
import ast
import re
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR)

COMPONENTS_DIR = os.path.join(BASE_DIR, "Docs", "visualizer", "components")
os.makedirs(COMPONENTS_DIR, exist_ok=True)

MODEL_SPECS = {
    "distilbert": {
        "family": "Encoder",
        "hidden_size": 768,
        "num_layers": 6,
        "num_heads": 12,
        "intermediate_size": 3072,
        "qkv_same": True
    },
    "roberta": {
        "family": "Encoder",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "qkv_same": True
    },
    "gpt2": {
        "family": "Causal Decoder",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "qkv_same": True
    },
    "qwen": {
        "family": "GQA Decoder",
        "hidden_size": 896,
        "num_layers": 24,
        "num_heads": 14,
        "num_kv_heads": 2,
        "intermediate_size": 4864,
        "qkv_same": False
    }
}


def analyze_file(file_path):
    """Parses a module file with AST and regex to extract structural components."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    is_attn = "attentions" in file_path

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()

    classes_found = []
    methods_found = []
    submodules_referenced = []
    docstring = ""

    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                methods_found.append(node.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "self":
                    submodules_referenced.append(node.attr)
        docstring = ast.get_docstring(tree) or ""
    except Exception:
        doc_m = re.search(r'"""(.*?)"""', src, re.DOTALL)
        if doc_m:
            docstring = doc_m.group(1).strip()

    first_line = docstring.split("\n")[0].strip() if docstring else base_name.replace("_", " ").title()
    first_line = re.sub(r'^[#\s\-*]+', '', first_line)

    # Markdown documentation math search
    md_dir = "Docs/content/attentions" if is_attn else "Docs/content/compressions"
    md_path = os.path.join(md_dir, f"{base_name}.md")
    math_latex = ""
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8", errors="ignore") as mf:
            mfc = mf.read()
            math_m = re.search(r'\$\$(.*?)\$\$', mfc, re.DOTALL)
            if math_m:
                math_latex = math_m.group(1).strip()

    low = (base_name + " " + docstring + " " + " ".join(classes_found)).lower()

    # Determine paradigm and component architecture
    if any(k in low for k in ['svd', 'low_rank', 'factor', 'rank', 'rpca', 'nmf', 'cur_linear', 'saes', 'lora']):
        paradigm = 'svd'
        category = 'Low-Rank SVD & Factorization'
        icon = '⚡'
    elif any(k in low for k in ['moe', 'expert', 'router', 'switch', 'routing', 'bam_umoe']):
        paradigm = 'moe'
        category = 'Mixture of Experts & Routing'
        icon = '🔀'
    elif any(k in low for k in ['mamba', 'ssm', 'linear_attention', 'deltanet', 'hgrn', 'griffin', 'cosformer', 'decay', 'simple_ssm']):
        paradigm = 'ssm'
        category = 'Linear Attention & State-Space (SSM)'
        icon = '🐍'
    elif any(k in low for k in ['mla', 'latent', 'cache', 'kvtc', 'cachenotes', 'hybridkv']):
        paradigm = 'mla'
        category = 'Latent & KV-Cache Compression'
        icon = '🧠'
    elif any(k in low for k in ['slice', 'prun', 'admm', 'prox', 'bonsai', 'flap', 'slim', 'sparsegpt']):
        paradigm = 'pruning'
        category = 'Structured Pruning & Channel Slicing'
        icon = '✂️'
    elif any(k in low for k in ['quant', 'int4', 'fp8', 'int8', 'astro', 'coopq']):
        paradigm = 'quant'
        category = 'Quantization & Mixed Precision'
        icon = '💎'
    else:
        paradigm = 'sparse'
        category = 'Dynamic & Efficient Attention'
        icon = '🌟'

    # Synthesize component topology
    components = []
    if paradigm == 'svd':
        components = [
            {"id": "v_factor", "name": "Factor V (Low-Rank Subspace Projection)", "role": "Projection to Rank r", "shape_transform": "[r × d_in]", "color": "#3acbb8"},
            {"id": "u_factor", "name": "Factor U (Full-Rank Expansion Projection)", "role": "Expansion to Output Space", "shape_transform": "[d_out × r]", "color": "#2dd4bf"},
            {"id": "adapter", "name": "Residual LoRA / Norm Calibration", "role": "Fidelity Compensation", "shape_transform": "[d_out × d_in] Residual", "color": "#5eead4"}
        ]
        if not math_latex:
            math_latex = r"W \approx U \cdot V, \quad U \in \mathbb{R}^{d_{out} \times r}, \; V \in \mathbb{R}^{r \times d_{in}}"
    elif paradigm == 'moe':
        components = [
            {"id": "router", "name": "Top-K Gating Router", "role": "Sparse Token Dispatch", "shape_transform": "[K × d_in] Softmax", "color": "#b57ee0"},
            {"id": "expert_0", "name": "Specialized Feed-Forward Expert 0", "role": "Semantic Representation Subspace", "shape_transform": "[d_ffn // 2 × d_in]", "color": "#c084fc"},
            {"id": "expert_1", "name": "Specialized Feed-Forward Expert 1", "role": "Semantic Representation Subspace", "shape_transform": "[d_ffn // 2 × d_in]", "color": "#a855f7"},
            {"id": "expert_2", "name": "Specialized Feed-Forward Expert 2", "role": "Semantic Representation Subspace", "shape_transform": "[d_ffn // 2 × d_in]", "color": "#9333ea"},
            {"id": "expert_3", "name": "Specialized Feed-Forward Expert 3", "role": "Semantic Representation Subspace", "shape_transform": "[d_ffn // 2 × d_in]", "color": "#7e22ce"}
        ]
        if not math_latex:
            math_latex = r"y = \sum_{i=1}^K G(x)_i \cdot \text{Expert}_i(x), \quad G(x) = \text{Softmax}(\text{TopK}(x W_g))"
    elif paradigm == 'ssm':
        components = [
            {"id": "in_proj", "name": "Input SVD / Linear Projection", "role": "Channel Expansion", "shape_transform": "[2 \cdot d_{in} × d_{in}]", "color": "#34d399"},
            {"id": "conv1d", "name": "Depthwise Causal 1D Convolution", "role": "Local Token Mixing", "shape_transform": "[d_{inner} × kernel=4]", "color": "#10b981"},
            {"id": "ssm_recurrence", "name": "Selective SSM Recurrence Matrix (A, B, C)", "role": "Continuous-Time State Space", "shape_transform": "[d_{state}=16 × d_{inner}]", "color": "#059669"},
            {"id": "out_proj", "name": "Output Gated Projection", "role": "Dimension Recovery", "shape_transform": "[d_{in} × d_{inner}]", "color": "#047857"}
        ]
        if not math_latex:
            math_latex = r"h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t"
    elif paradigm == 'mla':
        components = [
            {"id": "down_proj_kv", "name": "Latent KV Compression Projection", "role": "KV Cache Reduction", "shape_transform": "[d_c × d_{in}]", "color": "#f59e0b"},
            {"id": "up_proj_k", "name": "Key Dequantize / Unpack", "role": "Head Expansion", "shape_transform": "[n_h \cdot d_h × d_c]", "color": "#d97706"},
            {"id": "up_proj_v", "name": "Value Dequantize / Unpack", "role": "Head Expansion", "shape_transform": "[n_h \cdot d_h × d_c]", "color": "#b45309"},
            {"id": "q_rope", "name": "Decoupled RoPE Position Matrix", "role": "Rotary Geometry Injection", "shape_transform": "[d_R × d_{in}]", "color": "#fbbf24"}
        ]
        if not math_latex:
            math_latex = r"c^{KV}_t = W_{DKV} x_t, \quad k^C_t = W_{UK} c^{KV}_t, \quad v^C_t = W_{UV} c^{KV}_t"
    elif paradigm == 'pruning':
        components = [
            {"id": "orthogonal_rot", "name": "Orthogonal Transformation Matrix (Q)", "role": "Energy Concentration", "shape_transform": "[d_{in} × d_{in}] Orthogonal", "color": "#38bdf8"},
            {"id": "sliced_linear", "name": "Channel-Sliced Dense Linear Operator", "role": "45% Channel Elimination", "shape_transform": "[0.55 \cdot d_{out} × 0.55 \cdot d_{in}]", "color": "#0ea5e9"},
            {"id": "bias_correct", "name": "Mean Bias Correction Vector", "role": "Representation Recalibration", "shape_transform": "[0.55 \cdot d_{out}]", "color": "#0284c7"}
        ]
        if not math_latex:
            math_latex = r"W' = Q_2^T W Q_1, \quad W_{sliced} = W'[:d'_1, :d'_2]"
    elif paradigm == 'quant':
        components = [
            {"id": "scale_quant", "name": "Channel / Block Scales (FP8 / INT4)", "role": "Dynamic Range Quantization", "shape_transform": "[d_{out} // block_size]", "color": "#a855f7"},
            {"id": "packed_weight", "name": "Packed Low-Bit Integer Weight Matrix", "role": "4-bit Memory Footprint", "shape_transform": "[d_{out} × d_{in} // 2]", "color": "#c084fc"},
            {"id": "dequant_kernel", "name": "Fused GEMM Dequantization Kernel", "role": "Zero-Overhead Accumulation", "shape_transform": "TensorCore Fused", "color": "#e9d5ff"}
        ]
        if not math_latex:
            math_latex = r"W = S \odot (Q_W - Z), \quad Q_W \in \{-8, \dots, 7\}"
    else:
        components = [
            {"id": "sparse_mask", "name": "Dynamic Top-K Attention Sparsity Mask", "role": "O(N sqrt{N}) Token Filtering", "shape_transform": "[Seq × Seq] Sparse", "color": "#38bdf8"},
            {"id": "kernel_attn", "name": "Linearized Positive Kernel Projection", "role": "Softmax Approximation", "shape_transform": "[\phi(Q) \cdot \phi(K)^T]", "color": "#7dd3fc"},
            {"id": "out_norm", "name": "Output Layer Normalization", "role": "Attention Scale Stabilization", "shape_transform": "[d_{in}]", "color": "#bae6fd"}
        ]
        if not math_latex:
            math_latex = r"A_{ij} = \text{TopK}(\text{Softmax}(\frac{Q_i K_j^T}{\sqrt{d}}))"

    # Compute concrete parameters across all 4 models
    model_projections = {}
    for m_name, spec in MODEL_SPECS.items():
        h = spec["hidden_size"]
        ffn = spec["intermediate_size"]
        dense_qkv_params = (h * h) * 3 + (h * ffn * 2)
        
        if paradigm == 'svd':
            r = 64
            comp_qkv_params = (h * r + r * h) * 3 + (h * r + r * ffn) * 2
        elif paradigm == 'moe':
            comp_qkv_params = (h * h) * 3 + 4 * (h * (ffn // 2) * 2)
        elif paradigm == 'mla':
            comp_qkv_params = (h * h) + 2 * (h * 128) + (h * ffn * 2)
        elif paradigm == 'pruning':
            comp_qkv_params = int(dense_qkv_params * 0.30)
        elif paradigm == 'quant':
            comp_qkv_params = dense_qkv_params // 4
        else:
            comp_qkv_params = int(dense_qkv_params * 0.55)

        model_projections[m_name] = {
            "hidden_size": h,
            "intermediate_size": ffn,
            "baseline_layer_params": dense_qkv_params,
            "compressed_layer_params": comp_qkv_params,
            "savings_ratio": round(dense_qkv_params / max(1, comp_qkv_params), 2)
        }

    return {
        "id": base_name,
        "name": first_line[:75],
        "file": file_path,
        "type": "Attention Module" if is_attn else "Compression Engine",
        "category": category,
        "paradigm": paradigm,
        "icon": icon,
        "description": docstring[:400].replace("\n", " ").strip() or f"Production-grade {category} module.",
        "classes": classes_found,
        "methods_count": len(methods_found),
        "math_latex": math_latex,
        "components": components,
        "model_projections": model_projections
    }


def main():
    attn_files = sorted(glob.glob("Codes/framework/attentions/*.py"))
    comp_files = sorted(glob.glob("Codes/framework/compression_engine/*.py"))
    all_files = [f for f in (attn_files + comp_files) if not os.path.basename(f).startswith("__")]

    print(f"[Analyzing {len(all_files)} files in repository...]")
    all_modules = []

    for fpath in all_files:
        info = analyze_file(fpath)
        all_modules.append(info)

        # Write individual module component JSON
        comp_file = os.path.join(COMPONENTS_DIR, f"{info['id']}.json")
        with open(comp_file, "w", encoding="utf-8") as out:
            json.dump(info, out, indent=2)

    # Write consolidated registry
    reg_path = os.path.join(BASE_DIR, "Docs", "visualizer", "modules_components.json")
    with open(reg_path, "w", encoding="utf-8") as out:
        json.dump({"total": len(all_modules), "modules": all_modules}, out, indent=2)

    print(f"[SUCCESS] Exported {len(all_modules)} individual component files to {COMPONENTS_DIR}")
    print(f"[SUCCESS] Exported consolidated registry to {reg_path}")


if __name__ == "__main__":
    main()
