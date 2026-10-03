#!/usr/bin/env python3
"""
export_module_catalog.py
========================
Indexes all 240+ attention modules and compression engines in the repository,
extracting docstrings, categories, mathematical formulations, and metadata
into Docs/visualizer/modules-catalog.json.
"""

import os
import glob
import re
import json


def extract_module_info(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    is_attn = "attentions" in file_path

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    doc_match = re.search(r"\"\"\"(.*?)\"\"\"", content, re.DOTALL)
    raw_doc = doc_match.group(1).strip() if doc_match else ""
    first_line = raw_doc.split("\n")[0].strip() if raw_doc else base_name.replace("_", " ").title()
    first_line = re.sub(r"^[#\s\-*]+", "", first_line)

    md_dir = "Docs/content/attentions" if is_attn else "Docs/content/compressions"
    md_path = os.path.join(md_dir, f"{base_name}.md")
    math_formula = ""
    doc_markdown = ""

    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8", errors="ignore") as mf:
            md_content = mf.read()
            doc_markdown = md_content[:600]
            math_match = re.search(r"\$\$(.*?)\$\$", md_content, re.DOTALL)
            if math_match:
                math_formula = math_match.group(1).strip()
            header_match = re.search(r"#\s*(.*?)\n", md_content)
            if header_match:
                title = header_match.group(1).strip()
                if len(title) > len(first_line) and not title.lower().startswith("module"):
                    first_line = title

    low = (base_name + " " + raw_doc).lower()
    if any(
        k in low for k in ["svd", "low_rank", "factor", "rank", "rpca", "nmf", "cur_linear", "saes"]
    ):
        cat = "Low-Rank SVD & Factorization"
        paradigm = "svd"
        icon = "⚡"
    elif any(k in low for k in ["moe", "expert", "router", "switch", "routing", "bam_umoe"]):
        cat = "Mixture of Experts & Routing"
        paradigm = "moe"
        icon = "🔀"
    elif any(
        k in low
        for k in [
            "mamba",
            "ssm",
            "linear_attention",
            "deltanet",
            "hgrn",
            "griffin",
            "cosformer",
            "decay",
            "simple_ssm",
        ]
    ):
        cat = "Linear Attention & State-Space (SSM)"
        paradigm = "ssm"
        icon = "🐍"
    elif any(k in low for k in ["mla", "kv", "latent", "cache", "kvtc", "cachenotes", "hybridkv"]):
        cat = "Latent & KV-Cache Compression"
        paradigm = "mla"
        icon = "🧠"
    elif any(
        k in low for k in ["slice", "prun", "admm", "prox", "bonsai", "flap", "slim", "sparsegpt"]
    ):
        cat = "Structured Pruning & Channel Slicing"
        paradigm = "pruning"
        icon = "✂️"
    elif any(k in low for k in ["quant", "int4", "fp8", "int8", "astro", "coopq"]):
        cat = "Quantization & Mixed Precision"
        paradigm = "quant"
        icon = "💎"
    else:
        cat = "Dynamic & Efficient Attention"
        paradigm = "sparse"
        icon = "🌟"

    # Specific mathematical formulations dictionary for modules
    EXPLICIT_MODULE_MATH = {
        "aasvd": r"W \approx U_r \Sigma_r V_r^\top, \quad \min_{U, \Sigma, V} \left\| (W - U \Sigma V^\top) X^\top \right\|_F^2",
        "cur_moe_compressor": r"W \approx C \cdot U \cdot R, \quad C = W_{[:, J]}, \; R = W_{[I, :]}, \; U = C^\dagger W R^\dagger",
        "frandi_compressor": r"\Omega \sim \mathcal{N}(0, I), \; Y = (W W^\top)^q W \Omega, \; Q = \text{qr}(Y), \; W \approx Q (Q^\top W)",
        "grecodec_compressor": r"\min_{u, v} \|W - u v^\top\|_F^2 \implies u = \frac{W v}{\|v\|^2}, \; v = \frac{W^\top u}{\|u\|^2}",
        "hybrid_svd_layer": r"W = U_r \Sigma_r V_r^\top + M_{\text{sparse}}, \quad \|M_{\text{sparse}}\|_0 \le k",
        "kvtc_compressor": r"K_{\text{comp}} = K V_K^\top, \; V_{\text{comp}} = V V_V^\top, \quad V_K, V_V \in \mathbb{R}^{r \times d_k}",
        "lsc_compressor": r"\tilde{W}_{u,v} = \sum_{n=0}^{N-1} \sum_{m=0}^{M-1} W_{nm} \cos\left(\frac{\pi u (2n+1)}{2N}\right) \cos\left(\frac{\pi v (2m+1)}{2M}\right)",
        "mlorq_compressor": r"W \approx \text{Quant}(A) \cdot B^\top + S \cdot \text{scale}, \quad A \in \mathbb{R}^{d_{out} \times r}",
        "performer_attention": r"\text{Attn}(Q, K, V) = D^{-1} (\phi(Q) (\phi(K)^\top V)), \quad \phi(x) = \frac{h(x)}{\sqrt{m}} \exp\left(\omega^\top x - \frac{\|x\|^2}{2}\right)",
        "proxsparse_pruner": r"W^{(t+1)} = \text{prox}_{\lambda \|\cdot\|_1}\left(W^{(t)} - \eta \nabla f(W^{(t)})\right) = \text{sign}(W) \max(0, |W| - \lambda \eta)",
        "rpca_compressor": r"\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad W = L + S, \quad \text{rank}(L) \le r",
        "saes_svd_v2": r"W \approx U_r \Sigma_r V_r^\top + \text{SAE}(W - U_r \Sigma_r V_r^\top)",
        "slicegpt_compressor": r"W_{\text{pruned}} = (Q^\top W Q)_{[:, \mathcal{I}]}, \quad Q = \text{eigh}(X X^\top), \; |\mathcal{I}| = (1 - 0.45) d_{in}",
        "sparsegpt": r"\Delta W = -\frac{w_q}{[H^{-1}]_{qq}} H_{:, q}^{-1}, \quad H = 2 X X^\top",
        "squeezing_heads_distiller": r"\mathcal{L} = \alpha D_{\text{KL}}(A_{\text{teacher}} \parallel A_{\text{student}}) + \beta \|V_{\text{teacher}} - V_{\text{student}} W_{\text{proj}}\|_F^2",
        "svd_linear": r"W \approx U_r \Sigma_r V_r^\top = (U_r \sqrt{\Sigma_r}) (\sqrt{\Sigma_r} V_r^\top) = A B^\top",
        "swift_svd_act": r"\min_{A, B} \left\|(W - A B) \Sigma_{XX}^{1/2}\right\|_F^2, \quad \Sigma_{XX} = \frac{1}{N} X X^\top",
    }

    if base_name in EXPLICIT_MODULE_MATH:
        math_formula = EXPLICIT_MODULE_MATH[base_name]
    elif not math_formula or len(math_formula) < 15:
        if paradigm == "svd":
            math_formula = (
                r"W \approx U_r \Sigma_r V_r^\top = A B^\top, \quad r \ll \min(d_{in}, d_{out})"
            )
        elif paradigm == "moe":
            math_formula = r"y = \sum_{i=1}^K G(x)_i \cdot E_i(x), \quad G(x) = \text{Softmax}(\text{TopK}(x W_g, K=2))"
        elif paradigm == "ssm":
            math_formula = r"h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t, \quad \bar{A} = \exp(\Delta A)"
        elif paradigm == "mla":
            math_formula = r"c^{KV}_t = W^{DKV} x_t, \quad k^C_t = W^{UK} c^{KV}_t, \quad v^C_t = W^{UV} c^{KV}_t"
        elif paradigm == "pruning":
            math_formula = (
                r"W' = Q_2^\top W Q_1, \quad W_{sliced} = W'[:d'_1, :d'_2], \quad Q^\top Q = I"
            )
        elif paradigm == "quant":
            math_formula = (
                r"W = S \odot (Q_W - Z), \quad Q_W \in \{-8, \dots, 7\}, \quad S = \text{scale}"
            )
        else:
            math_formula = r"A_{ij} = \text{TopK}\left(\text{Softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right)\right)"

    # Clean description
    clean_desc = raw_doc.replace("\n", " ")[:350]
    if not clean_desc or len(clean_desc) < 20:
        clean_desc = (
            f"Production research module for {cat.lower()} across transformer architectures."
        )

    return {
        "id": base_name,
        "name": first_line[:65],
        "file": file_path,
        "type": "Attention Module" if is_attn else "Compression Engine",
        "category": cat,
        "paradigm": paradigm,
        "icon": icon,
        "description": clean_desc,
        "math": math_formula[:350],
        "markdown_preview": doc_markdown,
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)

    attn_files = sorted(glob.glob("neuralops/attentions/*.py"))
    comp_files = sorted(glob.glob("neuralops/compression_engine/*.py"))

    all_modules = []
    seen = set()

    for p in attn_files:
        if not p.endswith("__init__.py"):
            info = extract_module_info(p)
            all_modules.append(info)
            seen.add(info["id"])

    for p in comp_files:
        if not p.endswith("__init__.py"):
            info = extract_module_info(p)
            if info["id"] not in seen:
                all_modules.append(info)
                seen.add(info["id"])

    out_path = "Docs/visualizer/modules-catalog.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"total_count": len(all_modules), "generated_at": "2026-09-12", "modules": all_modules},
            f,
            indent=2,
        )

    print(f"[OK] Generated {out_path} with {len(all_modules)} modules across 7 research paradigms.")


if __name__ == "__main__":
    main()
