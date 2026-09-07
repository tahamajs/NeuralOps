#!/usr/bin/env python3
"""Extract per-module code facts (digests) for image-prompt generation.

For every canonical module in scratch/image_prompts/manifest.json, parse the
source with ast + regex and emit a compact digest containing: classes/bases,
docstrings, formula-like comment lines, tensor ops with shape literals,
constructor params, nn.Parameter shapes, and feature keyword flags.
Output: scratch/image_prompts/digests/dNN_<name>.md  (≤ ~70 lines each)
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path("/Users/tahamajs/Documents/uni/Project")
OUT = ROOT / "scratch/image_prompts/digests"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((ROOT / "scratch/image_prompts/manifest.json").read_text())
modules = manifest["attentions"] + manifest["compressors"]

FORMULA_RE = re.compile(r"[A-Za-z_][\w\s*+\-/()]*=\s*[^=\n]*(exp|softmax|sigmoid|sqrt|log|Σ|∑|λ|alpha|beta|tau|W\s|U\s|V\s|A\s|B\s|Σ)", re.I)
SHAPE_RE = re.compile(r"[\(\[]((?:[BbNnHhDdKkQqVvLlSsEeTtWwGg\d\*, ]){3,})[\)\]]")
OPS_RE = re.compile(r"\.(view|reshape|transpose|permute|matmul|einsum|contiguous|softmax|masked_fill|bmm|chunk|split|expand)\(")
FEATURES = {
    "lora": r"lora", "rope": r"rope|rotary", "gqa": r"\bgqa|kv_heads|n_kv|num_kv", "rope_partial": r"partial.*rope|rope.*partial",
    "window": r"window|sliding", "decay": r"decay", "gating": r"gate", "moe": r"\bmoe|expert", "topk": r"top_k|topk",
    "quant": r"quant|int4|int8|nf4", "causal": r"causal", "alibi": r"alibi", "svd": r"\bsvd", "rank": r"\brank",
    "lowrank": r"low_?rank", "cluster": r"cluster|kmeans", "pool": r"pool|average_pool", "stride": r"stride",
    "chunked": r"chunk", "conv1d": r"Conv1D", "peft": r"base_layer|peft", "bnb": r"nf4|bitsandbytes",
    "delta": r"delta_rule|deltanet", "ssm": r"\bssm|state_space|selective", "tt": r"tensor_train|tt_|ttlinear",
    "kronecker": r"kronecker|\btt_", "hash": r"hash", "fisher": r"fisher", "entropy": r"entropy", "hadamard": r"hadamard",
    "rotation": r"rotat", "sparsity": r"spars", "distill": r"distil", "calib": r"calibrat", "adaptive": r"adaptive",
}


def doc_head(node, n=6):
    doc = ast.get_docstring(node) or ""
    lines = [l.strip() for l in doc.splitlines() if l.strip()]
    return lines[:n]


def clean_formula_lines(src: str, limit=10):
    out = []
    for raw in src.splitlines():
        line = raw.strip()
        if line.startswith("#") and len(line) > 8:
            body = line.lstrip("#").strip()
            if re.search(r"[=≈≤≥→]|\\frac|\\sum|\\mathcal|O\(N", body) and not body.startswith(("http", "NOTE", "TODO")):
                out.append(body)
        if len(out) >= limit:
            break
    return out


def digest(mod: dict, idx: int) -> str:
    src_path = ROOT / mod["file"]
    src = src_path.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None

    lines = [f"# d{idx:03d} {mod['name']}  [{mod['registry']}]",
             f"aliases: {','.join(mod['aliases'])}",
             f"file: {mod['file']}  doc: {mod['doc'] or 'NONE'}"]

    if tree:
        mod_docs = doc_head(tree, 4)
        if mod_docs:
            lines.append("doc: " + " | ".join(mod_docs))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = ",".join(getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases)
                lines.append(f"class {node.name}({bases})")
                cd = doc_head(node, 2)
                if cd:
                    lines.append("  doc: " + " | ".join(cd))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                args = []
                for a in node.args.args[1:]:  # skip self
                    default = ""
                    lines_src = src.splitlines()
                    args.append(a.arg)
                lines.append("__init__(" + ", ".join(args[:12]) + ")")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
                lines.append("forward(" + ", ".join(a.arg for a in node.args.args[1:6]) + ")")

    # tensor ops + shapes (context window around ops)
    ops_found = []
    for i, line in enumerate(src.splitlines()):
        if OPS_RE.search(line):
            s = line.strip()[:150]
            ops_found.append(s)
    if ops_found:
        lines.append("tensor_ops:")
        lines += [f"  {o}" for o in ops_found[:8]]

    shapes = set()
    for m in SHAPE_RE.finditer(src):
        inner = m.group(1)
        if re.search(r"[BbNnHhDd]", inner) and "," in inner:
            shapes.add(f"[{inner.strip()}")
    if shapes:
        lines.append("shapes: " + " ".join(sorted(shapes)[:8]))

    params = re.findall(r"nn\.Parameter\(([^)]{5,90})\)", src)
    if params:
        lines.append("parameters: " + " | ".join(p.strip().replace('\n', ' ')[:60] for p in params[:5]))

    fml = clean_formula_lines(src)
    if fml:
        lines.append("formulas/comments:")
        lines += [f"  {f[:140]}" for f in fml]

    low = src.lower()
    feats = [k for k, pat in FEATURES.items() if re.search(pat, low)]
    if feats:
        lines.append("features: " + ",".join(sorted(feats)[:14]))

    lines.append("loc: " + str(len(src.splitlines())))
    return "\n".join(lines)


for i, mod in enumerate(modules, 1):
    text = digest(mod, i)
    (OUT / f"d{i:03d}_{mod['name']}.md").write_text(text)

print(f"wrote {len(modules)} digests to {OUT}")
