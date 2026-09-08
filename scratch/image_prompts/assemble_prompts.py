#!/usr/bin/env python3
"""Assemble module_image_prompts.md + module_image_prompts.json and verify coverage.

Concatenates scratch/image_prompts/sections/batch_*.md in canonical order,
verifies every one of the 220 canonical modules from the manifest appears
exactly once as a `## Module:` header, and emits the JSON twin.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/Users/tahamajs/Documents/uni/Project")
PROMPTS = ROOT / "scratch/image_prompts"
manifest = json.loads((PROMPTS / "manifest.json").read_text())
order = [m["name"] for m in manifest["attentions"] + manifest["compressors"]]

# gather section files in batch order
batches = sorted((PROMPTS / "sections").glob("batch_*.md"),
                 key=lambda p: int(re.search(r"batch_(\d+)", p.name).group(1)))
full = []
for b in batches:
    full.append(b.read_text())
body = "\n\n".join(full).strip() + "\n"

# find all module headers
found = re.findall(r"^## Module: ([\w\-\.]+)", body, flags=re.M)

missing = [n for n in order if n not in found]
extra = [n for n in found if n not in set(order)]
dupes = sorted({n for n in found if found.count(n) > 1})

header = (
    "# NeuralOps — Architecture Image Generation Prompts\n\n"
    "Paper-grade image-generation prompts for all **220 canonical registered modules** "
    f"({len(manifest['attentions'])} attentions + {len(manifest['compressors'])} compression engines) "
    "of the NeuralOps framework (`Codes/framework/attentions/`, `Codes/framework/compression_engine/`).\n\n"
    "Every prompt set was generated directly from the module's source code — class names, tensor shapes, "
    "formulas, and mechanism descriptions reflect the real implementation, not generic placeholders. "
    "Canonical module names/aliases follow `scratch/image_prompts/manifest.json` "
    "(alias-deduplicated registry resolution; counts reproducible via `scripts/count_modules.py`).\n\n"
    "Each module carries at least five self-contained prompts:\n"
    "1. **Base Architecture Diagram** — full schematic with data flow and shape annotations\n"
    "2. **Mathematical Core** — key formulas as large LaTeX with per-term visual annotation\n"
    "3. **Tensor Flow & Shape Trace** — step-by-step shape-annotated pipeline\n"
    "4. **Unique Mechanism Close-up** — zoom on the module's distinguishing idea\n"
    "5. **System Context & Complexity** — placement in a transformer layer + complexity inset\n\n"
    "Flagship modules (ASVD, Linformer, Mamba, MinimalSVD, Nyström, Performer, SVD-Linear, AASVD, "
    "FLAP, Hybrid-SVD, SliceGPT, SparseGPT, Subspace Projection, Swift-SVD) additionally include "
    "**Paper Comparison Figure** and **Attention/KV- or Spectrum-Visualization** prompts.\n\n"
    "All prompts share one style block (flat vector, #60A5FA/#A78BFA palette, engineering grid, 16:9) "
    "so figures generated across the set look consistent in the paper.\n\n"
    "---\n\n"
)

out_md = ROOT / "module_image_prompts.md"
out_md.write_text(header + body)

# JSON twin
sections = re.split(r"^## Module: ", body, flags=re.M)[1:]
json_out = {}
for sec in sections:
    name = sec.split("\n", 1)[0].strip()
    rest = sec.split("\n", 1)[1] if "\n" in sec else ""
    prompts = re.findall(r"\*\*(Image Prompt \d+ — [^*]+):\*\*\s*\n(.*?)(?=\n\n\*\*Image Prompt |\n\n## Module:|\Z)",
                         rest, flags=re.S)
    json_out[name] = {
        "image_prompt": prompts[0][1].strip() if prompts else "",
        "all_prompts": {title.strip(): text.strip() for title, text in prompts},
    }

out_json = ROOT / "module_image_prompts.json"
out_json.write_text(json.dumps(json_out, indent=1))

print(f"modules expected: {len(order)}")
print(f"module headers found: {len(found)} (unique: {len(set(found))})")
print(f"missing: {missing}")
print(f"extra: {extra}")
print(f"dupes: {dupes}")
print(f"json entries: {len(json_out)}")
print(f"md size: {out_md.stat().st_size/1024:.0f} KB, json size: {out_json.stat().st_size/1024:.0f} KB")
ok = not missing and not extra and not dupes and len(json_out) == len(order)
print("COVERAGE:", "PASS" if ok else "FAIL")
