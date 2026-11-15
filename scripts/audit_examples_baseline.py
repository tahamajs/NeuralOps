#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# Baseline static and dynamic audit scanner for NeuralOps examples suite (Wave 6).

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES_DIR = Path("examples")
OUTPUT_JSON = Path("audits/wave6/baseline_raw.json")
OUTPUT_MD = Path("audits/wave6/baseline_audit.md")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

print("Scanning examples 01 through 20...")
results = {}

for i in range(1, 21):
    ex_files = list(EXAMPLES_DIR.glob(f"{i:02d}_*.py"))
    if not ex_files:
        continue
    target = ex_files[0]
    code = target.read_text(encoding="utf-8")
    
    # 1. AST Analysis
    tree = ast.parse(code)
    randn_calls = []
    pass_fail_true = []
    silent_skips = []
    reimplemented_pillars = []
    fragile_paths = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                func_name = ast.unparse(node.func)
                if "torch.randn" in func_name:
                    randn_calls.append(node.lineno)
                if "pass_fail" in func_name:
                    args = [ast.unparse(a) for a in node.args]
                    if len(args) >= 2 and args[1] == "True":
                        pass_fail_true.append(node.lineno)
            except Exception:
                pass
        if isinstance(node, ast.ExceptHandler):
            if node.type:
                try:
                    type_str = ast.unparse(node.type)
                    if "ImportError" in type_str:
                        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            silent_skips.append(node.lineno)
                except Exception:
                    pass
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("pillar") or node.name.startswith("loss_pillar"):
                reimplemented_pillars.append((node.lineno, node.name))
    
    for idx, line in enumerate(code.splitlines(), 1):
        if re.search(r"\.transformer\.layer\[|\._wrapped_attention", line):
            fragile_paths.append(idx)
            
    # 2. Subprocess execution timing
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(target)],
            capture_output=True,
            text=True,
            timeout=45
        )
        ret_code = proc.returncode
        err_msg = proc.stderr[-200:].strip() if ret_code != 0 else ""
    except subprocess.TimeoutExpired:
        ret_code = 124
        err_msg = "TIMEOUT (>45s)"
    except Exception as exc:
        ret_code = 1
        err_msg = str(exc)
        
    elapsed = round(time.time() - t0, 2)
    
    results[target.name] = {
        "example_num": i,
        "exit_code": ret_code,
        "elapsed_seconds": elapsed,
        "error_snippet": err_msg,
        "randn_lines": randn_calls,
        "pass_fail_true_lines": pass_fail_true,
        "silent_skip_lines": silent_skips,
        "inline_pillars": reimplemented_pillars,
        "fragile_paths": fragile_paths,
    }
    print(f"  [{i:02d}] {target.name}: exit={ret_code} ({elapsed}s)")

OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"Wrote machine-readable baseline to {OUTPUT_JSON}")

# Render Markdown report
md_lines = [
    "# NeuralOps Wave 6 — Baseline Audit Report",
    "",
    f"**Scan Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
    f"**Total Examples Scanned:** {len(results)}",
    "",
    "## 1. Summary Status Table",
    "",
    "| # | File | Exit Code | Runtime (s) | `torch.randn` Lines | Hardcoded `pass_fail True` | Silent Skips | Status |",
    "|---|---|:---:|:---:|---|---|---|:---:|",
]

for name, r in sorted(results.items(), key=lambda x: x[1]["example_num"]):
    num = f"{r['example_num']:02d}"
    exit_c = r["exit_code"]
    sec = r["elapsed_seconds"]
    randn_str = ", ".join(str(l) for l in r["randn_lines"]) if r["randn_lines"] else "None"
    pf_str = ", ".join(str(l) for l in r["pass_fail_true_lines"]) if r["pass_fail_true_lines"] else "None"
    skip_str = ", ".join(str(l) for l in r["silent_skip_lines"]) if r["silent_skip_lines"] else "None"
    status_label = "CLEAN" if exit_c == 0 and not r["pass_fail_true_lines"] and not r["inline_pillars"] else "FLAGGED"
    md_lines.append(f"| {num} | `{name}` | {exit_c} | {sec}s | {randn_str} | {pf_str} | {skip_str} | **{status_label}** |")

md_lines.append("")
md_lines.append("## 2. Issues Requiring Remediation")
md_lines.append("- Examples with hardcoded pass_fail True must be refactored to compute boolean status.")
md_lines.append("- Examples with inline reimplemented pillars must use library APIs or raise NotImplementedError.")
md_lines.append("- Duplicate and timing-out examples (such as Example 13) must be deleted.")

OUTPUT_MD.write_text("\n".join(md_lines) + "\n")
print(f"Wrote Markdown baseline report to {OUTPUT_MD}")
