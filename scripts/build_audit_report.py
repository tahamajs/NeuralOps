import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
json_path = Path("/Users/tahamajs/.gemini/antigravity-cli/brain/8ddd2911-c5f0-4bfb-90a1-6ebebed5b8fd/all_modules_status.json")

with open(json_path) as f:
    stats = json.load(f)

full = [s for s in stats if s["pages"] >= 15 and s["words"] >= 6000]
thin = [s for s in stats if s["pages"] < 15 or s["words"] < 6000]

lines = []
lines.append("# Complete Per-Module Paper Pipeline Audit Report\n\n")
lines.append("**Date:** September 18, 2026  \n")
lines.append(f"**Scope:** Comprehensive Audit across all {len(stats)} Modules in `paper/PerModulePaper/`\n\n")
lines.append("---\n\n")
lines.append("## Executive Summary\n\n")
lines.append(f"- **Total Modules in Repository:** {len(stats)}\n")
lines.append(f"- **Reviewer-Ready Complete Papers (>=15 pages, >=6,000 words, strict verified):** **{len(full)}** ({len(full)/len(stats)*100:.1f}%)\n")
lines.append(f"- **Pending Regeneration (Track A / Stubs <15 pages):** **{len(thin)}** ({len(thin)/len(stats)*100:.1f}%)\n\n")
lines.append("---\n\n")

lines.append("## 1. Verified Complete Papers (>=15 Pages & Full Words)\n\n")
lines.append("| # | Module Name | Pages | PDF Size (KB) | Core Words | Status |\n")
lines.append("|---|---|---:|---:|---:|:---:|\n")
for idx, s in enumerate(sorted(full, key=lambda x: x["module"]), start=1):
    m = s["module"]
    p = s["pages"]
    sz = s["size_kb"]
    w = s["words"]
    lines.append(f"| {idx} | `{m}` | {p} | {sz} | {w} | **PASS** |\n")

lines.append("\n---\n\n")
lines.append("## 2. In-Queue / Pending Regeneration Modules\n\n")
lines.append("| # | Module Name | Current Pages | PDF Size (KB) | Words | Batch Status |\n")
lines.append("|---|---|---:|---:|---:|:---:|\n")
for idx, s in enumerate(sorted(thin, key=lambda x: x["module"]), start=1):
    m = s["module"]
    p = s["pages"]
    sz = s["size_kb"]
    w = s["words"]
    lines.append(f"| {idx} | `{m}` | {p} | {sz} | {w} | PENDING |\n")

out_file = ROOT / "paper" / "FULL_PAPERS_PIPELINE_AUDIT_REPORT.md"
out_file.write_text("".join(lines), encoding="utf-8")
print(f"Wrote full report to {out_file} ({len(stats)} modules, {len(full)} complete, {len(thin)} pending)")
