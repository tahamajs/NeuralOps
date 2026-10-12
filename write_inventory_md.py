import json
from pathlib import Path

with open("/tmp/module_inventory.json") as f:
    items = json.load(f)

attn_items = [it for it in items if it["type"] == "attention"]
comp_items = [it for it in items if it["type"] == "compressor"]

md = ["# Ground-Truth Failure and Delta Inventory\n"]
md.append("Independent audit based directly on raw disk logs and JSON results.\n")
md.append("## Registry Breakdown\n")
md.append(f"- **Total Attention Modules in Registry**: {len(attn_items)}")
md.append(f"- **Total Compressor Modules in Registry**: {len(comp_items)}\n")

md.append("## Attention Modules Requiring Action / Non-Pass\n")
md.append("| Module | Status | Delta | Student PPL | Root Cause / Error |")
md.append("|---|---|---|---|---|")
for it in attn_items:
    if it["status"] not in ("PASS", "PENDING"):
        mod = it["module"]
        st = it["status"]
        d = str(it.get("delta"))
        ppl = str(it.get("pre_ppl"))
        err = str(it.get("error") or "")
        md.append(f"| `{mod}` | **{st}** | {d} | {ppl} | {err} |")

md.append("\n## Compressor Modules Requiring Action / Non-Pass\n")
md.append("| Module | Status | Delta | Student PPL | Root Cause / Error |")
md.append("|---|---|---|---|---|")
for it in comp_items:
    if it["status"] != "PASS":
        mod = it["module"]
        st = it["status"]
        d = str(it.get("delta"))
        ppl = str(it.get("pre_ppl"))
        err = str(it.get("error") or "")
        md.append(f"| `{mod}` | **{st}** | {d} | {ppl} | {err} |")

out_path = Path("/mnt/d/SVD_Project/docs/DELTA_FAILURE_INVENTORY.md")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(md) + "\n")
print("Inventory written to", out_path)
