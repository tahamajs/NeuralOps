import re
import json
import sys
from pathlib import Path

results_dir = Path("/mnt/d/SVD_Project/results/delta_sweep")
with open("/tmp/registry_snapshot.json") as f:
    registry = json.load(f)

attn_keys = set(registry["attn_keys"])
comp_keys = set(registry["comp_keys"])

delta_pre_re = re.compile(r"\[SANITY\]\s+Delta pre-baseline\s*=\s*([-+]?[0-9]+\.?[0-9]*)")
pre_ppl_re = re.compile(r"\[SANITY\]\s+Student pre-KD PPL\s*=\s*([0-9]+\.?[0-9]*)")
delta_any_re = re.compile(r"DELTA[^0-9+\-]*([-+]?[0-9]+\.?[0-9]*)")
oom_re = re.compile(r"OutOfMemoryError|CUDA out of memory", re.IGNORECASE)
nan_re = re.compile(r"NaN|Loss is NaN", re.IGNORECASE)
sigint_re = re.compile(r"KeyboardInterrupt")

modules = {}
for m in sorted(attn_keys):
    modules[("attention", m)] = {"type": "attention", "module": m}
for m in sorted(comp_keys):
    modules[("compressor", m)] = {"type": "compressor", "module": m}

for (m_type, m_name), info in modules.items():
    log_file = results_dir / f"{m_type}_{m_name}.log"
    err_file = results_dir / f"{m_type}_{m_name}.err"
    json_file = results_dir / f"{m_type}_{m_name}.json"
    info["has_log"] = log_file.exists()
    info["has_err"] = err_file.exists()
    info["has_json"] = json_file.exists()
    delta = None
    pre_ppl = None
    error_msg = None
    status = "PENDING"
    
    if log_file.exists():
        content = log_file.read_text(errors="ignore")
        m_pre = delta_pre_re.search(content)
        m_ppl = pre_ppl_re.search(content)
        m_delta_any = delta_any_re.search(content)
        if m_ppl:
            pre_ppl = float(m_ppl.group(1))
        if m_pre:
            delta = float(m_pre.group(1))
        elif m_delta_any:
            delta = float(m_delta_any.group(1))
        if sigint_re.search(content):
            error_msg = "SIGINT / Interrupted"
            status = "INTERRUPTED"
        elif oom_re.search(content):
            error_msg = "CUDA Out of Memory"
            status = "OOM"
        elif "Loss is NaN" in content or "Loss is nan" in content:
            error_msg = "NaN Loss"
            status = "NAN_LOSS"
            
    if err_file.exists() and not error_msg:
        err_content = err_file.read_text(errors="ignore").strip()
        if err_content:
            if oom_re.search(err_content):
                error_msg = "CUDA Out of Memory"
                status = "OOM"
            elif sigint_re.search(err_content):
                error_msg = "SIGINT / Interrupted"
                status = "INTERRUPTED"
            else:
                lines = [l for l in err_content.splitlines() if l.strip()]
                error_msg = lines[-1][:120] if lines else ""
                status = "ERROR"
                
    if delta is not None:
        info["delta"] = delta
        info["pre_ppl"] = pre_ppl
        if delta > 1.0:
            status = "PASS"
        else:
            status = "DELTA_ZERO"
    elif status == "PENDING" and info["has_log"]:
        status = "INCOMPLETE"
        
    info["status"] = status
    info["error"] = error_msg

counts = {}
for info in modules.values():
    s = info["status"]
    t = info["type"]
    counts[(t, s)] = counts.get((t, s), 0) + 1

print("=" * 60)
print("AUDIT STATUS COUNTS")
print("=" * 60)
for (t, s), count in sorted(counts.items()):
    print(f"  {t:12s} {s:15s}: {count}")

with open("/tmp/module_inventory.json", "w") as f:
    json.dump([v for v in modules.values()], f, indent=2)

print("\nInventory saved to /tmp/module_inventory.json")
