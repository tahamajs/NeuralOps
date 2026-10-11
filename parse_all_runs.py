import csv
import json
import glob
import os
import re

logs = glob.glob("results/delta_sweep/*.log")

modules_data = []

for log_path in sorted(logs):
    base = os.path.basename(log_path)
    if not (base.startswith("compressor_") or base.startswith("attention_")):
        continue
    
    parts = base.split("_", 1)
    mod_type = parts[0]
    mod_name = parts[1].replace(".log", "")
    
    err_path = log_path.replace(".log", ".err")
    json_path = log_path.replace(".log", ".json")
    
    # Read log content
    content = ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        pass
        
    # Read err content
    err_content = ""
    if os.path.exists(err_path):
        try:
            with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                err_content = f.read()
        except Exception:
            pass

    # Extract Pre-KD PPL and Delta
    pre_ppl = None
    delta = None
    
    # Pattern 1: Delta pre-baseline = +8848.83 or -0.01
    m_delta = re.search(r"Delta pre-baseline\s*=\s*([+-]?\d+\.?\d*)", content)
    if m_delta:
        try:
            delta = float(m_delta.group(1))
        except:
            pass
            
    # Pattern 2: Pre-KD PPL for Module ..., Method C: 36.22
    m_pre = re.search(r"Pre-KD PPL for Module.*?:\s*(\d+\.?\d*)", content)
    if m_pre:
        try:
            pre_ppl = float(m_pre.group(1))
        except:
            pass

    # Pattern 3: check JSON if available
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                jd = json.load(jf)
                if delta is None and jd.get("delta") is not None:
                    delta = float(jd["delta"])
                if pre_ppl is None and jd.get("student_ppl") is not None:
                    pre_ppl = float(jd["student_ppl"])
        except Exception:
            pass

    # Check for TIMEOUT or ERROR
    has_timeout = "TIMEOUT - killing PID" in content or "exceeded 300s" in content
    has_crash = "Traceback (most recent call last):" in content or "Traceback (most recent call last):" in err_content
    
    status = "UNKNOWN"
    if delta is not None:
        if delta > 1.0:
            status = "PASS"
        else:
            status = "FAIL_DELTA"
    elif has_timeout:
        status = "TIMEOUT"
    elif has_crash:
        status = "ERROR"
    else:
        status = "UNKNOWN"

    modules_data.append({
        "module": mod_name,
        "type": mod_type,
        "status": status,
        "delta": delta,
        "pre_ppl": pre_ppl,
        "has_timeout": has_timeout,
        "has_crash": has_crash
    })

p = [m for m in modules_data if m["status"] == "PASS"]
fd = [m for m in modules_data if m["status"] == "FAIL_DELTA"]
er = [m for m in modules_data if m["status"] == "ERROR"]
to = [m for m in modules_data if m["status"] == "TIMEOUT"]
un = [m for m in modules_data if m["status"] == "UNKNOWN"]

print(f"PASS: {len(p)}")
print(f"FAIL_DELTA: {len(fd)}")
print(f"ERROR: {len(er)}")
print(f"TIMEOUT: {len(to)}")
print(f"UNKNOWN: {len(un)}")

print("\n--- FAIL_DELTA modules ---")
for m in fd:
    print(f"  {m['type']}:{m['module']} (delta={m['delta']}, pre_ppl={m['pre_ppl']})")

print("\n--- ERROR modules ---")
for m in er:
    print(f"  {m['type']}:{m['module']}")

print("\n--- TIMEOUT modules ---")
for m in to:
    print(f"  {m['type']}:{m['module']}")

print("\n--- UNKNOWN modules ---")
for m in un:
    print(f"  {m['type']}:{m['module']}")

