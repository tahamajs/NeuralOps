import csv
import json
import os

with open("results/delta_sweep/summary.csv", "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

pass_list = []
fail_delta_list = []
err_list = []
timeout_list = []
unk_list = []

for row in rows:
    mod = row["module"]
    t = row["type"]
    st = row["status"]
    d_str = row.get("delta", "")
    delta = None
    if d_str and d_str.strip():
        try:
            delta = float(d_str)
        except Exception:
            pass

    # If delta is empty in summary.csv, check the module json
    if delta is None:
        j_path = f"results/delta_sweep/{t}_{mod}.json"
        if os.path.exists(j_path):
            try:
                with open(j_path) as jf:
                    jd = json.load(jf)
                    delta = jd.get("delta")
            except Exception:
                pass

    if delta is not None:
        if delta > 1.0:
            pass_list.append((mod, delta))
        else:
            fail_delta_list.append((mod, delta))
    elif st == "TIMEOUT":
        timeout_list.append(mod)
    elif st == "ERROR":
        err_list.append(mod)
    else:
        unk_list.append(mod)

print(f"PASS: {len(pass_list)}")
print(f"FAIL_DELTA: {len(fail_delta_list)}")
print(f"ERROR: {len(err_list)}")
print(f"TIMEOUT: {len(timeout_list)}")
print(f"UNKNOWN: {len(unk_list)}")

print("\nFAIL_DELTA modules:")
for m, d in fail_delta_list:
    print(f"  {m} (delta={d})")

print("\nERROR modules:")
for m in err_list:
    print(f"  {m}")

print("\nTIMEOUT modules:")
for m in timeout_list:
    print(f"  {m}")

if unk_list:
    print("\nUNKNOWN modules:")
    for m in unk_list:
        print(f"  {m}")
