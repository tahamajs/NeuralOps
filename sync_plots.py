import os
import json
import subprocess
from pathlib import Path

KEY = os.path.expanduser("~/Desktop/host_access")
REMOTE = "Hadi Schumacher@192.168.1.100"

OUT_DIR = Path("Docs/results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = f"""ssh -i {KEY} -o StrictHostKeyChecking=no "{REMOTE}" "powershell -Command \\"cd 'C:\\Users\\Hadi Schumacher\\SVD_Project'; python -c \\\\\\"import json, os, glob; res={{}}; "
cmd += "for p in glob.glob('results/*/history/log_history.json'): "
cmd += "  d=os.path.basename(os.path.dirname(os.path.dirname(p))); "
cmd += "  try: res[d] = json.load(open(p, encoding='utf-8', errors='ignore')); "
cmd += "  except: pass; "
cmd += "print(json.dumps(res))\\\\\\"\\"""""

try:
    print(f"Running command: {cmd}")
    out = subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")
    start_idx = out.find("{")
    end_idx = out.rfind("}") + 1
    data = json.loads(out[start_idx:end_idx])
    
    for key, history in data.items():
        out_path = OUT_DIR / f"{key}_history.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
    print(f"Successfully saved {len(data)} plot histories to Docs/results/plots/")
except Exception as e:
    print("Failed to sync:", e)
