import json
from pathlib import Path

seen = {}
try:
    root = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
    log_file = root / "audit" / "module_test_log.jsonl"
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                gpt2 = rec.get('models', {}).get('gpt2', {})
                d = gpt2.get('diff', 0.0)
                s = gpt2.get('status', '')
                err = gpt2.get('error', '')
                if s == 'FAIL' or (isinstance(d, (int, float)) and d > 1.0):
                    seen[rec['module']] = (d, str(err)[:50])
            except Exception:
                pass
except Exception as ex:
    print(f"Error reading file: {ex}")

for m in sorted(seen.keys(), key=lambda x: seen[x][0]):
    print(f"{m:35s} diff={seen[m][0]:.3f}  err={seen[m][1]}")
