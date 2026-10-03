import json
from pathlib import Path

root = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
log_path = root / "audit" / "module_test_log.jsonl"

if log_path.exists():
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                rec = json.loads(line)
                gpt2 = rec.get('models', {}).get('gpt2', {})
                d = gpt2.get('diff', 999.0)
                if d < 0.1 and gpt2.get('status') == 'OK':
                    print(f"{rec['module']:30s} gpt2_diff={d:.6f}")
            except Exception:
                pass
