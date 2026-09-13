import re
from pathlib import Path

REPO_DIR = Path("D:/SVD_Project")
LOG_FILE = REPO_DIR / "finish_benchmark.log"

log_text = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
ran_matches = re.findall(r'\[(\d+)/80\] Running:\s*([a-zA-Z0-9_\-]+)', log_text)

ran_order = []
seen = set()
for num, mod in ran_matches:
    num = int(num)
    if num not in seen:
        seen.add(num)
        ran_order.append((num, mod))

print(f"Modules executed or in progress in current orchestrator: {len(ran_order)} / 80")
if ran_order:
    last_num, last_mod = ran_order[-1]
    print(f"Current module: [{last_num}/80] {last_mod}")

# Check runner files
all_runners = sorted([p.stem.replace("run_all_4_", "") for p in REPO_DIR.glob("run_all_4_*.py")])
print(f"Total run_all_4_*.py files in repository: {len(all_runners)}")

# Also check finish_benchmark.py queue definition
import importlib.util
spec = importlib.util.spec_from_file_location("fb", str(REPO_DIR / "finish_benchmark.py"))
fb = importlib.util.module_from_spec(spec)
try:
    # discover remaining
    pending_runners = [fb.extract_module_name(r) for r in fb.discover_remaining()]
    print(f"Currently not passing >= 56%: {len(pending_runners)}")
except Exception as e:
    print(f"Error inspecting discover_remaining: {e}")
