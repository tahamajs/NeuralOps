import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

skip_logic = """
    completed_runs = []
    if MASTER_LOG.exists():
        with open(MASTER_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "<< SUCCESS:" in line or "⚠️ NOTICE:" in line:
                    if "attn=" in line and "task=" in line:
                        parts = line.split(" | ")
                        a = [p for p in parts if "attn=" in p]
                        t = [p for p in parts if "task=" in p]
                        if a and t:
                            attn_val = a[0].split("=")[1].strip()
                            task_val = t[0].split("=")[1].strip()
                            completed_runs.append((attn_val, task_val))

    log_master(f"Trajectory Matrix: {len(ATTN_MODELS)} Models x {len(TASKS)} Tasks = {total} Runs")
"""

content = content.replace(
    'log_master(f"Trajectory Matrix: {len(ATTN_MODELS)} Models x {len(TASKS)} Tasks = {total} Runs")',
    skip_logic
)

loop_logic = """
        for attn in ATTN_MODELS:
            idx += 1
            if (attn, task) in completed_runs:
                log_master(f"[{idx}/{total}] -- SKIPPING: attn={attn} | task={task} (Already in master log)")
                continue

            log_master(f"[{idx}/{total}] >> Launching: attn={attn} | task={task} | epochs=0.5 | seed=42")
"""

content = re.sub(
    r"\s*for attn in ATTN_MODELS:\s*idx \+= 1\s*log_master[^>]*>> Launching[^)]*\)",
    loop_logic,
    content
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
