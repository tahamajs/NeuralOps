import re
from pathlib import Path

file_path = Path('Codes/framework/live_monitor.py')
content = file_path.read_text()

start_str = "    log_candidates = ["
end_str = "    ]"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

replacement = """    log_candidates = [
        PROJECT_DIR / "run_quality_sweep_final_run.log",
        Path("run_quality_sweep_final_run.log"),
        PROJECT_DIR / "run_quality_sweep_final.log",
        PROJECT_DIR / "adaptive_sweep_v3.log",
        Path("adaptive_sweep_v3.log"),
        PROJECT_DIR / "full_sweep_v2.log",
        Path("full_sweep_v2.log"),
        PROJECT_DIR / "full_sweep.log",
        Path("full_sweep.log"),
        PROJECT_DIR / "comprehensive_run_detached.log",
        Path("comprehensive_run_detached.log")
    ]"""

new_content = content[:start_idx] + replacement + content[end_idx:]
file_path.write_text(new_content)
print("Updated successfully")
