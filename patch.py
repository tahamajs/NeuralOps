from pathlib import Path

file_path = Path('Codes/framework/live_monitor.py')
content = file_path.read_text()

old_list = """    log_candidates = [
        PROJECT_DIR / "adaptive_sweep_v3.log","""

new_list = """    log_candidates = [
        PROJECT_DIR / "run_quality_sweep_final_run.log",
        Path("run_quality_sweep_final_run.log"),
        PROJECT_DIR / "adaptive_sweep_v3.log","""

new_content = content.replace(old_list, new_list, 1)
file_path.write_text(new_content)
print("patched")
