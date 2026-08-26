from pathlib import Path

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

old_candidates = '''    active_csv_candidates = [
        PROJECT_DIR / "results" / "run_quality_fixed_final" / "summary.csv",
        Path("results/run_quality_fixed_final/summary.csv"),
    ]'''

new_candidates = '''    active_csv_candidates = [
        PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv",
        Path("results/run_quality_pretrained/summary.csv"),
        PROJECT_DIR / "results" / "run_quality_fixed_final" / "summary.csv",
        Path("results/run_quality_fixed_final/summary.csv"),
    ]'''

content = content.replace(old_candidates, new_candidates, 1)

old_logs = '''    log_candidates = [
        PROJECT_DIR / "run_quality_sweep_final_run.log",
        Path("run_quality_sweep_final_run.log"),'''

new_logs = '''    log_candidates = [
        PROJECT_DIR / "run_quality_pretrained.log",
        Path("run_quality_pretrained.log"),
        PROJECT_DIR / "run_quality_sweep_final_run.log",
        Path("run_quality_sweep_final_run.log"),'''

content = content.replace(old_logs, new_logs, 1)

old_exclude = '        if run_id in ("run_quality_", "run_quality_fixed_1", "run_quality_20260826_175056"):\n            return\n        if run_id == "run_quality_fixed_final" and is_sweep_running():\n            return'
new_exclude = '        if run_id in ("run_quality_", "run_quality_fixed_1", "run_quality_20260826_175056"):\n            return\n        if run_id in ("run_quality_fixed_final", "run_quality_pretrained") and is_sweep_running():\n            return'
content = content.replace(old_exclude, new_exclude, 1)

p.write_text(content, encoding="utf-8")
print("Updated live_monitor.py for pretrained run.")
