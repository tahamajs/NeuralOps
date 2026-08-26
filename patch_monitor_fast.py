from pathlib import Path

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

old_check = '''    if (PROJECT_DIR / "results" / "run_quality_pretrained").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv",
            Path("results/run_quality_pretrained/summary.csv"),
        ]'''

new_check = '''    if (PROJECT_DIR / "results" / "run_quality_fast").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv",
            Path("results/run_quality_fast/summary.csv"),
        ]
    elif (PROJECT_DIR / "results" / "run_quality_pretrained").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv",
            Path("results/run_quality_pretrained/summary.csv"),
        ]'''

content = content.replace(old_check, new_check, 1)

old_log_cands = '''    log_candidates = [
        PROJECT_DIR / "run_quality_pretrained.log",'''

new_log_cands = '''    log_candidates = [
        PROJECT_DIR / "run_quality_fast.log",
        Path("run_quality_fast.log"),
        PROJECT_DIR / "run_quality_pretrained.log",'''

content = content.replace(old_log_cands, new_log_cands, 1)

old_comb = 'tot_comb = 60 if is_pre else 240'
new_comb = '''is_fast = (PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv").exists()
    tot_comb = 6 if is_fast else (60 if is_pre else 240)'''
content = content.replace(old_comb, new_comb, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py patched for run_quality_fast.")
