from pathlib import Path

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

old_block = '''    active_csv_candidates = [
        PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv",
        Path("results/run_quality_pretrained/summary.csv"),
        PROJECT_DIR / "results" / "run_quality_fixed_final" / "summary.csv",
        Path("results/run_quality_fixed_final/summary.csv"),
    ]'''

new_block = '''    if (PROJECT_DIR / "results" / "run_quality_pretrained").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv",
            Path("results/run_quality_pretrained/summary.csv"),
        ]
    else:
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_fixed_final" / "summary.csv",
            Path("results/run_quality_fixed_final/summary.csv"),
        ]'''

content = content.replace(old_block, new_block, 1)
p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py patched so it never falls back to old run!")
