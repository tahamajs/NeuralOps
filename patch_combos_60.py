from pathlib import Path

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

# In _build_stats_response:
old_prog = '    progress = compute_eta_and_progress(results, total_combos=240, window_size=50)'
new_prog = '''    is_pre = (PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv").exists()
    tot_comb = 60 if is_pre else 240
    progress = compute_eta_and_progress(results, total_combos=tot_comb, window_size=50)'''
content = content.replace(old_prog, new_prog, 1)

# In list_runs:
old_num = '"num_combinations": 240,'
new_num = '"num_combinations": 60 if (PROJECT_DIR / "results" / "run_quality_pretrained" / "summary.csv").exists() else 240,'
content = content.replace(old_num, new_num, 1)

p.write_text(content, encoding="utf-8")
print("Combo count dynamically configured.")
