import re
from pathlib import Path

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

if "def sanitize_json" not in content:
    content = content.replace("import json", "import json\nimport math")
    sanitize_func = '''
def sanitize_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj
'''
    content = content.replace("class RunManager:", sanitize_func + "\nclass RunManager:")

# Replace json.dumps(res) -> json.dumps(sanitize_json(res))
content = content.replace("json.dumps(response)", "json.dumps(sanitize_json(response))")
content = content.replace("json.dumps({\"runs\": runs_list})", "json.dumps(sanitize_json({\"runs\": runs_list}))")
content = content.replace("json.dumps(run_data)", "json.dumps(sanitize_json(run_data))")
content = content.replace("json.dumps(run_data, indent=2)", "json.dumps(sanitize_json(run_data), indent=2)")
content = content.replace("json.dumps({\"latex\": latex_code})", "json.dumps(sanitize_json({\"latex\": latex_code}))")
content = content.replace("json.dumps(generate_plot_data(results), indent=2)", "json.dumps(sanitize_json(generate_plot_data(results)), indent=2)")
content = content.replace("json.dumps(res)", "json.dumps(sanitize_json(res))")
content = content.replace("json.dumps({\"success\": success, \"message\": msg})", "json.dumps(sanitize_json({\"success\": success, \"message\": msg}))")

# Track run_quality_ep1
old_csv_cand = '''    if (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists():'''
new_csv_cand = '''    if (PROJECT_DIR / "results" / "run_quality_ep1").exists() or (PROJECT_DIR / "run_quality_ep1.log").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_ep1" / "summary.csv",
            Path("results/run_quality_ep1/summary.csv"),
        ]
    elif (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists():'''
content = content.replace(old_csv_cand, new_csv_cand, 1)

old_log_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_screening_fast.log",'''
new_log_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_ep1.log",
        Path("run_quality_ep1.log"),
        PROJECT_DIR / "run_quality_screening_fast.log",'''
content = content.replace(old_log_cand, new_log_cand, 1)

old_parse_is_screening = '''    is_screening_fast = (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()
    is_screening = is_screening_fast or (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()'''
new_parse_is_screening = '''    is_screening_ep1 = (PROJECT_DIR / "results" / "run_quality_ep1").exists() or (PROJECT_DIR / "run_quality_ep1.log").exists()
    is_screening_fast = is_screening_ep1 or (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()
    is_screening = is_screening_fast or (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()'''
content = content.replace(old_parse_is_screening, new_parse_is_screening, 1)

old_log_parse_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_screening_fast.log",'''
new_log_parse_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_ep1.log",
        Path("run_quality_ep1.log"),
        PROJECT_DIR / "run_quality_screening_fast.log",'''
content = content.replace(old_log_parse_cand, new_log_parse_cand, 1)

old_sum_parse_cand = '''    summary_candidates = [
        PROJECT_DIR / "results" / "run_quality_screening_fast" / "summary.csv",'''
new_sum_parse_cand = '''    summary_candidates = [
        PROJECT_DIR / "results" / "run_quality_ep1" / "summary.csv",
        Path("results/run_quality_ep1/summary.csv"),
        PROJECT_DIR / "results" / "run_quality_screening_fast" / "summary.csv",'''
content = content.replace(old_sum_parse_cand, new_sum_parse_cand, 1)

# Modify epochs for ep1
old_setup = '''    modules_progress = []
    total_seeds = 1 if is_screening else 3
    if 'parsed_seeds' in locals() and parsed_seeds:
        total_seeds = parsed_seeds
    seeds_order = [42] if total_seeds == 1 else [42, 123, 777]
    total_epochs = 2 if is_screening else 3
    max_steps_per_epoch = 526 if is_screening else 1052
    pace_per_seed = 75.0 if is_screening else 160.0'''
new_setup = '''    modules_progress = []
    total_seeds = 1 if is_screening else 3
    if 'parsed_seeds' in locals() and parsed_seeds:
        total_seeds = parsed_seeds
    seeds_order = [42] if total_seeds == 1 else [42, 123, 777]
    total_epochs = 1 if is_screening_ep1 else (2 if is_screening else 3)
    max_steps_per_epoch = 526 if is_screening else 1052
    pace_per_seed = 55.0 if is_screening_ep1 else (75.0 if is_screening else 160.0)'''
content = content.replace(old_setup, new_setup, 1)

old_det_cands = '''    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_screening_fast" / "detailed_results.json",'''
new_det_cands = '''    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_ep1" / "detailed_results.json",
        Path("results/run_quality_ep1/detailed_results.json"),
        PROJECT_DIR / "results" / "run_quality_screening_fast" / "detailed_results.json",'''
content = content.replace(old_det_cands, new_det_cands, 1)

old_list_names = '''            screen_fast_exists = (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()
            screen_exists = screen_fast_exists or (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()'''
new_list_names = '''            screen_ep1_exists = (PROJECT_DIR / "results" / "run_quality_ep1").exists() or (PROJECT_DIR / "run_quality_ep1.log").exists()
            screen_fast_exists = screen_ep1_exists or (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()
            screen_exists = screen_fast_exists or (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()'''
content = content.replace(old_list_names, new_list_names, 1)

old_run_name = '''                r_name = "Phase 2 Quality Screening (1 Seed, 20 Modules)" if screen_fast_exists else ("Phase 2 Quality Screening (20 Modules)" if screen_exists else "Phase 2 Quality Sweep (Fast)")'''
new_run_name = '''                r_name = "Phase 2 Quality Screening (1-Seed 1-Epoch)" if screen_ep1_exists else ("Phase 2 Quality Screening (1 Seed, 20 Modules)" if screen_fast_exists else ("Phase 2 Quality Screening (20 Modules)" if screen_exists else "Phase 2 Quality Sweep (Fast)"))'''
content = content.replace(old_run_name, new_run_name, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py successfully patched for ep1 and NaNs!")
