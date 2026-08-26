from pathlib import Path

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

# 1. Update active_csv_candidates in get_completed_results
old_csv_cand = '''    if (PROJECT_DIR / "results" / "run_quality_fast").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv",
            Path("results/run_quality_fast/summary.csv"),
        ]'''

new_csv_cand = '''    if (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_screening" / "summary.csv",
            Path("results/run_quality_screening/summary.csv"),
        ]
    elif (PROJECT_DIR / "results" / "run_quality_fast").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv",
            Path("results/run_quality_fast/summary.csv"),
        ]'''

content = content.replace(old_csv_cand, new_csv_cand, 1)

# 2. Update log_candidates in get_completed_results
old_log_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_fast.log",'''

new_log_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_screening.log",
        Path("run_quality_screening.log"),
        PROJECT_DIR / "run_quality_fast.log",'''

content = content.replace(old_log_cand, new_log_cand, 1)

# 3. Update FAST_MODULE_LIST and parse_fast_sweep_live_progress to support screening
old_fast_list = '''FAST_MODULE_LIST = [
    "full",
    "svd_linear_attention",
    "minimal_svd",
    "mamba",
    "performer_attention",
    "linformer_attention",
]'''

new_fast_list = '''SCREENING_MODULE_LIST = [
    "full", "svd_linear_attention", "minimal_svd", "aegis_v6", "leka_v3",
    "nasc_attention", "svd_mamba", "mamba", "samba_m3", "gated_deltanet_block",
    "griffin_ssd", "jamba2", "spectron", "block_transformer_hiba",
    "performer_attention", "cosformer", "linformer_attention",
    "galerkin_attention", "adaptive_span", "soa_attn",
]

FAST_MODULE_LIST = [
    "full",
    "svd_linear_attention",
    "minimal_svd",
    "mamba",
    "performer_attention",
    "linformer_attention",
]'''

content = content.replace(old_fast_list, new_fast_list, 1)

# Update parser logic in parse_fast_sweep_live_progress to dynamically detect screening vs fast
old_parser_start = '''    log_candidates = [
        PROJECT_DIR / "run_quality_fast.log",
        Path("run_quality_fast.log"),
    ]
    log_file = None
    for lc in log_candidates:
        if lc.exists() and lc.stat().st_size > 0:
            log_file = lc
            break

    summary_candidates = [
        PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv",
        Path("results/run_quality_fast/summary.csv"),
    ]'''

new_parser_start = '''    is_screening = (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()
    active_module_list = SCREENING_MODULE_LIST if is_screening else FAST_MODULE_LIST

    log_candidates = [
        PROJECT_DIR / "run_quality_screening.log",
        Path("run_quality_screening.log"),
        PROJECT_DIR / "run_quality_fast.log",
        Path("run_quality_fast.log"),
    ]
    log_file = None
    for lc in log_candidates:
        if lc.exists() and lc.stat().st_size > 0:
            log_file = lc
            break

    summary_candidates = [
        PROJECT_DIR / "results" / "run_quality_screening" / "summary.csv",
        Path("results/run_quality_screening/summary.csv"),
        PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv",
        Path("results/run_quality_fast/summary.csv"),
    ]'''

content = content.replace(old_parser_start, new_parser_start, 1)

# Replace FAST_MODULE_LIST usage with active_module_list
content = content.replace("total_units = len(FAST_MODULE_LIST) * total_seeds", "total_units = len(active_module_list) * total_seeds")
content = content.replace("curr_idx = FAST_MODULE_LIST.index(current_mod) if current_mod in FAST_MODULE_LIST else 0", "curr_idx = active_module_list.index(current_mod) if current_mod in active_module_list else 0")
content = content.replace("for idx, mod in enumerate(FAST_MODULE_LIST):", "for idx, mod in enumerate(active_module_list):")
content = content.replace('"total": 6,', '"total": len(active_module_list),')

# 4. In detailed_results candidates
old_det_cand = '''    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_fast" / "detailed_results.json",'''

new_det_cand = '''    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_screening" / "detailed_results.json",
        Path("results/run_quality_screening/detailed_results.json"),
        PROJECT_DIR / "results" / "run_quality_fast" / "detailed_results.json",'''

content = content.replace(old_det_cand, new_det_cand, 1)

# 5. In RunManager.list_runs()
old_run_mgr = '''            fast_exists = (PROJECT_DIR / "results" / "run_quality_fast").exists() or (PROJECT_DIR / "run_quality_fast.log").exists()
            running = is_sweep_running()
            if running or fast_exists or active_results:
                is_active = True
                p_cnt = sum(1 for r in active_results if r.get("status") == "PASS")
                f_cnt = sum(1 for r in active_results if r.get("status") == "FAIL")
                t_cnt = sum(1 for r in active_results if r.get("status") == "TIMEOUT")
                ts_str = datetime.now().isoformat()
                if active_results and "timestamp" in active_results[0]:
                    ts_val = active_results[0]["timestamp"]
                    ts_str = ts_val.isoformat() if isinstance(ts_val, datetime) else str(ts_val)

                runs.append({
                    "run_id": "run_active",
                    "name": "Phase 2 Quality Sweep (Fast)",
                    "timestamp": ts_str,
                    "num_combinations": 6,'''

new_run_mgr = '''            screen_exists = (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()
            fast_exists = (PROJECT_DIR / "results" / "run_quality_fast").exists() or (PROJECT_DIR / "run_quality_fast.log").exists()
            running = is_sweep_running()
            if running or screen_exists or fast_exists or active_results:
                is_active = True
                p_cnt = sum(1 for r in active_results if r.get("status") == "PASS")
                f_cnt = sum(1 for r in active_results if r.get("status") == "FAIL")
                t_cnt = sum(1 for r in active_results if r.get("status") == "TIMEOUT")
                ts_str = datetime.now().isoformat()
                if active_results and "timestamp" in active_results[0]:
                    ts_val = active_results[0]["timestamp"]
                    ts_str = ts_val.isoformat() if isinstance(ts_val, datetime) else str(ts_val)

                num_c = 20 if screen_exists else 6
                r_name = "Phase 2 Quality Screening (20 Modules)" if screen_exists else "Phase 2 Quality Sweep (Fast)"

                runs.append({
                    "run_id": "run_active",
                    "name": r_name,
                    "timestamp": ts_str,
                    "num_combinations": num_c,'''

content = content.replace(old_run_mgr, new_run_mgr, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py updated for 20-module screening run!")
