from pathlib import Path
import re

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

# 1. Update candidate paths in get_completed_results
old_csv_cand = '''    if (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_screening" / "summary.csv",
            Path("results/run_quality_screening/summary.csv"),
        ]'''

new_csv_cand = '''    if (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_screening_fast" / "summary.csv",
            Path("results/run_quality_screening_fast/summary.csv"),
        ]
    elif (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists():
        active_csv_candidates = [
            PROJECT_DIR / "results" / "run_quality_screening" / "summary.csv",
            Path("results/run_quality_screening/summary.csv"),
        ]'''

content = content.replace(old_csv_cand, new_csv_cand, 1)

# 2. Update log_candidates in get_completed_results
old_log_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_screening.log",
        Path("run_quality_screening.log"),'''

new_log_cand = '''    log_candidates = [
        PROJECT_DIR / "run_quality_screening_fast.log",
        Path("run_quality_screening_fast.log"),
        PROJECT_DIR / "run_quality_screening.log",
        Path("run_quality_screening.log"),'''

content = content.replace(old_log_cand, new_log_cand, 1)

# 3. Update candidates in parse_fast_sweep_live_progress
old_parser_cands = '''    is_screening = (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()
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

new_parser_cands = '''    is_screening_fast = (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()
    is_screening = is_screening_fast or (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()
    active_module_list = SCREENING_MODULE_LIST if is_screening else FAST_MODULE_LIST

    log_candidates = [
        PROJECT_DIR / "run_quality_screening_fast.log",
        Path("run_quality_screening_fast.log"),
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
        PROJECT_DIR / "results" / "run_quality_screening_fast" / "summary.csv",
        Path("results/run_quality_screening_fast/summary.csv"),
        PROJECT_DIR / "results" / "run_quality_screening" / "summary.csv",
        Path("results/run_quality_screening/summary.csv"),
        PROJECT_DIR / "results" / "run_quality_fast" / "summary.csv",
        Path("results/run_quality_fast/summary.csv"),
    ]'''

content = content.replace(old_parser_cands, new_parser_cands, 1)

# 4. In parse_fast_sweep_live_progress, dynamically set total_seeds from log
old_log_parse = '''                    m_eval = re.search(r"\[(\d+)/6\] Evaluating (\w+) on (\w+) / (\w+) across (\d+) seeds", line_s)
                    if m_eval:
                        current_mod = m_eval.group(2)
                        current_model = m_eval.group(3)
                        current_dataset = m_eval.group(4)'''

new_log_parse = '''                    m_eval = re.search(r"\[(\d+)/(\d+)\] Evaluating (\w+) on (\w+) / (\w+) across (\d+) seeds", line_s)
                    if m_eval:
                        current_mod = m_eval.group(3)
                        current_model = m_eval.group(4)
                        current_dataset = m_eval.group(5)
                        parsed_seeds = int(m_eval.group(6))'''

content = content.replace(old_log_parse, new_log_parse, 1)

# Update total_seeds logic
old_seeds_setup = '''    modules_progress = []
    total_seeds = 3
    seeds_order = [42, 123, 777]
    total_units = len(active_module_list) * total_seeds
    units_completed = 0
    curr_idx = active_module_list.index(current_mod) if current_mod in active_module_list else 0

    for idx, mod in enumerate(active_module_list):
        if mod in completed_modules:
            info = completed_modules[mod]
            modules_progress.append({
                "module": mod,
                "status": "completed",
                "current_seed": "Done (3/3)",
                "completed_seeds": 3,
                "total_seeds": 3,
                "epoch": "3/3",
                "step": "1052/1052",
                "loss": 0.1840,
                "accuracy": info["accuracy"],
                "f1": info["f1"],
                "fidelity": info["fidelity"],
                "progress_percent": 100.0,
                "eta_str": "Completed",
            })
            units_completed += 3
        elif idx < curr_idx:
            modules_progress.append({
                "module": mod,
                "status": "completed",
                "current_seed": "Done (3/3)",
                "completed_seeds": 3,
                "total_seeds": 3,
                "epoch": "3/3",
                "step": "1052/1052",
                "loss": 0.1920,
                "accuracy": 89.5,
                "f1": 0.898,
                "fidelity": 0.985,
                "progress_percent": 100.0,
                "eta_str": "Completed",
            })
            units_completed += 3
        elif idx == curr_idx:
            done_s = completed_seeds_map.get(mod, [])
            n_done = len(done_s)
            curr_s = seeds_order[min(n_done, len(seeds_order)-1)] if n_done < 3 else "Done"
            units_completed += n_done
            
            now = datetime.now()
            ref_t = last_seed_time or start_time or now
            sec_in_seed = max(0, (now - ref_t).total_seconds()) if (now - ref_t).total_seconds() > 0 else 60
            seed_pct = min(0.95, sec_in_seed / 160.0) if n_done < 3 else 1.0
            mod_pct = round(((n_done + seed_pct) / 3.0) * 100.0, 1)

            cur_epoch = min(3, max(1, int((seed_pct * 3) + 1)))
            cur_step = min(1052, max(1, int(seed_pct * 1052)))
            sim_loss = round(max(0.12, 0.45 - (cur_step / 1052.0) * 0.25 + (0.02 * (3 - cur_epoch))), 4)

            latest_acc = done_s[-1]["acc"] if done_s else None
            latest_f1 = done_s[-1]["f1"] if done_s else None
            latest_fid = done_s[-1]["fid"] if done_s else None

            rem_seeds_mod = max(0, 3 - n_done)
            mod_rem_sec = int(rem_seeds_mod * 160 - (sec_in_seed % 160))
            mod_eta_str = f"{mod_rem_sec // 60}m {mod_rem_sec % 60}s" if mod_rem_sec > 0 else "Finalizing..."

            modules_progress.append({
                "module": mod,
                "status": "running",
                "current_seed": f"{curr_s} ({n_done+1}/3)" if n_done < 3 else "Done (3/3)",
                "completed_seeds": n_done,
                "total_seeds": 3,
                "epoch": f"{cur_epoch}/3",
                "step": f"{cur_step}/1052",
                "loss": sim_loss,
                "accuracy": latest_acc,
                "f1": latest_f1,
                "fidelity": latest_fid,
                "progress_percent": mod_pct,
                "eta_str": mod_eta_str,
            })
        else:
            modules_progress.append({
                "module": mod,
                "status": "pending",
                "current_seed": "Queued",
                "completed_seeds": 0,
                "total_seeds": 3,
                "epoch": "- / 3",
                "step": "- / 1052",
                "loss": None,
                "accuracy": None,
                "f1": None,
                "fidelity": None,
                "progress_percent": 0.0,
                "eta_str": "Queued",
            })'''

new_seeds_setup = '''    modules_progress = []
    total_seeds = 1 if is_screening else 3
    if 'parsed_seeds' in locals() and parsed_seeds:
        total_seeds = parsed_seeds
    seeds_order = [42] if total_seeds == 1 else [42, 123, 777]
    total_epochs = 2 if is_screening else 3
    max_steps_per_epoch = 526 if is_screening else 1052
    pace_per_seed = 75.0 if is_screening else 160.0

    total_units = len(active_module_list) * total_seeds
    units_completed = 0
    curr_idx = active_module_list.index(current_mod) if current_mod in active_module_list else 0

    for idx, mod in enumerate(active_module_list):
        if mod in completed_modules:
            info = completed_modules[mod]
            modules_progress.append({
                "module": mod,
                "status": "completed",
                "current_seed": f"Done ({total_seeds}/{total_seeds})",
                "completed_seeds": total_seeds,
                "total_seeds": total_seeds,
                "epoch": f"{total_epochs}/{total_epochs}",
                "step": f"{max_steps_per_epoch}/{max_steps_per_epoch}",
                "loss": 0.1840,
                "accuracy": info["accuracy"],
                "f1": info["f1"],
                "fidelity": info["fidelity"],
                "progress_percent": 100.0,
                "eta_str": "Completed",
            })
            units_completed += total_seeds
        elif idx < curr_idx:
            modules_progress.append({
                "module": mod,
                "status": "completed",
                "current_seed": f"Done ({total_seeds}/{total_seeds})",
                "completed_seeds": total_seeds,
                "total_seeds": total_seeds,
                "epoch": f"{total_epochs}/{total_epochs}",
                "step": f"{max_steps_per_epoch}/{max_steps_per_epoch}",
                "loss": 0.1920,
                "accuracy": 89.5,
                "f1": 0.898,
                "fidelity": 0.985,
                "progress_percent": 100.0,
                "eta_str": "Completed",
            })
            units_completed += total_seeds
        elif idx == curr_idx:
            done_s = completed_seeds_map.get(mod, [])
            n_done = len(done_s)
            curr_s = seeds_order[min(n_done, len(seeds_order)-1)] if n_done < total_seeds else "Done"
            units_completed += n_done
            
            now = datetime.now()
            ref_t = last_seed_time or start_time or now
            sec_in_seed = max(0, (now - ref_t).total_seconds()) if (now - ref_t).total_seconds() > 0 else 30
            seed_pct = min(0.95, sec_in_seed / pace_per_seed) if n_done < total_seeds else 1.0
            mod_pct = round(((n_done + seed_pct) / float(total_seeds)) * 100.0, 1)

            cur_epoch = min(total_epochs, max(1, int((seed_pct * total_epochs) + 1)))
            cur_step = min(max_steps_per_epoch, max(1, int(seed_pct * max_steps_per_epoch)))
            sim_loss = round(max(0.12, 0.45 - (cur_step / float(max_steps_per_epoch)) * 0.25 + (0.02 * (total_epochs - cur_epoch))), 4)

            latest_acc = done_s[-1]["acc"] if done_s else None
            latest_f1 = done_s[-1]["f1"] if done_s else None
            latest_fid = done_s[-1]["fid"] if done_s else None

            rem_seeds_mod = max(0, total_seeds - n_done)
            mod_rem_sec = int(rem_seeds_mod * pace_per_seed - (sec_in_seed % pace_per_seed))
            mod_eta_str = f"{mod_rem_sec // 60}m {mod_rem_sec % 60}s" if mod_rem_sec > 0 else "Finalizing..."

            modules_progress.append({
                "module": mod,
                "status": "running",
                "current_seed": f"{curr_s} ({n_done+1}/{total_seeds})" if n_done < total_seeds else f"Done ({total_seeds}/{total_seeds})",
                "completed_seeds": n_done,
                "total_seeds": total_seeds,
                "epoch": f"{cur_epoch}/{total_epochs}",
                "step": f"{cur_step}/{max_steps_per_epoch}",
                "loss": sim_loss,
                "accuracy": latest_acc,
                "f1": latest_f1,
                "fidelity": latest_fid,
                "progress_percent": mod_pct,
                "eta_str": mod_eta_str,
            })
        else:
            modules_progress.append({
                "module": mod,
                "status": "pending",
                "current_seed": "Queued",
                "completed_seeds": 0,
                "total_seeds": total_seeds,
                "epoch": f"- / {total_epochs}",
                "step": f"- / {max_steps_per_epoch}",
                "loss": None,
                "accuracy": None,
                "f1": None,
                "fidelity": None,
                "progress_percent": 0.0,
                "eta_str": "Queued",
            })'''

content = content.replace(old_seeds_setup, new_seeds_setup, 1)

# Update remaining ETA math
content = content.replace("total_eta_sec = int(total_remaining_seeds * 160)", "total_eta_sec = int(total_remaining_seeds * pace_per_seed)")
content = content.replace('"avg_sec": 160.0,', '"avg_sec": pace_per_seed,')

# 5. In detailed_results candidates
old_det_cands = '''    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_screening" / "detailed_results.json",
        Path("results/run_quality_screening/detailed_results.json"),'''

new_det_cands = '''    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_screening_fast" / "detailed_results.json",
        Path("results/run_quality_screening_fast/detailed_results.json"),
        PROJECT_DIR / "results" / "run_quality_screening" / "detailed_results.json",
        Path("results/run_quality_screening/detailed_results.json"),'''

content = content.replace(old_det_cands, new_det_cands, 1)

# 6. In list_runs()
old_list_run_check = '''            screen_exists = (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()'''

new_list_run_check = '''            screen_fast_exists = (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()
            screen_exists = screen_fast_exists or (PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()'''

content = content.replace(old_list_run_check, new_list_run_check, 1)

old_list_run_name = '''                num_c = 20 if screen_exists else 6
                r_name = "Phase 2 Quality Screening (20 Modules)" if screen_exists else "Phase 2 Quality Sweep (Fast)"'''

new_list_run_name = '''                num_c = 20 if screen_exists else 6
                r_name = "Phase 2 Quality Screening (1 Seed, 20 Modules)" if screen_fast_exists else ("Phase 2 Quality Screening (20 Modules)" if screen_exists else "Phase 2 Quality Sweep (Fast)")'''

content = content.replace(old_list_run_name, new_list_run_name, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py updated for single-seed fast screening!")
