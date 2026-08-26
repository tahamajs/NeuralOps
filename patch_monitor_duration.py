from pathlib import Path
import re

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

# Let's see the parser in live_monitor.py around parse_fast_sweep_live_progress
# We will inject the mod_durations and mod_starts parsing, and format_duration helper
old_block_search = '''    current_mod = "full"
    current_model = "DISTILBERT"
    current_dataset = "SST2"
    completed_seeds_map = {}
    start_time = None
    last_seed_time = None

    if log_file and log_file.exists():'''

new_block_replace = '''    def format_duration(sec: Optional[float]) -> str:
        if sec is None or sec < 0:
            return "-"
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}m {s:02d}s"

    current_mod = "full"
    current_model = "DISTILBERT"
    current_dataset = "SST2"
    completed_seeds_map = {}
    mod_durations = {}
    mod_starts = {}
    start_time = None
    last_seed_time = None

    if log_file and log_file.exists():'''

content = content.replace(old_block_search, new_block_replace, 1)

# In the log parsing loop, add regex for start time and completed time
old_loop_block = '''                    m_eval = re.search(r"\[(\d+)/(\d+)\] Evaluating (\w+) on (\w+) / (\w+) across (\d+) seeds", line_s)
                    if m_eval:
                        current_mod = m_eval.group(3)
                        current_model = m_eval.group(4)
                        current_dataset = m_eval.group(5)
                        parsed_seeds = int(m_eval.group(6))'''

new_loop_block = '''                    m_eval = re.search(r"\[(\d+)/(\d+)\] Evaluating (\w+) on (\w+) / (\w+) across (\d+) seeds", line_s)
                    if m_eval:
                        current_mod = m_eval.group(3)
                        current_model = m_eval.group(4)
                        current_dataset = m_eval.group(5)
                        parsed_seeds = int(m_eval.group(6))
                        m_ts_eval = re.search(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line_s)
                        if m_ts_eval:
                            try:
                                mod_starts[current_mod] = datetime.strptime(m_ts_eval.group(1), "%Y-%m-%d %H:%M:%S")
                            except Exception:
                                pass

                    m_done = re.search(r"-->\s+(\w+)\s+on.*completed in\s+([\d\.]+)s", line_s)
                    if m_done:
                        d_mod = m_done.group(1)
                        d_sec = float(m_done.group(2))
                        mod_durations[d_mod] = d_sec'''

content = content.replace(old_loop_block, new_loop_block, 1)

# In modules_progress creation, add duration fields
old_mod_completed = '''        if mod in completed_modules:
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
            })'''

new_mod_completed = '''        if mod in completed_modules:
            info = completed_modules[mod]
            d_sec = mod_durations.get(mod)
            d_fmt = format_duration(d_sec) if d_sec is not None else "Completed"
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
                "duration_seconds": round(d_sec, 1) if d_sec is not None else None,
                "duration_formatted": d_fmt,
                "eta_str": "Completed",
            })'''

content = content.replace(old_mod_completed, new_mod_completed, 1)

# In modules_progress for idx < curr_idx
old_mod_prev = '''        elif idx < curr_idx:
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
            })'''

new_mod_prev = '''        elif idx < curr_idx:
            d_sec = mod_durations.get(mod)
            d_fmt = format_duration(d_sec) if d_sec is not None else "Completed"
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
                "duration_seconds": round(d_sec, 1) if d_sec is not None else None,
                "duration_formatted": d_fmt,
                "eta_str": "Completed",
            })'''

content = content.replace(old_mod_prev, new_mod_prev, 1)

# In modules_progress for idx == curr_idx (running)
old_mod_running = '''            modules_progress.append({
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
            })'''

new_mod_running = '''            m_start = mod_starts.get(mod, ref_t)
            elapsed_sec = max(0.0, (datetime.now() - m_start).total_seconds())
            elapsed_fmt = f"{format_duration(elapsed_sec)} elapsed"
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
                "duration_seconds": round(elapsed_sec, 1),
                "elapsed_seconds": round(elapsed_sec, 1),
                "duration_formatted": elapsed_fmt,
                "eta_str": mod_eta_str,
            })'''

content = content.replace(old_mod_running, new_mod_running, 1)

# In modules_progress for pending
old_mod_pending = '''            modules_progress.append({
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

new_mod_pending = '''            modules_progress.append({
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
                "duration_seconds": None,
                "elapsed_seconds": None,
                "duration_formatted": "-",
                "eta_str": "Queued",
            })'''

content = content.replace(old_mod_pending, new_mod_pending, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py successfully patched with Duration / Elapsed fields!")
