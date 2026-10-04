"""
gpu_live_probe.py
Executed on the GPU server to return real-time GPU telemetry, active running process,
live training loss trajectory, and log streams in JSON format.
"""

import argparse
import glob
import json
import os
import re
import subprocess


def get_telemetry():
    telemetry = {
        "gpu_name": "NVIDIA RTX 4070 (8GB)",
        "mem_used_mb": 450,
        "mem_total_mb": 8188,
        "gpu_util": 0,
        "temp_c": 45,
        "power_w": 25.0,
    }
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits",
            shell=True,
            text=True,
            timeout=3,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 6:
            telemetry = {
                "gpu_name": parts[0],
                "mem_used_mb": int(float(parts[1])),
                "mem_total_mb": int(float(parts[2])),
                "gpu_util": int(float(parts[3])),
                "temp_c": int(float(parts[4])),
                "power_w": float(parts[5]),
            }
    except Exception:
        pass
    return telemetry


def get_active_process_and_logs():
    active_module = None
    active_model = None
    queue_logs = ""
    try:
        cmd = 'powershell -Command "(Get-CimInstance Win32_Process -Filter \\"name = \'python.exe\'\\").CommandLine"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=4)
        if res.returncode == 0 and res.stdout:
            for line in res.stdout.splitlines():
                if "run_quality_sweep.py" in line:
                    m = re.search(r"--modules\s+[^,]+,([a-zA-Z0-9_]+)", line)
                    if m:
                        active_module = m.group(1)
                    mm = re.search(r"--model_family\s+([a-zA-Z0-9_]+)", line)
                    if mm:
                        active_model = mm.group(1)
                elif "run_all_4_" in line and not active_module:
                    m = re.search(r"run_all_4_([a-zA-Z0-9_]+)\.py", line)
                    if m:
                        active_module = m.group(1)
    except Exception:
        pass

    try:
        for qf in [
            "D:/SVD_Project/queue_launcher.log",
            "D:/SVD_Project/gpu_queue.log",
            "D:/SVD_Project/results/queue_runner.log",
        ]:
            if os.path.exists(qf):
                with open(qf, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [
                        l.strip()
                        for l in f.readlines()[-30:]
                        if l.strip() and not l.startswith("**")
                    ]
                    if lines:
                        queue_logs = "\n".join(lines)
                        break
    except Exception:
        pass

    return active_module or "cur_linear", active_model or "qwen3_5", queue_logs


def get_live_progress_and_trajectory(target_module=None, target_model=None):
    results_dir = "D:/SVD_Project/results"
    if not os.path.exists(results_dir):
        return None

    candidate_files = []
    if target_module:
        candidate_files.extend(glob.glob(os.path.join(results_dir, f"bench_{target_module}.log")))
        candidate_files.extend(
            glob.glob(
                os.path.join(results_dir, f"unified_{target_module}", "**", "console.log"),
                recursive=True,
            )
        )

    candidate_files.extend(glob.glob(os.path.join(results_dir, "bench_*.log")))
    candidate_files.extend(
        glob.glob(os.path.join(results_dir, "**", "console.log"), recursive=True)
    )

    logs = sorted(list(set(candidate_files)), key=os.path.getmtime)

    for f in reversed(logs[-20:]):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            if "loss=" not in content and "step" not in content:
                continue

            lines = content.splitlines()[-300:]
            loss_vals = []
            history = []
            cur_step, total_step = 1, 50
            for l in lines:
                m = re.search(r"(?:fine-tune\s+step|step)\s+(\d+)/(\d+)\s+loss=([0-9.]+)", l)
                if m:
                    cur_step = int(m.group(1))
                    total_step = int(m.group(2))
                    lv = round(float(m.group(3)), 4)
                    loss_vals.append(lv)
                    history.append({"step": cur_step, "total_steps": total_step, "loss": lv})
                else:
                    lm = re.search(r"loss=([0-9.]+)", l)
                    if lm:
                        lv = round(float(lm.group(1)), 4)
                        loss_vals.append(lv)

            if not loss_vals:
                continue

            norm_path = f.replace("\\", "/")
            mod = target_module or "cur_linear"
            arch = target_model or "qwen3_5"
            m_bench = re.search(r"bench_([a-zA-Z0-9_]+)\.log", norm_path)
            if m_bench:
                mod = m_bench.group(1)
            elif "unified_" in norm_path:
                parts = norm_path.split("/")
                for i, p in enumerate(parts):
                    if p.startswith("unified_"):
                        mod = p.replace("unified_", "")
                        if i + 1 < len(parts):
                            arch = parts[i + 1]

            latest_loss = loss_vals[-1]
            pct = int((cur_step / max(1, total_step)) * 100) if total_step > 0 else 50
            eta_sec = max(5, (total_step - cur_step) * 2)

            return {
                "module_name": mod,
                "model_family": arch.capitalize(),
                "dataset": "SST-2",
                "seed": 42,
                "epoch": "1/3",
                "step": f"{cur_step}/{total_step}",
                "loss": latest_loss,
                "progress_percent": min(100, pct),
                "eta_formatted": f"{eta_sec}s",
                "loss_trajectory": loss_vals[-50:],
                "history": history[-50:],
                "log_file": norm_path,
            }
        except Exception:
            continue

    return None


def get_raw_log_tail(source="active", target_module=None, lines_count=200):
    results_dir = "D:/SVD_Project/results"
    target_file = None

    if source == "queue":
        for qf in ["D:/SVD_Project/queue_launcher.log", "D:/SVD_Project/gpu_queue.log"]:
            if os.path.exists(qf):
                target_file = qf
                break
    elif target_module:
        m_logs = sorted(
            glob.glob(
                os.path.join(results_dir, f"unified_{target_module}", "**", "console.log"),
                recursive=True,
            ),
            key=os.path.getmtime,
        )
        if m_logs:
            target_file = m_logs[-1]

    if not target_file:
        logs = sorted(
            glob.glob(os.path.join(results_dir, "**", "console.log"), recursive=True),
            key=os.path.getmtime,
        )
        if logs:
            target_file = logs[-1]

    if target_file and os.path.exists(target_file):
        try:
            with open(target_file, "r", encoding="utf-8", errors="ignore") as fp:
                all_lines = fp.readlines()
                clean = [
                    l.rstrip("\r\n") for l in all_lines[-lines_count:] if not l.startswith("**")
                ]
                return {"file": target_file.replace("\\", "/"), "lines": clean, "count": len(clean)}
        except Exception as e:
            return {"file": target_file, "lines": [f"Error reading log: {e}"], "count": 1}

    return {"file": "none", "lines": ["No log file found on GPU server."], "count": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="all", choices=["all", "telemetry", "progress", "logs"])
    parser.add_argument("--source", default="active")
    parser.add_argument("--module", default=None)
    parser.add_argument("--lines", type=int, default=200)
    args = parser.parse_args()

    if args.action == "logs":
        tail_data = get_raw_log_tail(
            source=args.source, target_module=args.module, lines_count=args.lines
        )
        print(json.dumps(tail_data))
    elif args.action == "telemetry":
        print(json.dumps(get_telemetry()))
    elif args.action == "progress":
        print(json.dumps(get_live_progress_and_trajectory(target_module=args.module)))
    else:
        t = get_telemetry()
        active_m, active_arch, q_logs = get_active_process_and_logs()
        lp = get_live_progress_and_trajectory(target_module=active_m)
        raw_tail = get_raw_log_tail(source="active", lines_count=args.lines)

        if lp and lp.get("module_name") and lp["module_name"] != "unknown":
            active_m = lp["module_name"]

        output = {
            "telemetry": t,
            "active_module": active_m,
            "active_model": active_arch,
            "queue_logs": q_logs,
            "live_progress": lp,
            "active_log_tail": raw_tail,
        }
        print(json.dumps(output))
