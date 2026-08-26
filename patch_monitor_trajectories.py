from pathlib import Path
import re

p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

# 1. In _build_stats_response, parse trajectories from detailed_results.json
old_traj_block = '''    serializable_results = [
        {
            "id": idx,
            "track": r.get("track", "quality_phase2"),
            "module": r["module"],
            "model": r["model"],
            "dataset": r["dataset"],
            "status": r["status"],
            "fwd_ms": r.get("fwd_ms"),
            "bwd_ms": r.get("bwd_ms"),
            "mem_mb": r.get("mem_mb"),
            "acc_pct": r.get("acc_pct"),
            "f1_score": r.get("f1_score"),
            "fidelity_score": r.get("fidelity_score"),
            "throughput": r.get("throughput"),
            "p_value": r.get("p_value"),
            "ci_lower": r.get("ci_lower"),
            "ci_upper": r.get("ci_upper"),
            "error_detail": r.get("error_detail", "None / Success"),
            "time": r["timestamp"].strftime("%H:%M:%S") if isinstance(r["timestamp"], datetime) else str(r["timestamp"]),
        }
        for idx, r in enumerate(results, 1)
    ]'''

new_traj_block = '''    trajectories = {}
    det_candidates = [
        PROJECT_DIR / "results" / "run_quality_fast" / "detailed_results.json",
        Path("results/run_quality_fast/detailed_results.json"),
        PROJECT_DIR / "results" / "run_quality_pretrained" / "detailed_results.json",
        Path("results/run_quality_pretrained/detailed_results.json"),
    ]
    for dpath in det_candidates:
        if dpath.exists() and dpath.stat().st_size > 0:
            try:
                with open(dpath, "r", encoding="utf-8") as f:
                    det_items = json.load(f)
                    for item in det_items:
                        m_name = item.get("module_name", item.get("module"))
                        if m_name and "trajectories" in item and item["trajectories"]:
                            loss_list = item["trajectories"].get("loss", [])
                            if len(loss_list) > 100:
                                stride = max(1, len(loss_list) // 100)
                                ds_loss = [loss_list[i] for i in range(0, len(loss_list), stride)]
                            else:
                                ds_loss = loss_list
                            trajectories[m_name] = {
                                "loss": ds_loss,
                                "grad_norm": item["trajectories"].get("grad_norm", [])[:100],
                                "fwd_latency_ms": item["trajectories"].get("fwd_latency_ms", [])[:100],
                            }
            except Exception:
                pass

    serializable_results = [
        {
            "id": idx,
            "track": r.get("track", "quality_phase2"),
            "module": r["module"],
            "model": r["model"],
            "dataset": r["dataset"],
            "status": r["status"],
            "fwd_ms": r.get("fwd_ms"),
            "bwd_ms": r.get("bwd_ms"),
            "mem_mb": r.get("mem_mb"),
            "acc_pct": r.get("acc_pct"),
            "f1_score": r.get("f1_score"),
            "fidelity_score": r.get("fidelity_score"),
            "throughput": r.get("throughput"),
            "p_value": r.get("p_value"),
            "ci_lower": r.get("ci_lower"),
            "ci_upper": r.get("ci_upper"),
            "error_detail": r.get("error_detail", "None / Success"),
            "time": r["timestamp"].strftime("%H:%M:%S") if isinstance(r["timestamp"], datetime) else str(r["timestamp"]),
            "trajectories": trajectories.get(r["module"], {}),
        }
        for idx, r in enumerate(results, 1)
    ]'''

content = content.replace(old_traj_block, new_traj_block, 1)

# Include "trajectories": trajectories in response
old_resp = '''        "live_progress": live_prog,
        "module_progress": mod_prog,
        "results": serializable_results,'''

new_resp = '''        "live_progress": live_prog,
        "module_progress": mod_prog,
        "trajectories": trajectories,
        "results": serializable_results,'''

content = content.replace(old_resp, new_resp, 1)

p.write_text(content, encoding="utf-8")
print("Codes/framework/live_monitor.py patched with trajectories support!")
