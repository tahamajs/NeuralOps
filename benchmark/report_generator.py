"""Comprehensive report, summary table, leaderboard, and visualization generator."""

from __future__ import annotations
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_scores(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Apply multi-dimensional scoring rubric across quality, throughput, stability, and robustness.
    """
    scores: Dict[str, Dict[str, float]] = {}

    # Find full baseline reference
    full_entry = results.get("full", {})
    full_glue_avg = full_entry.get("avg_glue", 0.85) or 0.85
    full_tps = full_entry.get("profiling", {}).get("configurations", {}).get("bs32_seq512", {}).get("tokens_per_sec", 15000.0) or 15000.0
    full_mem = full_entry.get("profiling", {}).get("configurations", {}).get("bs32_seq512", {}).get("peak_vram_mb", 1000.0) or 1000.0
    if not isinstance(full_mem, (int, float)) or full_mem <= 0:
        full_mem = 1000.0

    for model_name, data in results.items():
        if data.get("status") not in ("OK", "SANITY_PASSED"):
            scores[model_name] = {
                "quality_score": 0.0,
                "efficiency_score": 0.0,
                "stability_score": 0.0,
                "robustness_score": 0.0,
                "total_score": 0.0,
            }
            continue

        # 1. Quality Score (40%)
        m_glue = data.get("avg_glue", 0.0)
        quality_score = min(max((m_glue / max(full_glue_avg, 1e-4)) * 100.0, 0.0), 120.0)

        # 2. Efficiency Score (20%)
        cfg_512 = data.get("profiling", {}).get("configurations", {}).get("bs32_seq512", {})
        m_tps = cfg_512.get("tokens_per_sec", 0.0) or 0.0
        m_mem = cfg_512.get("peak_vram_mb", 0.0)

        speed_ratio = (m_tps / max(full_tps, 1.0)) if full_tps > 0 else 1.0
        if isinstance(m_mem, (int, float)) and m_mem > 0:
            mem_ratio = (full_mem / m_mem)
        else:
            mem_ratio = 1.0

        eff_raw = (0.6 * min(speed_ratio, 2.0)) + (0.4 * min(mem_ratio, 2.0))
        efficiency_score = min(eff_raw * 50.0, 100.0)

        # 3. Stability Score (20%)
        stab_info = data.get("stability", {})
        tot_seeds = stab_info.get("total_seeds", 3) or 3
        fail_seeds = stab_info.get("failed_seeds", 0) or 0
        succ_seeds = max(tot_seeds - fail_seeds, 0)
        stability_score = (succ_seeds / tot_seeds) * 100.0

        # 4. Robustness Score (20%)
        cv_lr = stab_info.get("cv_lr", 0.1) or 0.1
        robustness_score = max((1.0 - min(cv_lr, 1.0)) * 100.0, 0.0)

        # Total Composite Score
        total_score = (
            0.40 * quality_score +
            0.20 * efficiency_score +
            0.20 * stability_score +
            0.20 * robustness_score
        )

        scores[model_name] = {
            "quality_score": float(quality_score),
            "efficiency_score": float(efficiency_score),
            "stability_score": float(stability_score),
            "robustness_score": float(robustness_score),
            "total_score": float(total_score),
        }

    return scores


def generate_summary_table(results: Dict[str, Any], scores: Dict[str, Dict[str, float]], output_path: str) -> pd.DataFrame:
    """Generate structured summary DataFrame and export to CSV."""
    rows = []
    # Collect all tasks that appear in results
    standard_order = ["sst2", "qnli", "rte", "mrpc", "cola", "stsb", "qqp", "mnli"]
    observed_tasks = set()
    for data in results.values():
        for t in data.get("glue_tasks", {}).keys():
            observed_tasks.add(t)

    tasks = [t for t in standard_order if t in observed_tasks] + [t for t in sorted(observed_tasks) if t not in standard_order]
    if not tasks:
        tasks = ["sst2", "mrpc"]

    for model_name, data in results.items():
        m_scores = scores.get(model_name, {})
        tot_score = m_scores.get("total_score", 0.0)

        glue_results = data.get("glue_tasks", {})
        task_vals = {}
        for t in tasks:
            t_res = glue_results.get(t, {})
            t_score = t_res.get("eval_metrics", {}).get("score", t_res.get("eval_metrics", {}).get("accuracy", None))
            task_vals[t] = f"{t_score:.4f}" if t_score is not None else "N/A"

        avg_glue = data.get("avg_glue", 0.0)
        avg_str = f"{avg_glue:.4f}" if avg_glue > 0 else "N/A"

        prof_512 = data.get("profiling", {}).get("configurations", {}).get("bs32_seq512", {})
        tps = prof_512.get("tokens_per_sec")
        tps_str = f"{tps:,.1f}" if isinstance(tps, (int, float)) else "N/A"

        vram = prof_512.get("peak_vram_mb")
        vram_str = f"{vram:.1f} MB" if isinstance(vram, (int, float)) else str(vram or "N/A")

        num_params = data.get("num_params", 0)
        param_str = f"{num_params / 1e6:.2f}M" if num_params > 0 else "N/A"

        is_stable = data.get("stability", {}).get("stable", False)
        stab_str = "PASS" if is_stable else ("FAIL" if data.get("status") == "OK" else "N/A")

        row = {
            "Model": model_name,
            "Score": round(tot_score, 2),
            "Avg GLUE": avg_str,
        }
        for t in tasks:
            row[t.upper()] = task_vals.get(t, "N/A")

        row.update({
            "Params": param_str,
            "Tokens/sec (512)": tps_str,
            "Peak VRAM (512)": vram_str,
            "Stability": stab_str,
            "Status": data.get("status", "UNKNOWN"),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    if "Score" in df.columns and not df.empty:
        df = df.sort_values(by="Score", ascending=False)

    df.to_csv(output_path, index=False)
    return df


def generate_leaderboard_markdown(df: pd.DataFrame, output_path: str) -> None:
    """Generate Markdown leaderboard file with formatted rankings and status highlights."""
    if df.empty:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 🏆 Attention Mechanism Benchmark Leaderboard\n\nNo models evaluated yet.\n")
        return

    cols = list(df.columns)
    header = ["Rank"] + cols
    header_line = "| " + " | ".join(header) + " |"
    align_line = "|:---:|" + "|".join([":---:" if c in ("Score", "Rank", "Stability", "Status") else ":---" for c in cols]) + "|"

    lines = [
        "# 🏆 Attention Mechanism Benchmark Leaderboard",
        "",
        "> Comprehensive evaluation of custom linear, low-rank, state-space, kernelized, and sparse attention architectures.",
        "",
        header_line,
        align_line,
    ]

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        row_vals = [str(rank)]
        for c in cols:
            val = str(row[c])
            if c == "Model":
                row_vals.append(f"**{val}**")
            elif c == "Score":
                row_vals.append(f"**{val}**")
            elif c == "Stability":
                row_vals.append(f"**{val}**" if val == "PASS" else f"`{val}`")
            elif c == "Status":
                row_vals.append(f"`{val}`")
            else:
                row_vals.append(val)
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.extend([
        "",
        "### 📊 Scoring Rubric Breakdown",
        "- **Quality (40%)**: Ratio of model Average GLUE score against Full Standard Softmax Attention.",
        "- **Efficiency (20%)**: Hardware execution throughput (tokens/sec) and GPU VRAM footprint at sequence length 512.",
        "- **Stability (20%)**: Convergence consistency across multi-seed runs without majority-class collapse.",
        "- **Robustness (20%)**: Hyperparameter sensitivity coefficient of variation (CV) across learning rates and batch sizes.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_plots(results: Dict[str, Any], scores: Dict[str, Dict[str, float]], plots_dir: Path) -> None:
    """Render training curves, sequence scaling graphs, and multi-metric radar charts."""
    curves_dir = plots_dir / "training_curves"
    scaling_dir = plots_dir / "scaling"
    radar_dir = plots_dir / "radar_charts"

    curves_dir.mkdir(parents=True, exist_ok=True)
    scaling_dir.mkdir(parents=True, exist_ok=True)
    radar_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Loss Curves
    for model_name, data in results.items():
        glue_tasks = data.get("glue_tasks", {})
        for task_name, t_data in glue_tasks.items():
            loss_hist = t_data.get("loss_history", [])
            if loss_hist:
                plt.figure(figsize=(7, 4))
                plt.plot(loss_hist, label=f"{model_name} ({task_name})", color="#005CC5", linewidth=1.5)
                plt.title(f"Training Loss: {model_name} on {task_name.upper()}")
                plt.xlabel("Optimization Step")
                plt.ylabel("Loss")
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend()
                plt.tight_layout()
                plt.savefig(curves_dir / f"{model_name}_{task_name}_loss.png", dpi=150)
                plt.close()

    # 2. Sequence Length Scaling Plot
    plt.figure(figsize=(8, 5))
    for model_name, data in results.items():
        scaling = data.get("profiling", {}).get("scaling_curve", {}).get("by_seq_len", [])
        bs32_entries = [e for e in scaling if e.get("batch_size") == 32 and e.get("tokens_per_sec") is not None]
        if bs32_entries:
            seq_lens = [e["seq_len"] for e in bs32_entries]
            tps = [e["tokens_per_sec"] for e in bs32_entries]
            plt.plot(seq_lens, tps, marker="o", label=model_name, linewidth=1.8)

    plt.title("Inference Throughput vs Sequence Length (Batch Size = 32)")
    plt.xlabel("Sequence Length (Tokens)")
    plt.ylabel("Throughput (Tokens / Second)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scaling_dir / "throughput_scaling.png", dpi=150)
    plt.close()

    # 3. Radar Charts for Top Models
    top_models = sorted(scores.keys(), key=lambda k: scores[k].get("total_score", 0), reverse=True)[:5]
    if top_models:
        categories = ["Quality", "Efficiency", "Stability", "Robustness"]
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for m in top_models:
            s = scores[m]
            vals = [
                s.get("quality_score", 0),
                s.get("efficiency_score", 0),
                s.get("stability_score", 0),
                s.get("robustness_score", 0),
            ]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.8, label=f"{m} ({s.get('total_score', 0):.1f})")
            ax.fill(angles, vals, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, weight="bold")
        ax.set_ylim(0, 100)
        plt.title("Top Architecture Comparison Radar", size=13, y=1.08, weight="bold")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(radar_dir / "top_models_radar.png", dpi=150)
        plt.close()


def generate_all_reports(results_path: str, output_dir: str) -> None:
    """Master report generation entry point."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    scores = calculate_scores(results)
    csv_path = out_p / "summary.csv"
    md_path = out_p / "leaderboard.md"
    plots_dir = out_p / "plots"

    df = generate_summary_table(results, scores, str(csv_path))
    generate_leaderboard_markdown(df, str(md_path))
    generate_plots(results, scores, plots_dir)
    print(f"Reports successfully generated in {output_dir}")
