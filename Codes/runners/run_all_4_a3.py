import subprocess
import os
import sys
import time
import json
import pandas as pd

os.chdir(r"D:\svd_project")

models = [
    {
        "name": "DistilBERT",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\distilbert_sst2 "
            r"--model_family bert "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\distilbert "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 50 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\distilbert\summary.csv",
    },
    {
        "name": "RoBERTa",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\roberta_sst2 "
            r"--model_family roberta "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\roberta "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 50 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--parent_checkpoint results\roberta_sst2_lora_baseline\finetuned_model.pt "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\roberta\summary.csv",
    },
    {
        "name": "GPT-2",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\gpt2_sst2 "
            r"--model_family gpt "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\gpt2 "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 50 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--parent_checkpoint results\gpt2_sst2_lora_baseline\finetuned_model.pt "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\gpt2\summary.csv",
    },
]

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL A3 COMPRESSOR GPU BENCHMARK SWEEP", flush=True)
print("Standard Config: module=a3, dataset=sst2, recovery_steps=100, batch_size=8, precision=fp16", flush=True)
print("=" * 80, flush=True)

results_summary = []

for m in models:
    print(f"\n==================================================================", flush=True)
    print(f">>> [RUNNING MODEL] {m['name']}", flush=True)
    print(f"==================================================================", flush=True)
    t0 = time.time()
    res = subprocess.run(m["cmd"], shell=True)
    elapsed = time.time() - t0
    print(f"\n>>> [FINISHED] {m['name']} in {elapsed:.1f}s (Exit code: {res.returncode})", flush=True)
    
    # Read summary
    status = "ERROR"
    acc = 0.0
    threshold = 56.0
    if os.path.exists(m["summary"]):
        try:
            df = pd.read_csv(m["summary"])
            row = df[df["module_name"] == "a3"]
            if not row.empty:
                status = row["status"].values[0]
                acc = float(row["accuracy_pct"].values[0])
                if "threshold_pct" in row:
                    threshold = float(row["threshold_pct"].values[0])
        except Exception as e:
            print(f"Error parsing {m['summary']}: {e}", flush=True)
            
    results_summary.append({
        "Model": m["name"],
        "Module": "a3",
        "Status": status,
        "Accuracy (%)": acc,
        "Threshold (%)": threshold,
        "Elapsed (s)": round(elapsed, 1),
    })

print("\n" + "=" * 80, flush=True)
print("FINAL 4-MODEL A3 GPU BENCHMARK RESULTS TABLE", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(results_summary)
print(res_df.to_markdown(index=False) if hasattr(res_df, "to_markdown") else res_df.to_string(index=False), flush=True)

# Save consolidated results to JSON and CSV
out_csv = r"D:\SVD_Project\results\unified_a3_4models_summary.csv"
out_json = r"D:\SVD_Project\results\unified_a3_4models_summary.json"
res_df.to_csv(out_csv, index=False)
with open(out_json, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved consolidated summaries to:\n- {out_csv}\n- {out_json}", flush=True)

# Generate and save comparative plots in results directory
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _mod_slug = res_df.columns[2].replace(" Acc", "").lower() if len(res_df.columns) > 2 else "module"
    _plots_dir = out_dir / f"unified_{_mod_slug}"
    _plots_dir.mkdir(parents=True, exist_ok=True)

    _p_models = [str(s.get("Model", "")) for s in final_stats]
    _b_accs = [float(str(s.get("Baseline Acc", "0")).replace("%", "")) if str(s.get("Baseline Acc", "N/A")) != "N/A" else 0.0 for s in final_stats]
    _m_acc_col = [k for k in final_stats[0].keys() if "Acc" in k and k not in ("Baseline Acc", "Delta Acc")][0] if final_stats else "Acc"
    _m_accs = [float(str(s.get(_m_acc_col, "0")).replace("%", "")) if str(s.get(_m_acc_col, "N/A")) != "N/A" else 0.0 for s in final_stats]
    _c_rates = [float(str(s.get("Comp Rate", "1.0")).replace("x", "")) if str(s.get("Comp Rate", "N/A")) != "N/A" else 1.0 for s in final_stats]

    # Chart 1: Accuracy Comparison
    _x = np.arange(len(_p_models))
    _w = 0.35
    _fig, _ax1 = plt.subplots(figsize=(9, 5))
    _r1 = _ax1.bar(_x - _w/2, _b_accs, _w, label='Baseline Acc', color='#4C72B0')
    _r2 = _ax1.bar(_x + _w/2, _m_accs, _w, label=f'{_mod_slug.upper()} Acc', color='#55A868')
    _ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    _ax1.set_title(f'Unified 4-Model Quality Evaluation ({_mod_slug.upper()})', fontweight='bold')
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_p_models, fontweight='bold')
    _ax1.set_ylim(0, 105)
    _ax1.legend(loc='upper right')
    _ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for _rect in _r1:
        _h = _rect.get_height()
        if _h > 0: _ax1.annotate(f'{_h:.1f}%', xy=(_rect.get_x() + _rect.get_width()/2, _h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for _rect in _r2:
        _h = _rect.get_height()
        if _h > 0: _ax1.annotate(f'{_h:.1f}%', xy=(_rect.get_x() + _rect.get_width()/2, _h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(_plots_dir / f"{_mod_slug}_4models_accuracy_comparison.png", dpi=300)
    plt.close()

    # Chart 2: Pareto Frontier
    _fig, _ax2 = plt.subplots(figsize=(8, 5))
    for _i, _m_name in enumerate(_p_models):
        if _m_accs[_i] > 0:
            _ax2.scatter(_c_rates[_i], _m_accs[_i], s=160, label=_m_name, zorder=5)
            _ax2.annotate(f"{_m_name} ({_c_rates[_i]:.2f}x, {_m_accs[_i]:.1f}%)", (_c_rates[_i], _m_accs[_i]), xytext=(7, -4), textcoords='offset points', fontsize=9)
    _ax2.set_xlabel('Compression Rate (x)', fontweight='bold')
    _ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    _ax2.set_title(f'Pareto Frontier: Accuracy vs Compression ({_mod_slug.upper()})', fontweight='bold')
    _ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(_plots_dir / f"{_mod_slug}_pareto_accuracy_vs_compression.png", dpi=300)
    plt.close()
    print(f"Generated and saved comparative visualization charts to {_plots_dir}", flush=True)
except Exception as _e_p:
    print(f"[WARN] Could not generate comparative plots: {_e_p}", flush=True)

