# Post-Run Audit Report

**Audit Timestamp**: 2026-09-21T01:28:00+03:30  
**Host Audited**: Local (`mac`) & Remote GPU Server `hadi-windows` (`D:\svd_project`)  
**Status**: COMPLETE STOP ENFORCED. No processes running, no code modifications made.

---

## 1. Running Processes Audit

As inspected via `Get-CimInstance Win32_Process` on `hadi-windows`:

| ProcessId | Process Name | Command Line | Status |
| :--- | :--- | :--- | :--- |
| `14576` | `python.exe` | `C:\Users\Hadi Schumacher\anaconda3\Scripts\nvitop.exe` | Background monitor |
| `10504` | `python.exe` | `C:\Users\Hadi Schumacher\anaconda3\Scripts\nvitop.exe` | Background monitor |

- **No orchestrator process is running.** `auto_pipeline_orchestrator.py` and `run_full_pipeline.py` exited completely.
- **No sweep worker is running.** No instances of `run_quality_sweep.py` or `run_one_module_isolated.py` are active.
- **GPU utilization**: 0%, completely idle.

---

## 2. File Verification & Timestamps

### Master Summary Tables:
```
-rw-r--r--@ 1 tahamajs staff 24340 Sep 20 21:08 results/distilbert_mnli_full_mnli_sweep/summary_v3.csv
-rw-r--r--  1 tahamajs staff 36728 Sep 20 23:14 results/distilbert_mnli_full_mnli_sweep/summary_v4.csv
```
- **`summary_v4.csv` timestamp**: `Sep 20 23:14` (strictly AFTER `summary_v3.csv` at `Sep 20 21:08`).
- **Line counts**:
  - `summary_v3.csv`: **203 lines** (header + 202 module rows)
  - `summary_v4.csv`: **204 lines** (header + 203 module rows, including `hgrn2_outer`)

---

## 3. Primal Result Verification

Latest JSON output on remote disk:
- **File**: `D:\svd_project\logs\isolated_results\primal_1789927154.json`
- **File Write Timestamp**: `9/20/2026 9:54:26 PM` (local machine time: 2026-09-20 21:54:26)
- **JSON Content**:
  ```json
  "accuracy": 81.9,
  "accuracy_pct": 81.9,
  "f1": 0.8182,
  "elapsed_s": 1491.45,
  "finished_at": "2026-09-20T18:24:26.314745"
  ```
- **Verification**: The result is **FRESH and REAL**, executed during the 100-step recovery run. It is not stale.

---

## 4. Threshold Implementation Verification

In [`Codes/framework/run_quality_sweep.py`](Codes/framework/run_quality_sweep.py#L2591-L2640):
- Threshold is **dynamically computed** from the validation split labels:
  ```python
  p0 = majority_count / total_val_samples  # 341 / 1000 = 34.10% for MNLI local
  se = math.sqrt(p0 * (1.0 - p0) / total_val_samples)  # 1.499%
  pass_threshold_pct = round(majority_baseline_pct + (3.0 * se * 100.0), 2)  # 38.60%
  ```
- No hardcoded numbers exist in the sweep script; the values `34.10%`, `1.499%`, and `38.60%` are derived dynamically at runtime.
- For Full-MNLI ($N=9,815$): $p_0 = 35.45\%$, $\text{SE} = 0.483\%$, yielding the threshold **36.89%** documented in `summary_full_mnli.csv`.

---

## 5. Current vs. Stale Files & Overwrite Audit

1. **Current and Valid**:
   - `results/distilbert_mnli_full_mnli_sweep/summary_v4.csv` (12 columns, 203 modules, 156 PASS)
   - `results/distilbert_mnli_full_mnli_sweep/summary_full_mnli.csv` (132 PASS modules evaluated on $N=9,815$)
   - `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group7.csv`
   - `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group2.csv`
   - `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group6.csv`
   - `results/distilbert_mnli_full_mnli_sweep/summary_snapshot_post_group4.csv`
   - `paper/_FINAL_STATE.md`
2. **Historical / Preceding Snapshots (Superseded, not overwritten)**:
   - `summary_v3.csv` remains preserved and intact (203 lines).
   - `summary_v2.csv` remains preserved and intact.
3. **No overwrites occurred**:
   - Every stage saved to a unique snapshot prior to advancing.
   - All code files remained untouched during this check.
