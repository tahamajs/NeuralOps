"""Execute and verify the complete 20-Notebook Suite (00 to 19) in-place on remote GPU.

Author: Taha Majlesi
"""
import glob
import json
import os
import subprocess
import sys
import time

NOTEBOOKS = [
    "00_master_neuralops_complete_system_walkthrough.ipynb",
    "01_quickstart_surgery_and_healing.ipynb",
    "02_roofline_and_triton_kernels.ipynb",
    "03_custom_plugins_and_extensibility.ipynb",
    "04_end_to_end_3level_benchmarking.ipynb",
    "05_master_108_cuda_and_26_triton_kernels_benchmark.ipynb",
    "05_paper_empirical_results_master.ipynb",
    "06_master_all_131_attention_mechanisms_interactive.ipynb",
    "07_master_all_72_compression_engines_quant_and_svd.ipynb",
    "08_six_pillar_distillation_and_surgical_recovery.ipynb",
    "09_production_serving_paged_cache_and_speculative_decoding.ipynb",
    "10_long_context_extrapolation_32k.ipynb",
    "11_cuda_vs_triton_vs_pytorch_matrix.ipynb",
    "12_svd_whitening_spectral_allocation.ipynb",
    "13_quantization_pareto_fp8_int4_kmeans.ipynb",
    "14_structured_pruning_token_shedding.ipynb",
    "15_downstream_glue_lra_evaluation.ipynb",
    "16_roofline_ada_rtx4070_saturation.ipynb",
    "17_speculative_paged_kv_throughput.ipynb",
    "18_cross_seed_statistical_variance.ipynb",
    "19_publication_figures_latex_generator.ipynb",
]

NB_DIR = "notebooks"
JUPYTER_BIN = "/mnt/d/venv/bin/jupyter"
PYTHON_BIN = "/mnt/d/venv/bin/python3"


def audit_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    src0 = "".join(cells[0].get("source", [])) if cells else ""
    has_author = "Taha Majlesi" in src0
    executed_code = 0
    errors = []
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code":
            if c.get("execution_count") is not None:
                executed_code += 1
            for out in c.get("outputs", []):
                if out.get("output_type") == "error":
                    errors.append((i, out.get("ename"), out.get("evalue"), out.get("traceback")))
    return len(cells), has_author, executed_code, errors


def execute_notebook(nb_name):
    path = os.path.join(NB_DIR, nb_name)
    print(f"\n{'='*75}\nExecuting: {nb_name} on GPU (Ada Lovelace RTX 4070)\n{'='*75}")
    t0 = time.time()
    cmd = [
        JUPYTER_BIN,
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"FAILED (returncode={proc.returncode}, took {dt:.1f}s)")
        print("STDERR:\n", proc.stderr[-2000:])
        return False, dt, proc.stderr
    cell_count, has_author, exec_code, errors = audit_notebook(path)
    if errors:
        print(f"FAILED with {len(errors)} cell error(s):")
        for cell_idx, ename, evalue, tb in errors:
            print(f"  Cell {cell_idx}: {ename}: {evalue}")
            if tb:
                print("\n".join(tb[-10:]))
        return False, dt, str(errors)
    print(
        f"SUCCESS: {nb_name} | {cell_count} cells | Author: {has_author} | ExecCode: {exec_code} | Duration: {dt:.1f}s"
    )
    return True, dt, ""


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    to_run = [target] if target else NOTEBOOKS
    results = {}
    failed = []

    print(f"Beginning execution of {len(to_run)} notebooks on remote GPU...")
    for nb in to_run:
        ok, dt, err = execute_notebook(nb)
        results[nb] = (ok, dt)
        if not ok:
            failed.append(nb)
            print(f"Aborting batch on failure of {nb} for closed-loop debugging.")
            break

    print(f"\n{'='*75}\nEXECUTION SUMMARY:\n{'='*75}")
    for nb, (ok, dt) in results.items():
        status = "PASS [100% OK]" if ok else "FAIL [ERROR]"
        print(f"{nb:60s} | {status} | {dt:5.1f}s")

    if failed:
        sys.exit(1)
    print(f"\nAll {len(results)} evaluated notebooks PASSED with 0 errors!")


if __name__ == "__main__":
    main()
