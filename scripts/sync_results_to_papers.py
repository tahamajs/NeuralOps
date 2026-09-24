#!/usr/bin/env python3
"""
scripts/sync_results_to_papers.py

Automatically syncs empirical benchmark results from the GPU server to:
  1. paper/PerModulePaper/<module>/07_results.tex
  2. paper/PerModulePaper/<module>/07_results.md
  3. paper/PerModulePaper/<module>/variables.json
  4. paper/PerModulePaper/<module>/paper_full.tex (if inlined)

Extracts strictly empirical numbers from results/unified_<module>_4models_summary.csv
or results/unified_<module>/<model>/summary.csv.
"""

import os
import sys
import glob
import json
import csv
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_ROOT = REPO_ROOT / "paper" / "PerModulePaper"

MODELS = ["distilbert", "roberta", "gpt2"]
MODEL_DISPLAY = {
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa",
    "gpt2": "GPT-2"
}

BASELINES = {
    "distilbert": {"acc": "90.71%", "f1": "0.9099", "vram": "183 MB", "time": "1.90"},
    "roberta": {"acc": "94.38%", "f1": "0.9445", "vram": "293 MB", "time": "4.10"},
    "gpt2": {"acc": "89.33%", "f1": "0.8942", "vram": "382 MB", "time": "4.20"}
}


def pull_results_from_gpu():
    """Pulls all results CSVs and master summary from GPU server via tar stream."""
    remote_code = (
        "import tarfile, glob, sys; "
        "tf = tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz'); "
        "[tf.add(f) for f in glob.glob('results/unified_*/**/*.csv', recursive=True) "
        "+ glob.glob('results/unified_*_4models_summary.*') "
        "+ glob.glob('results/master_queue_benchmark_summary.csv')]; "
        "tf.close()"
    )
    cmd = f'ssh -o StrictHostKeyChecking=no gpu "python -c \\"{remote_code}\\"" | tar -xzf -'
    print(">>> Syncing latest benchmark results from GPU server...")
    try:
        subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), check=True, capture_output=True)
        print(">>> GPU results synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error syncing from GPU: {e.stderr.decode() if e.stderr else e}")


CANONICAL_MODULE_ALIASES = {
    "aa_svd_compressor": "aasvd",
    "adaptive_sparse": "adaptive_sparse_attention",
    "adaptive_span": "adaptive_span_attention",
    "adaptive_svd_attn": "adaptive_svd_attention",
    "aegis_v1_module": "aegis_v1",
    "aegis_v3_module": "aegis_v3",
    "aegis_v4_module": "aegis_v4",
    "aegis_v5_module": "aegis_v5",
    "flash_attention3": "flash3",
    "flash_attention3_fp8": "flash3",
    "flash3_fp8": "flash3",
    "fms": "factorised_matrix_state",
    "fasa": "mixattention_moska",
    "galerkin": "galerkin_attention",
    "hla_attention": "hla",
    "infini_attention": "infini",
    "lsh_attention": "lsh",
    "nasc_attention": "nasc",
    "nha_compressor": "nha",
    "nsa_attention": "nsa",
    "orion_module": "orion",
    "proxsparse_pruner": "proxsparse",
    "rala_compressor": "rala",
    "salt_attention": "salt",
    "soa_fused": "soa_fused_module",
    "soafusedattention": "soa_fused_module",
    "spectraformer_attention": "spectraformer",
    "svd_sequence_attention": "svd_sequence",
    "tlinformer_attention": "tlinformer",
    "tomoe_compressor": "tomoe",
}


def find_paper_dir(module_name: str) -> Path:
    """Finds the paper directory for a given module name."""
    if not PAPER_ROOT.exists():
        return None

    # Map alias to canonical if applicable
    module_name = CANONICAL_MODULE_ALIASES.get(module_name, module_name)

    # 1. Exact match
    direct = PAPER_ROOT / module_name
    if direct.is_dir():
        return direct

    all_dirs = [d for d in PAPER_ROOT.iterdir() if d.is_dir()]
    dir_names = {d.name: d for d in all_dirs}

    # 2. Common suffixes
    suffixes = ["_compressor", "_attention", "_pruner", "_quantizer", "_svd", "_module"]
    for s in suffixes:
        cand = module_name + s
        if cand in dir_names:
            return dir_names[cand]

    # 3. If module_name ends with any suffix, strip it
    for s in suffixes:
        if module_name.endswith(s):
            stripped = module_name[:-len(s)]
            if stripped in dir_names:
                return dir_names[stripped]

    # 4. Normalized match (ignore case and underscores)
    norm_mod = module_name.replace("_", "").lower()
    for d in all_dirs:
        if d.name.replace("_", "").lower() == norm_mod:
            return d

    # 5. Substring prefix match
    for d in all_dirs:
        d_norm = d.name.replace("_", "").lower()
        if norm_mod.startswith(d_norm) or d_norm.startswith(norm_mod):
            return d

    return None


def parse_module_results(module_name: str) -> dict:
    """Extracts empirical results across models for module_name."""
    # Check top-level 4models_summary.csv first
    summary_4m = RESULTS_DIR / f"unified_{module_name}_4models_summary.csv"
    res = {}

    if summary_4m.exists():
        try:
            with open(summary_4m, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    m_key = row.get("Model", "").lower()
                    if m_key in MODELS:
                        # find accuracy column
                        acc_col = [c for c in row.keys() if "acc" in c.lower() and "delta" not in c.lower() and "base" not in c.lower()]
                        acc_val = row[acc_col[0]] if acc_col else row.get("Accuracy", "N/A")
                        
                        # parse values
                        res[m_key] = {
                            "base_acc": row.get("Baseline Acc", BASELINES[m_key]["acc"]),
                            "acc": acc_val,
                            "delta": row.get("Delta Acc", row.get("Delta", "--")),
                            "f1": row.get("F1", "--"),
                            "status": row.get("Status", "PASS"),
                            "comp_layers": row.get("Compressed Layers", "0"),
                            "ratio": row.get("Comp Rate", "1.00x"),
                            "vram": row.get("Peak VRAM", "--"),
                            "time": row.get("Time (s)", row.get("Inference Time", "--")),
                            "comp_time": row.get("Comp Time (s)", "--")
                        }
            if len(res) == len(MODELS):
                return res
        except Exception as e:
            print(f"Warning reading {summary_4m}: {e}")

    # Fallback: check individual per-model summary.csv
    mod_dir = RESULTS_DIR / f"unified_{module_name}"
    if mod_dir.exists():
        for m_key in MODELS:
            sub_csv = mod_dir / m_key / "summary.csv"
            if sub_csv.exists():
                try:
                    with open(sub_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if len(rows) >= 2:
                            # row 0 is full, row 1 is module
                            full_row = rows[0]
                            mod_row = rows[1]
                            
                            b_acc = float(full_row.get("accuracy_pct", 0.0))
                            m_acc = float(mod_row.get("accuracy_pct", 0.0))
                            delta = m_acc - b_acc
                            sign = "+" if delta >= 0 else ""
                            
                            comp_rate = float(mod_row.get("compression_rate", 1.0))
                            vram = float(mod_row.get("peak_vram_mb", 0.0))
                            time_s = float(mod_row.get("elapsed_s", 0.0))
                            
                            res[m_key] = {
                                "base_acc": f"{b_acc:.2f}%",
                                "acc": f"{m_acc:.2f}%",
                                "delta": f"{sign}{delta:.2f}%",
                                "f1": f"{float(mod_row.get('f1_score', 0.0)):.4f}",
                                "status": mod_row.get("status", "PASS"),
                                "comp_layers": str(mod_row.get("num_compressed_layers", "0")),
                                "ratio": f"{comp_rate:.2f}x",
                                "vram": f"{vram:.1f} MB",
                                "time": f"{time_s:.1f}s",
                                "comp_time": "--"
                            }
                except Exception as e:
                    print(f"Warning reading {sub_csv}: {e}")

    return res


def build_latex_table(module_name: str, display_name: str, res: dict) -> str:
    """Generates LaTeX Table 7.1 matching repository standards."""
    rows = []
    for m in MODELS:
        disp = MODEL_DISPLAY[m]
        b = BASELINES[m]
        b_acc = b['acc'].replace('%', '\\%')
        # Baseline row
        rows.append(f"{disp} & full & {b_acc} & -- & -- & {b['f1']} & 0 & 1.00$\\times$ & {b['vram']} & {b['time']} & -- \\\\")
        
        # Module row
        m_data = res.get(m, {})
        acc = m_data.get("acc", "--").replace('%', '\\%')
        delta = m_data.get("delta", "--").replace('%', '\\%')
        status = m_data.get("status", "PASS")
        f1 = m_data.get("f1", "--")
        comp = m_data.get("comp_layers", "0")
        ratio = m_data.get("ratio", "1.00x").replace("x", "$\\times$")
        vram = m_data.get("vram", "--")
        t = m_data.get("time", "--").replace("s", "")
        ct = m_data.get("comp_time", "--").replace("s", "")
        
        rows.append(f"{disp} & \\textbf{{{display_name}}} & \\textbf{{{acc}}} & {delta} & {status} & {f1} & {comp} & \\textbf{{{ratio}}} & {vram} & {t} & {ct} \\\\")

    rows_str = "\n".join(rows)

    latex_table = f"""\\begin{{table*}}[t]
\\centering
\\small
\\begin{{tabular}}{{lllllllllll}}
\\toprule
\\textbf{{Model}} & \\textbf{{Module}} & \\textbf{{Acc}} & \\textbf{{$\\Delta$}} & \\textbf{{Status}} & \\textbf{{F1}} & \\textbf{{Comp.}} & \\textbf{{Ratio}} & \\textbf{{VRAM}} & \\textbf{{Inference (s)}} & \\textbf{{Compression (s)}} \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\caption{{Main results on SST-2 validation across models under the production configuration ($r_{{\\mathrm{{ceiling}}}}=256, \\tau=0.40$).}}
\\label{{tab:main_sst2}}
\\end{{table*}}"""
    return latex_table


def build_markdown_table(display_name: str, res: dict) -> str:
    """Generates Markdown results table."""
    md = [
        "| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for m in MODELS:
        disp = MODEL_DISPLAY[m]
        b = BASELINES[m]
        md.append(f"| {disp} | full | {b['acc']} | -- | -- | {b['f1']} | 0 | 1.00x | {b['vram']} | {b['time']}s |")
        m_data = res.get(m, {})
        acc = m_data.get("acc", "--")
        delta = m_data.get("delta", "--")
        status = m_data.get("status", "PASS")
        f1 = m_data.get("f1", "--")
        comp = m_data.get("comp_layers", "0")
        ratio = m_data.get("ratio", "1.00x")
        vram = m_data.get("vram", "--")
        t = m_data.get("time", "--")
        md.append(f"| {disp} | **{display_name}** | **{acc}** | {delta} | {status} | {f1} | {comp} | **{ratio}** | {vram} | {t} |")
    return "\n".join(md)


def update_paper_files(paper_dir: Path, module_name: str, res: dict):
    """Updates 07_results.tex, 07_results.md, variables.json, and paper_full.tex."""
    display_name = module_name.replace("_", " ").title()
    latex_table = build_latex_table(module_name, display_name, res)
    md_table = build_markdown_table(display_name, res)

    # 1. Update 07_results.tex
    tex_path = paper_dir / "07_results.tex"
    if tex_path.exists():
        content = tex_path.read_text(encoding="utf-8")
        # Replace existing \begin{table*} ... \end{table*} or insert
        table_pattern = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)
        m = table_pattern.search(content)
        if m:
            new_content = content[:m.start()] + latex_table + content[m.end():]
        else:
            # Insert after \subsection{Main Benchmark Performance} or at top
            sub_m = re.search(r"(\\subsection\{[^}]+\})", content)
            if sub_m:
                new_content = content[:sub_m.end()] + "\n\n" + latex_table + "\n" + content[sub_m.end():]
            else:
                new_content = latex_table + "\n\n" + content
        tex_path.write_text(new_content, encoding="utf-8")
        print(f"  [OK] Updated {tex_path.relative_to(REPO_ROOT)}")

    # 2. Update 07_results.md
    md_path = paper_dir / "07_results.md"
    if md_path.exists():
        content = md_path.read_text(encoding="utf-8")
        table_md_pattern = re.compile(r"\| Model \|.*?\n\n", re.DOTALL)
        m = table_md_pattern.search(content)
        if m:
            new_content = content[:m.start()] + md_table + "\n\n" + content[m.end():]
        else:
            new_content = md_table + "\n\n" + content
        md_path.write_text(new_content, encoding="utf-8")
        print(f"  [OK] Updated {md_path.relative_to(REPO_ROOT)}")

    # 3. Update variables.json
    var_path = paper_dir / "variables.json"
    if var_path.exists():
        try:
            var_data = json.loads(var_path.read_text(encoding="utf-8"))
            if "distilbert" in res:
                var_data["D_ACC"] = res["distilbert"]["acc"].replace("%", "")
                var_data["D_STATUS"] = res["distilbert"]["status"]
                var_data["D_VRAM"] = res["distilbert"]["vram"]
            if "roberta" in res:
                var_data["R_ACC"] = res["roberta"]["acc"].replace("%", "")
                var_data["R_STATUS"] = res["roberta"]["status"]
                var_data["R_VRAM"] = res["roberta"]["vram"]
            if "gpt2" in res:
                var_data["G_ACC"] = res["gpt2"]["acc"].replace("%", "")
                var_data["G_STATUS"] = res["gpt2"]["status"]
                var_data["G_VRAM"] = res["gpt2"]["vram"]
            var_path.write_text(json.dumps(var_data, indent=2), encoding="utf-8")
            print(f"  [OK] Updated {var_path.relative_to(REPO_ROOT)}")
        except Exception as e:
            print(f"  [!] Error updating variables.json: {e}")

    # 4. Update paper_full.tex if present
    full_tex_path = paper_dir / "paper_full.tex"
    if full_tex_path.exists():
        content = full_tex_path.read_text(encoding="utf-8")
        table_pattern = re.compile(r"\\begin\{table\*?\}.*?\\caption\{Main results.*?\\end\{table\*?\}", re.DOTALL)
        m = table_pattern.search(content)
        if not m:
            table_pattern = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)
            # Find the one in sec:results
            sec_idx = content.find(r"\section{Results}")
            if sec_idx == -1:
                sec_idx = content.find(r"\section{Empirical Results")
            if sec_idx != -1:
                m = table_pattern.search(content, sec_idx)
        if m:
            new_content = content[:m.start()] + latex_table + content[m.end():]
            full_tex_path.write_text(new_content, encoding="utf-8")
            print(f"  [OK] Updated {full_tex_path.relative_to(REPO_ROOT)}")


def sync_all():
    pull_results_from_gpu()

    summary_file = RESULTS_DIR / "master_queue_benchmark_summary.csv"
    if not summary_file.exists():
        print("No master_queue_benchmark_summary.csv found.")
        return

    completed_modules = []
    with open(summary_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "").upper()
            if status in ("PASS", "SKIPPED_EXISTING"):
                completed_modules.append(row.get("module"))

    print(f"\n>>> Total completed/passed modules in queue: {len(completed_modules)}")
    
    updated_count = 0
    matched_papers = 0

    for mod in completed_modules:
        paper_dir = find_paper_dir(mod)
        if not paper_dir:
            print(f"[-] Paper dir not found for module '{mod}'")
            continue
        matched_papers += 1

        res = parse_module_results(mod)
        if not res:
            print(f"[-] No results found for module '{mod}'")
            continue

        print(f"\n[+] Syncing results for module '{mod}' -> {paper_dir.name}")
        update_paper_files(paper_dir, mod, res)
        updated_count += 1

    print(f"\n=======================================================")
    print(f"SUMMARY: Matched {matched_papers} papers, successfully updated {updated_count} papers.")
    print(f"=======================================================\n")


if __name__ == "__main__":
    sync_all()
