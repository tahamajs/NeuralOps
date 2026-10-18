"""
Comprehensive Synchronization of 130 Canonical Attention Modules
Across README, ICLR Paper, Master Thesis, and Verification Scripts.
"""
import re
from pathlib import Path

def sync_all(repo_root: Path):
    # =========================================================================
    # 1. README.md
    # =========================================================================
    readme_path = repo_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()

        replacements = [
            ("131 canonical attention mechanisms", "130 canonical attention mechanisms"),
            ("203 out of 203 modules passed with 100% operational certification** (131/131 Attention Mechanisms", "202 out of 202 modules passed with 100% operational certification** (130/130 Attention Mechanisms"),
            ("- **131 Canonical Attention Mechanisms**:", "- **130 Canonical Attention Mechanisms**:"),
            ("(131 Attention + 72 Compression)", "(130 Attention + 72 Compression)"),
            ("(131 Canonical Modules)", "(130 Canonical Modules)"),
            ("for 131 attentions & 72 compressors", "for 130 attentions & 72 compressors"),
            ("across all 131 canonical subquadratic attention mechanisms.", "across all 130 canonical subquadratic attention mechanisms."),
            ("| **Attention Mechanisms (`ATTENTION_REGISTRY`)** | **131** | **131** | 0 | **100.0% PASS** |", "| **Attention Mechanisms (`ATTENTION_REGISTRY`)** | **130** | **130** | 0 | **100.0% PASS** |"),
            ("All 131 attention modules pass forward execution", "All 130 attention modules pass forward execution"),
            ("Transplant any of the 131 subquadratic attention mechanisms", "Transplant any of the 130 subquadratic attention mechanisms"),
            ("Counts (131 Attentions, 72 Compressors, 26 Triton Kernels)", "Counts (130 Attentions, 72 Compressors, 26 Triton Kernels)"),
            ("AttnZoo[131 Canonical Attention Modules ATTENTION_REGISTRY]", "AttnZoo[130 Canonical Attention Modules ATTENTION_REGISTRY]"),
            ("NeuralOps categorizes all 131 attention mechanisms across", "NeuralOps categorizes all 130 attention mechanisms across"),
            ("131 CANONICAL ATTENTION MECHANISMS TAXONOMY", "130 CANONICAL ATTENTION MECHANISMS TAXONOMY"),
            ("### Complete 131-Module Forensic Hardware Audit Table", "### Complete 130-Module Forensic Hardware Audit Table"),
            ("| All 131 Subquadratic Attention Mechanisms | **1,048 / 1,048** |", "| All 130 Subquadratic Attention Mechanisms | **1,040 / 1,040** |"),
            ("Run end-to-end 131 attention mechanisms evaluation", "Run end-to-end 130 attention mechanisms evaluation"),
            ("invariant (131/72/26)", "invariant (130/72/26)"),
            ("| 131 Attention + 72 Compression Capabilities |", "| 130 Attention + 72 Compression Capabilities |"),
            ("all 131 attention mechanisms across", "all 130 attention mechanisms across"),
            ("any of the 131 architectures", "any of the 130 architectures"),
            ("- **131 Attention Mechanisms**", "- **130 Attention Mechanisms**"),
        ]

        for old, new in replacements:
            if old in readme:
                readme = readme.replace(old, new)

        # Process table rows
        lines = readme.splitlines()
        new_lines = []
        in_table = False
        idx_counter = 1

        for line in lines:
            if line.startswith("| 1 | `adaptive_span_attention`"):
                in_table = True

            if in_table and "adasplash_attention" in line:
                continue

            if in_table and line.startswith("| ") and not line.startswith("| #") and not line.startswith("| :-"):
                m = re.match(r"^\|\s*(\d+)\s*\|\s*(.*)$", line)
                if m:
                    orig_num = int(m.group(1))
                    line = f"| {idx_counter} | {m.group(2)}"
                    idx_counter += 1
                    if orig_num == 131:
                        in_table = False
            new_lines.append(line)

        new_readme = "\n".join(new_lines) + "\n"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_readme)
        print(f"[OK] README.md updated: table has {idx_counter - 1} rows.")

    # =========================================================================
    # 2. ICLR 2027 Submission LaTeX Files
    # =========================================================================
    iclr_conf = repo_root / "paper" / "iclr2027-submission" / "iclr2027_conference.tex"
    if iclr_conf.exists():
        with open(iclr_conf, "r", encoding="utf-8") as f:
            tex = f.read()

        iclr_replacements = [
            ("Study of 131 Attention Mechanisms", "Study of 130 Attention Mechanisms"),
            ("131 canonical attention mechanisms", "130 canonical attention mechanisms"),
            ("131 Canonical Attention Mechanisms", "130 Canonical Attention Mechanisms"),
            ("all 131 canonical attention mechanisms", "all 130 canonical attention mechanisms"),
            ("our 131 attention mechanisms", "our 130 attention mechanisms"),
            ("evaluating 131 attention mechanisms", "evaluating 130 attention mechanisms"),
            ("All 131 Canonical Attention Mechanisms", "All 130 Canonical Attention Mechanisms"),
            ("all 131 canonical attention mechanisms", "all 130 canonical attention mechanisms"),
            (r"\texttt{adaptive\_span}, \texttt{adasplash},", r"\texttt{adaptive\_span},"),
        ]
        for old, new in iclr_replacements:
            if old in tex:
                tex = tex.replace(old, new)

        with open(iclr_conf, "w", encoding="utf-8") as f:
            f.write(tex)
        print("[OK] paper/iclr2027-submission/iclr2027_conference.tex updated.")

    table_131_tex = repo_root / "paper" / "iclr2027-submission" / "tables" / "table_all_131_attentions.tex"
    if table_131_tex.exists():
        with open(table_131_tex, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace("All 131 Canonical Attention Mechanisms", "All 130 Canonical Attention Mechanisms")

        lines = content.splitlines()
        new_lines = []
        idx = 1
        for line in lines:
            if r"\texttt{adasplash_attention}" in line:
                continue
            m = re.match(r"^(\d+)\s*&\s*(\\texttt\{.*)$", line)
            if m:
                line = f"{idx} & {m.group(2)}"
                idx += 1
            new_lines.append(line)

        with open(table_131_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[OK] table_all_131_attentions.tex updated: total entries = {idx - 1}.")

    table_master_tex = repo_root / "paper" / "iclr2027-submission" / "tables" / "table_master_benchmark_matrix.tex"
    if table_master_tex.exists():
        with open(table_master_tex, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()
        new_lines = [l for l in lines if not any(x in l for x in [r"\texttt{adasplash}", r"\texttt{adasplash_attention}", r"\texttt{adasplash_inject}"])]
        with open(table_master_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print("[OK] table_master_benchmark_matrix.tex updated (removed adasplash rows).")

    # =========================================================================
    # 3. Thesis Final LaTeX Files
    # =========================================================================
    thesis_main = repo_root / "thesis_final" / "master_thesis.tex"
    if thesis_main.exists():
        with open(thesis_main, "r", encoding="utf-8") as f:
            tex = f.read()
        tex = tex.replace("کالبدشکافی ۱۳۱ مکانیزم توجه متمایز", "کالبدشکافی ۱۳۰ مکانیزم توجه متمایز")
        tex = tex.replace("131 Attention Mechanisms", "130 Attention Mechanisms")
        tex = tex.replace("تمامی این ۱۳۱ ماژول در جدول", "تمامی این ۱۳۰ ماژول در جدول")
        with open(thesis_main, "w", encoding="utf-8") as f:
            f.write(tex)
        print("[OK] thesis_final/master_thesis.tex updated.")

    chapter_attns = repo_root / "thesis_final" / "chapter_attentions.tex"
    if chapter_attns.exists():
        with open(chapter_attns, "r", encoding="utf-8") as f:
            tex = f.read()
        # Comment out adasplash_attention inclusion
        tex = tex.replace(
            r"\input{attentions_latex/individual_modules/adasplash_attention.tex}",
            r"% \input{attentions_latex/individual_modules/adasplash_attention.tex} % PURGED: O(N^2) memory leak"
        )
        with open(chapter_attns, "w", encoding="utf-8") as f:
            f.write(tex)
        print("[OK] thesis_final/chapter_attentions.tex updated (commented out adasplash_attention).")

    app_all = repo_root / "thesis_final" / "chapter_all_modules_appendix.tex"
    if app_all.exists():
        with open(app_all, "r", encoding="utf-8") as f:
            tex = f.read()
        tex = tex.replace("تحلیلی ۱۳۱ مکانیزم متعارف توجه", "تحلیلی ۱۳۰ مکانیزم متعارف توجه")

        lines = tex.splitlines()
        new_lines = []
        idx = 1
        in_table = False
        for line in lines:
            if line.startswith(r"\begin{longtable}"):
                in_table = True
            if in_table and line.startswith(r"\end{longtable}"):
                in_table = False

            if in_table and r"\texttt{adasplash_attention}" in line:
                continue

            if in_table:
                m = re.match(r"^(\d+)\s*&\s*(\\texttt\{.*)$", line)
                if m:
                    line = f"{idx} & {m.group(2)}"
                    idx += 1
            new_lines.append(line)

        with open(app_all, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[OK] thesis_final/chapter_all_modules_appendix.tex updated: total entries = {idx - 1}.")

    app_emp = repo_root / "thesis_final" / "app_131_attentions_empirical.tex"
    if app_emp.exists():
        with open(app_emp, "r", encoding="utf-8") as f:
            tex = f.read()

        lines = tex.splitlines()
        new_lines = []
        idx = 1
        for line in lines:
            if r"\texttt{adasplash_attention}" in line:
                continue
            m = re.match(r"^(\d+)\s*&\s*(\\texttt\{.*)$", line)
            if m:
                line = f"{idx} & {m.group(2)}"
                idx += 1
            new_lines.append(line)

        with open(app_emp, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[OK] thesis_final/app_131_attentions_empirical.tex updated: total entries = {idx - 1}.")

    # =========================================================================
    # 4. scripts/utils/verify_readme_counts.py
    # =========================================================================
    verify_script = repo_root / "scripts" / "utils" / "verify_readme_counts.py"
    if verify_script.exists():
        with open(verify_script, "r", encoding="utf-8") as f:
            script_code = f.read()

        # Ensure validation strictly enforces 130 canonical attentions
        if "if actual_attn != 130:" not in script_code:
            script_code = script_code.replace(
                "actual_attn = len(ATTENTION_REGISTRY)",
                "actual_attn = len(ATTENTION_REGISTRY)\nif actual_attn != 130:\n    errors.append(f'Canonical attention count mismatch: expected 130, got {actual_attn}')"
            )
        with open(verify_script, "w", encoding="utf-8") as f:
            f.write(script_code)
        print("[OK] scripts/utils/verify_readme_counts.py updated to enforce 130 attention mechanisms.")

if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    sync_all(root)
