# -*- coding: utf-8 -*-
"""Export master matrix latex module.

Author: Taha Majlesi
"""

# Author: Taha Majlesi
import csv

input_csv = "results/MASTER_RESULTS_MATRIX.csv"
out_tex = "paper/iclr2027-submission/tables/table_master_benchmark_matrix.tex"
out_tex_neurips = "paper/neurips2026-submission/tables/table_master_benchmark_matrix.tex"

rows = []
with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

attentions = [r for r in rows if r["Architecture Category"] == "attention"][:45]
compressors = [r for r in rows if r["Architecture Category"] == "compressor"][:35]
kernels = [r for r in rows if r["Architecture Category"] in ("cuda_kernel", "triton_kernel")][:25]

selected = attentions + compressors + kernels


def fmt(v, default="-"):
    """Execute fmt routine."""
    if not v or v.strip() in ("", "None", "-"):
        return default
    try:
        f = float(v)
        if f == int(f) and "." not in str(v):
            return str(int(f))
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return v.replace("_", r"\_")


def esc(s):
    """Execute esc routine."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


eol = " \\\\"

tex_content = []
tex_content.append(r"\begin{center}")
tex_content.append(r"\tiny")
tex_content.append(r"\setlength{\tabcolsep}{2.5pt}")
tex_content.append(r"\begin{longtable}{lllccccccr}")
tex_content.append(
    r"\caption{\textbf{Comprehensive Master Benchmark Matrix: Physical GPU Runs across Representative Attentions, Compressors, and Kernels.}} \label{tab:master_benchmark_matrix}"
    + eol
)
tex_content.append(r"\toprule")
tex_content.append(
    r"\textbf{Module Name} & \textbf{Category} & \textbf{Family} & \textbf{CR} & \textbf{SST-2 (\%)} & \textbf{MNLI (\%)} & \textbf{AG News (\%)} & \textbf{PPL} & \textbf{VRAM (MB)} & \textbf{Speedup}"
    + eol
)
tex_content.append(r"\midrule")
tex_content.append(r"\endfirsthead")
tex_content.append(r"\toprule")
tex_content.append(
    r"\textbf{Module Name} & \textbf{Category} & \textbf{Family} & \textbf{CR} & \textbf{SST-2 (\%)} & \textbf{MNLI (\%)} & \textbf{AG News (\%)} & \textbf{PPL} & \textbf{VRAM (MB)} & \textbf{Speedup}"
    + eol
)
tex_content.append(r"\midrule")
tex_content.append(r"\endhead")
tex_content.append(r"\midrule")
tex_content.append(r"\multicolumn{10}{r}{\emph{Continued on next page\dots}}" + eol)
tex_content.append(r"\endfoot")
tex_content.append(r"\bottomrule")
tex_content.append(r"\endlastfoot")

for r in selected:
    mod = esc(r["Module Name"])
    cat = esc(r["Architecture Category"])
    fam = esc(r["Family"][:14]) if r["Family"] else "-"
    cr = fmt(r["Compression Ratio"]) + r"$\times$" if r["Compression Ratio"] and r["Compression Ratio"] != "-" else "-"
    sst = fmt(r["SST-2 Accuracy (%)"]) + r"\%" if r["SST-2 Accuracy (%)"] and r["SST-2 Accuracy (%)"] != "-" else "-"
    mnli = fmt(r["MNLI Accuracy (%)"]) + r"\%" if r["MNLI Accuracy (%)"] and r["MNLI Accuracy (%)"] != "-" else "-"
    ag = (
        fmt(r["AG News Accuracy (%)"]) + r"\%"
        if r["AG News Accuracy (%)"] and r["AG News Accuracy (%)"] != "-"
        else "-"
    )
    ppl = fmt(r["WikiText-2 Perplexity"])
    vram = fmt(r["Peak VRAM (MB)"])
    sp = fmt(r["Speedup (x)"]) + r"$\times$" if r["Speedup (x)"] and r["Speedup (x)"] != "-" else "-"

    line = f"\\texttt{{{mod}}} & {cat} & {fam} & {cr} & {sst} & {mnli} & {ag} & {ppl} & {vram} & {sp}" + eol
    tex_content.append(line)

tex_content.append(r"\end{longtable}")
tex_content.append(r"\end{center}")

with open(out_tex, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_content) + "\n")

with open(out_tex_neurips, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_content) + "\n")

print(f"Successfully generated {out_tex} and {out_tex_neurips} with {len(selected)} entries.")
