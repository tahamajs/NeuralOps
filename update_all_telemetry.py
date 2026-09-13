import json, re

with open("master_verified_results.json", "r") as f:
    bench_data = json.load(f).get("modules", {})

with open("paper/neuralops_paper_final/appendices/app_06_visual_catalog.tex", "r", encoding="utf-8") as f:
    content = f.read()

def find_bench(mod_name):
    clean = mod_name.replace(r"\_", "_")
    candidates = [
        clean,
        clean.replace("_attention", ""),
        clean.replace("_compressor", ""),
        clean.replace("_engine", ""),
        clean.replace("_block", ""),
        clean.replace("_module", ""),
        clean.replace("_distiller", ""),
        clean.replace("_pruner", "")
    ]
    for c in candidates:https://brave.com/leo/?mtm_source=brave-browser&mtm_medium=brave-ntp&mtm_campaign=brave-leo-launch&mtm_content=99425607-2724-4502-a2a9-ea6822c1ab93
        if c in bench_data:
            return clean, bench_data[c]
    return clean, None

def generate_table(raw_mod_name):
    clean_name, bdata = find_bench(raw_mod_name)
    if not bdata or not bdata.get("models"):
        return None
    
    models = bdata["models"]
    rows = []
    # Sort models nicely: DistilBERT, RoBERTa, BERT, GPT-2, Qwen, etc.
    model_order = ["DistilBERT", "RoBERTa", "BERT", "GPT-2", "Qwen"]
    sorted_keys = sorted(models.keys(), key=lambda x: model_order.index(x) if x in model_order else 99)
    
    for mname in sorted_keys:
        minfo = models[mname]
        st = minfo.get("status", "N/A")
        if "SKIP" in st:
            continue
        acc_val = minfo.get("accuracy_pct")
        acc = f"{acc_val:.2f}\\%" if acc_val is not None else "N/A"
        f1_val = minfo.get("f1_score")
        if f1_val is not None:
            f1_str = f"{f1_val*100:.2f}\\%" if f1_val <= 1.0 else f"{f1_val:.2f}\\%"
        else:
            f1_str = "---"
        vram_val = minfo.get("peak_vram_mb")
        vram = f"{vram_val:.1f} MB" if vram_val is not None else "---"
        thru_val = minfo.get("throughput_tokens_sec")
        thru = f"{thru_val:,.0f} tok/s" if thru_val is not None else "---"
        rows.append(f"{mname} & \\texttt{{{st}}} & {acc} & {f1_str} & {vram} & {thru} \\\\")
    
    if not rows:
        return None
        
    caption_text = "\\textbf{Empirical Benchmark Telemetry for \\texttt{" + raw_mod_name + "}}. Measured on NVIDIA RTX 4070 Laptop GPU across standard GLUE tasks."
    label_text = f"tab:telemetry_{clean_name}"
    rows_text = "\n".join(rows)
    table_latex = f"""\\begin{{table}}[h]
\\centering
\\small
\\caption{{{caption_text}}}
\\label{{{label_text}}}
\\begin{{tabularx}}{{\\columnwidth}}{{lccccc}}
\\toprule
\\textbf{{Model Family}} & \\textbf{{Status}} & \\textbf{{Accuracy}} & \\textbf{{F1-Score}} & \\textbf{{Peak VRAM}} & \\textbf{{Throughput}} \\\\
\\midrule
{rows_text}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}"""
    return table_latex

# Replace old dummy tables in content
# The old table pattern is:
# \begin{table}[h]
# \centering
# \small
# \caption{\textbf{Empirical Benchmark Telemetry for \texttt{<raw_mod_name>}}...
# ...
# \end{table}

pattern = re.compile(r'\\begin\{table\}\[h\]\s*\\centering\s*\\small\s*\\caption\{\\textbf\{Empirical Benchmark Telemetry for \\texttt\{([^\}]+)\}\}.*?\\end\{table\}', re.DOTALL)

updated_count = 0
def replacer(match):
    global updated_count
    raw_mod = match.group(1)
    new_tbl = generate_table(raw_mod)
    if new_tbl:
        updated_count += 1
        return new_tbl
    return match.group(0)

new_content = pattern.sub(replacer, content)

print(f"Total tables updated across all modules: {updated_count}")

with open("paper/neuralops_paper_final/appendices/app_06_visual_catalog.tex", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Saved updated app_06_visual_catalog.tex successfully!")
