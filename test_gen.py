import json, re

with open("master_verified_results.json", "r") as f:
    bench_data = json.load(f).get("modules", {})

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
    for c in candidates:
        if c in bench_data:
            return clean, bench_data[c]
    return clean, None

def generate_table(raw_mod_name):
    clean_name, bdata = find_bench(raw_mod_name)
    if not bdata or not bdata.get("models"):
        return None
    
    models = bdata["models"]
    rows = []
    for mname, minfo in models.items():
        st = minfo.get("status", "N/A")
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

print(generate_table("adaptive\\_span\\_attention"))
print("-" * 50)
print(generate_table("fusion\\_attn"))
print("-" * 50)
print(generate_table("primal\\_attention"))
