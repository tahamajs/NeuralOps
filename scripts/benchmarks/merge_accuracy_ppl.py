# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Merge accuracy ppl module.
"""

import json
import os
import sys


def merge(node_name, device_name):
    """Execute merge routine."""
    p0 = f"results/accuracy_and_ppl_benchmark_{node_name}_part0.json"
    p1 = f"results/accuracy_and_ppl_benchmark_{node_name}_part1.json"
    if not (os.path.exists(p0) and os.path.exists(p1)):
        print(f"Skipping {node_name}: missing part files ({p0}, {p1})")
        return
    with open(p0) as f:
        d0 = json.load(f)
    with open(p1) as f:
        d1 = json.load(f)
    merged = {
        "device": device_name,
        "baseline_roberta_accuracy_pct": d0.get("baseline_roberta_accuracy_pct", 95.0),
        "baseline_gpt2_ppl": d0.get("baseline_gpt2_ppl", 27.56),
        "results": d0["results"] + d1["results"],
    }
    print(f"Merged {node_name}: {len(merged['results'])} modules")
    with open(f"results/accuracy_and_ppl_benchmark_{node_name}.json", "w") as f:
        json.dump(merged, f, indent=2)
    with open(f"results/accuracy_and_ppl_benchmark_{node_name}.csv", "w") as f:
        f.write("module,accuracy_pct,perplexity,duration_s,error\n")
        for r in merged["results"]:
            f.write(f'"{r["module"]}",{r["accuracy_pct"]},{r["perplexity"]},{r["duration_s"]},"{r["error"]}"\n')
    with open(f"results/accuracy_and_ppl_benchmark_{node_name}.md", "w") as f:
        f.write(f"# Accuracy & Perplexity Benchmark (`{node_name}` - `{device_name}`)\n\n")
        f.write(f"- **Baseline RoBERTa SST-2 Accuracy**: **{merged['baseline_roberta_accuracy_pct']}%**\n")
        f.write(f"- **Baseline GPT-2 WikiText-2 PPL**: **{merged['baseline_gpt2_ppl']}**\n\n")
        f.write("| # | Attention Module | SST-2 Accuracy (%) | WikiText-2 PPL | Latency (s) | Status / Error |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :--- |\n")
        for i, r in enumerate(merged["results"], 1):
            acc_val = f"**{r['accuracy_pct']}%**" if r["accuracy_pct"] is not None else "N/A"
            ppl_val = f"**{r['perplexity']}**" if r["perplexity"] is not None else "N/A"
            err_val = r["error"] if r["error"] else "OK"
            f.write(f"| {i} | `{r['module']}` | {acc_val} | {ppl_val} | {r['duration_s']} | {err_val} |\n")


if __name__ == "__main__":
    node = sys.argv[1] if len(sys.argv) > 1 else "taha"
    dev = sys.argv[2] if len(sys.argv) > 2 else "NVIDIA GPU"
    merge(node, dev)
