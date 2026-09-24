import subprocess
import os

modules = ["moba", "gqa", "hla_inject"]
for mod in modules:
    print(f"=== Running PPL for {mod} ===")
    cmd = [
        "python", "D:/SVD_Project/Codes/framework/eval_perplexity.py",
        "--model", "D:/SVD_Project/models/gpt2",
        "--module", mod,
        "--output_dir", f"D:/SVD_Project/results/wikitext2_{mod}/gpt2"
    ]
    subprocess.run(cmd, check=True)
