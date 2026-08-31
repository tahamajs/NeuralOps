import csv
import json

csv_path = r"D:\SVD_Project\results\final_paper_run\summary.csv"
out_csv = r"D:\SVD_Project\results\final_paper_run\sst2_plotting_data.csv"

def classify_family(name):
    n = name.lower()
    if any(k in n for k in ["svd", "spectral", "rpca", "pca", "asvd", "saes", "subspace", "grecodec"]):
        return "SVD & Spectral Factorization"
    elif any(k in n for k in ["moe", "expert", "tomoe", "bam", "reap"]):
        return "Mixture-of-Experts (MoE)"
    elif any(k in n for k in ["ssm", "mamba", "hgrn", "rnn", "griffin", "s4", "samba", "simple_ssm"]):
        return "Linear RNN & State Space Models (SSM)"
    elif any(k in n for k in ["prun", "proxsparse", "sparse", "flap", "slice", "slim", "bonsai", "lsc"]):
        return "Pruning & Sparsity"
    elif any(k in n for k in ["quant", "kmeans", "int4", "fp8", "coopq", "hash", "admm"]):
        return "Quantization & Hash Compression"
    elif any(k in n for k in ["cur"]):
        return "CUR Decompositions"
    elif any(k in n for k in ["rope", "gqa", "wgqa", "kv", "cachenotes", "zerokv", "hybridkv", "tpa"]):
        return "KV Cache & Architectural Attention"
    elif any(k in n for k in ["flash", "cosformer", "fasa", "faq", "eat", "cluster", "astro", "nystrom", "performer", "linear", "linformer", "galerkin", "infini", "mixattention", "moba", "nsa", "nexus", "leka", "aegis", "enlighten", "span", "block", "fusion", "compot", "come", "dobi", "kronecker", "tt", "lsh", "modegpt", "nli", "nmf", "oats", "oha", "orion", "phantom", "rala", "regla", "rpc", "salr", "salt", "sdrm", "soka", "spectron", "squeezing", "uniql", "vig", "vspec", "xattention", "zeros", "liftr", "lillama", "lola", "mla", "primal", "hymba", "jamba"]):
        return "Efficient Attention & Kernelization"
    else:
        return "Other / Hybrid"

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    sst2_pass = [r for r in reader if r["dataset"].strip().upper() == "SST2" and r["status"].strip().upper() == "PASS"]

fieldnames = ["module_name", "family", "accuracy", "f1_score", "fidelity_score", "throughput_tokens_sec", "peak_vram_mb", "fwd_ms", "bwd_ms"]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sst2_pass:
        writer.writerow({
            "module_name": r["module_name"],
            "family": classify_family(r["module_name"]),
            "accuracy": float(r["mean_accuracy"]),
            "f1_score": float(r["f1_score"]),
            "fidelity_score": float(r["fidelity_score"]),
            "throughput_tokens_sec": float(r["throughput_tokens_sec"]),
            "peak_vram_mb": float(r["peak_vram_mb"]),
            "fwd_ms": float(r["fwd_ms"]),
            "bwd_ms": float(r["bwd_ms"]),
        })

print(f"Generated {out_csv} with {len(sst2_pass)} records.")
