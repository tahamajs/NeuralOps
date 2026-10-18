# -*- coding: utf-8 -*-
# Author: Taha Majlesi
import os
import time
import torch
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM
from neuralops.utils.replace_attention import replace_attention

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device} ({torch.cuda.get_device_name(0)})")

# 1. Prepare Architectures
print("\n=== Initializing Qwen-2.5 and LLaMA-3 Architectures ===")
# Use local cached config for Qwen
cfg_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots")
snap = os.listdir(cfg_path)[0]
cfg_qwen = AutoConfig.from_pretrained(os.path.join(cfg_path, snap))

# LLaMA-3 architecture configuration (16 layers, 2048 hidden, 16 heads)
cfg_llama = AutoConfig.for_model(
    "llama",
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=8192
)

architectures = [
    ("Qwen-2.5-0.5B (24L, 896D)", cfg_qwen),
    ("LLaMA-3-Arch (12L, 1024D)", cfg_llama)
]

seq_lens = [1024, 2048, 4096]
mechanisms = ["baseline", "svd_linear_attention", "gla_attention"]

results = []

for arch_name, cfg in architectures:
    print(f"\n==================== ARCHITECTURE: {arch_name} ====================")
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length L = {seq_len} ---")
        inputs = torch.randint(0, 1000, (1, seq_len), device=device)
        
        for mech in mechanisms:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                m = AutoModelForCausalLM.from_config(cfg).to(device).eval()
                if mech != "baseline":
                    m = replace_attention(m, attention_key=mech)
                    
                # Warmup
                with torch.no_grad():
                    _ = m(inputs)
                    
                torch.cuda.synchronize()
                iters = 5
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iters):
                        out = m(inputs)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) / iters * 1000
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                finite = torch.isfinite(out.logits).all().item()
                
                print(f"{arch_name[:12]} | L={seq_len:4d} | Mech: {mech:<22} | Latency: {latency_ms:6.2f} ms | VRAM: {peak_vram:7.2f} MB | Finite: {finite}")
                
                results.append({
                    "Architecture": arch_name,
                    "Sequence_Length": seq_len,
                    "Attention_Mechanism": mech,
                    "Latency_ms": round(latency_ms, 2),
                    "Peak_VRAM_MB": round(peak_vram, 2),
                    "Finite_Logits": finite,
                    "Status": "OK"
                })
            except torch.cuda.OutOfMemoryError:
                print(f"{arch_name[:12]} | L={seq_len:4d} | Mech: {mech:<22} | OOM")
                results.append({
                    "Architecture": arch_name,
                    "Sequence_Length": seq_len,
                    "Attention_Mechanism": mech,
                    "Latency_ms": None,
                    "Peak_VRAM_MB": None,
                    "Finite_Logits": False,
                    "Status": "OOM"
                })
            except Exception as e:
                print(f"{arch_name[:12]} | L={seq_len:4d} | Mech: {mech:<22} | Error: {e}")
                results.append({
                    "Architecture": arch_name,
                    "Sequence_Length": seq_len,
                    "Attention_Mechanism": mech,
                    "Latency_ms": None,
                    "Peak_VRAM_MB": None,
                    "Finite_Logits": False,
                    "Status": f"ERROR: {e}"
                })

# Export CSV
df = pd.DataFrame(results)
os.makedirs("/mnt/d/SVD_linear_Attention/docs/benchmarks", exist_ok=True)
csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_qwen_surgery.csv"
df.to_csv(csv_path, index=False)
print(f"\nExported {csv_path} successfully.")

# Export LaTeX Deliverable
latex_content = r"""% Publication Benchmark Deliverable: Qwen-2.5 & LLaMA-3 Architecture Model Surgery
% Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128

\begin{table*}[t]
\centering
\small
\caption{\textbf{Model Surgery Scaling Benchmarks on Qwen-2.5 and LLaMA-3 Architecture Blocks.} Forward step latency and peak GPU memory footprint across sequence lengths $L \in \{1024, 2048, 4096\}$ on physical RTX 4070 Laptop GPU (8GB VRAM).}
\label{tab:qwen_llama_surgery}
\begin{tabular}{llcccccc}
\toprule
\textbf{Architecture} & \textbf{Attention Mechanism} & \multicolumn{2}{c}{$L=1024$} & \multicolumn{2}{c}{$L=2048$} & \multicolumn{2}{c}{$L=4096$} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
 & & \textbf{Latency (ms)} & \textbf{VRAM (MB)} & \textbf{Latency (ms)} & \textbf{VRAM (MB)} & \textbf{Latency (ms)} & \textbf{VRAM (MB)} \\
\midrule
"""

for arch_name in df["Architecture"].unique():
    sub_df = df[df["Architecture"] == arch_name]
    for mech in ["baseline", "svd_linear_attention", "gla_attention"]:
        name_short = arch_name.split()[0]
        mech_name = mech.replace("_", " ").title()
        row_str = f"{name_short} & {mech_name}"
        for l in [1024, 2048, 4096]:
            m_row = sub_df[(sub_df["Attention_Mechanism"] == mech) & (sub_df["Sequence_Length"] == l)]
            if len(m_row) > 0 and m_row.iloc[0]["Status"] == "OK":
                lat = f"{m_row.iloc[0]['Latency_ms']:.1f}"
                vram = f"{m_row.iloc[0]['Peak_VRAM_MB']:.0f}"
                row_str += f" & {lat} & {vram}"
            else:
                row_str += " & OOM & OOM"
        row_str += r" \\" + "\n"
        latex_content += row_str
    latex_content += "\\midrule\n"

latex_content = latex_content.rstrip("\\midrule\n") + "\n" + r"""\bottomrule
\end{tabular}
\end{table*}
"""

from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
tex_path = out_dir / "PAPER_TABLES_QWEN_SURGERY.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_content)
print(f"Exported {tex_path} successfully.")
