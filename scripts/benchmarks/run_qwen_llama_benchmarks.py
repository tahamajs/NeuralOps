#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run qwen llama benchmarks module.
"""

import os
import time
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM
from neuralops.utils.replace_attention import replace_attention

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device} ({torch.cuda.get_device_name(0)})")

print("\n=== Initializing Qwen-2.5 and LLaMA-3 Architectures ===")
try:
    cfg_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots")
    snap = os.listdir(cfg_path)[0]
    cfg_qwen = AutoConfig.from_pretrained(os.path.join(cfg_path, snap))
except Exception:
    try:
        cfg_qwen = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception:
        cfg_qwen = AutoConfig.for_model(
            "qwen2",
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            max_position_embeddings=32768,
        )

cfg_llama = AutoConfig.for_model(
    "llama",
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=8192,
)

architectures = [("Qwen-2.5-0.5B (24L, 896D)", cfg_qwen), ("LLaMA-3-Arch (12L, 1024D)", cfg_llama)]

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

                with torch.no_grad():
                    _ = m(inputs)

                torch.cuda.synchronize()
                iters = 5
                if device == "cuda":
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    with torch.no_grad():
                        start_evt.record()
                        for _ in range(iters):
                            out = m(inputs)
                        end_evt.record()
                    torch.cuda.synchronize()
                    latency_ms = start_evt.elapsed_time(end_evt) / iters
                else:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        for _ in range(iters):
                            out = m(inputs)
                    latency_ms = (time.perf_counter() - t0) / iters * 1000.0
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
                finite = torch.isfinite(out.logits).all().item()

                print(
                    f"{arch_name[:12]} | L={seq_len:4d} | Mech: {mech:<22} | Latency: {latency_ms:6.2f} ms | VRAM: {peak_vram:7.2f} MB | Finite: {finite}"
                )

                results.append(
                    {
                        "Architecture": arch_name,
                        "Sequence_Length": seq_len,
                        "Attention_Mechanism": mech,
                        "Latency_ms": round(latency_ms, 2),
                        "Peak_VRAM_MB": round(peak_vram, 2),
                        "Finite_Logits": finite,
                        "Status": "OK",
                    }
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{arch_name[:12]} | L={seq_len:4d} | Mech: {mech:<22} | OOM")
                results.append(
                    {
                        "Architecture": arch_name,
                        "Sequence_Length": seq_len,
                        "Attention_Mechanism": mech,
                        "Latency_ms": None,
                        "Peak_VRAM_MB": None,
                        "Finite_Logits": False,
                        "Status": "OOM",
                    }
                )
            except Exception as e:
                print(f"{arch_name[:12]} | L={seq_len:4d} | Mech: {mech:<22} | Error: {e}")
                results.append(
                    {
                        "Architecture": arch_name,
                        "Sequence_Length": seq_len,
                        "Attention_Mechanism": mech,
                        "Latency_ms": None,
                        "Peak_VRAM_MB": None,
                        "Finite_Logits": False,
                        "Status": f"ERROR: {e}",
                    }
                )

df = pd.DataFrame(results)
repo_root = Path(__file__).resolve().parent.parent.parent
bench_dir = repo_root / "docs" / "benchmarks"
bench_dir.mkdir(parents=True, exist_ok=True)
csv_path = bench_dir / "paper_benchmarks_qwen_surgery.csv"
df.to_csv(csv_path, index=False)
print(f"\nExported {csv_path} successfully.")

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

latex_content = (
    latex_content.rstrip("\\midrule\n")
    + "\n"
    + r"""\bottomrule
\end{tabular}
\end{table*}
"""
)

from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
tex_path = out_dir / "PAPER_TABLES_QWEN_SURGERY.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_content)
print(f"Exported {tex_path} successfully.")
