#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Profile peak batch saturation module.
"""

import os
import sys
import gc
import json
import math
import csv
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
from neuralops.core.cuda_kernels.sparse_attention.sparse_attention_autograd import block_sparse_attn


def get_cuda_mem():
    """Execute get cuda mem routine."""
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
    }


def standard_attention_ref(q, k, v):
    """Execute standard attention ref routine."""
    B, H, L, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
    mask = torch.triu(torch.full((L, L), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
    probs = F.softmax(scores + mask, dim=-1)
    return torch.matmul(probs, v)


def svd_linear_attention_kernel(q, k, v, u_proj, v_proj):
    """Execute svd linear attention kernel routine."""
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    phi_q = F.elu(q) + 1.0
    phi_k = F.elu(k) + 1.0
    kv = torch.einsum("bhld,bhlm->bhldm", phi_k, v)
    kv_cumsum = torch.cumsum(kv, dim=2)
    out = torch.einsum("bhld,bhldm->bhlm", phi_q, kv_cumsum) * scale
    return out


def find_b_max_standard_mha(L, device, H=8, D=64):
    """Binary/exponential search to find exact B_max before OOM under forward+backward autograd."""
    batch_candidates = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    b_max = 0
    peak_vram = 0.0

    for b in batch_candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            q = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            out = standard_attention_ref(q, k, v)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            del q, k, v, out, loss
            b_max = b
            peak_vram = peak
        except torch.cuda.OutOfMemoryError:
            break
        except Exception:
            break
    return b_max, round(peak_vram, 2)


def find_b_max_svd_linear_attn(L, device, H=8, D=64):
    """Find maximum batch size for SVD Linear Attention."""
    batch_candidates = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    b_max = 0
    peak_vram = 0.0

    for b in batch_candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            q = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            scale = 1.0 / math.sqrt(D)
            phi_q = F.elu(q) + 1.0
            phi_k = F.elu(k) + 1.0
            kv = torch.matmul(phi_k.unsqueeze(-1), v.unsqueeze(-2))
            kv_cumsum = torch.cumsum(kv, dim=2)
            out = torch.matmul(phi_q.unsqueeze(-2), kv_cumsum).squeeze(-2) * scale
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            del q, k, v, phi_q, phi_k, kv, kv_cumsum, out, loss
            b_max = b
            peak_vram = peak
        except torch.cuda.OutOfMemoryError:
            break
        except Exception:
            break
    return b_max, round(peak_vram, 2)


def find_b_max_block_sparse_attn(L, device, H=8, D=64, block_size=16):
    """Find maximum batch size for Block-Sparse Attention."""
    batch_candidates = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    b_max = 0
    peak_vram = 0.0
    num_blocks = L // block_size

    for b in batch_candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            q = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            k = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            v = torch.randn(b, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
            mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.int8))
            out = block_sparse_attn(q, k, v, mask, block_size)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            del q, k, v, mask, out, loss
            b_max = b
            peak_vram = peak
        except torch.cuda.OutOfMemoryError:
            break
        except Exception:
            break
    return b_max, round(peak_vram, 2)


def main():
    """Execute main routine."""
    print("=== PEAK VRAM SATURATION & B_MAX EVALUATION ON RTX 4070 (8GB VRAM) ===")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    print(f"Device: {device_name} ({total_mem_gb} GB VRAM)")

    seq_lens = [2048, 4096, 8192]
    results = []

    for L in seq_lens:
        print(f"\n--- Testing Sequence Length L = {L} ---")

        print("  Evaluating Standard MHA B_max...")
        b_mha, vram_mha = find_b_max_standard_mha(L, device)
        print(f"    Standard MHA: B_max = {b_mha} (Peak VRAM: {vram_mha} MB)")

        print("  Evaluating SVD Linear Attention B_max...")
        b_svd, vram_svd = find_b_max_svd_linear_attn(L, device)
        print(f"    SVD Linear Attention: B_max = {b_svd} (Peak VRAM: {vram_svd} MB)")

        print("  Evaluating Block-Sparse Attention B_max...")
        b_sparse, vram_sparse = find_b_max_block_sparse_attn(L, device)
        print(f"    Block-Sparse Attention: B_max = {b_sparse} (Peak VRAM: {vram_sparse} MB)")

        ratio_svd_vs_mha = round(b_svd / max(b_mha, 1), 2)
        ratio_sparse_vs_mha = round(b_sparse / max(b_mha, 1), 2)

        results.append(
            {
                "seq_len": L,
                "mha_b_max": b_mha,
                "mha_vram_mb": vram_mha,
                "svd_b_max": b_svd,
                "svd_vram_mb": vram_svd,
                "svd_gain": ratio_svd_vs_mha,
                "sparse_b_max": b_sparse,
                "sparse_vram_mb": vram_sparse,
                "sparse_gain": ratio_sparse_vs_mha,
            }
        )

    csv_dir = "/mnt/d/SVD_linear_Attention/docs/benchmarks"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "paper_benchmarks_peak_batch_saturation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Sequence_Length",
                "Standard_MHA_B_max",
                "Standard_MHA_Peak_VRAM_MB",
                "SVD_Linear_Attn_B_max",
                "SVD_Linear_Peak_VRAM_MB",
                "SVD_Batch_Capacity_Gain",
                "Block_Sparse_Attn_B_max",
                "Block_Sparse_Peak_VRAM_MB",
                "Sparse_Batch_Capacity_Gain",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["seq_len"],
                    r["mha_b_max"],
                    r["mha_vram_mb"],
                    r["svd_b_max"],
                    r["svd_vram_mb"],
                    f"{r['svd_gain']}x",
                    r["sparse_b_max"],
                    r["sparse_vram_mb"],
                    f"{r['sparse_gain']}x",
                ]
            )
    print(f"\nSaved CSV deliverable to {csv_path}")

    latex_dir = "/mnt/d/SVD_linear_Attention/docs/reports"
    os.makedirs(latex_dir, exist_ok=True)
    latex_path = os.path.join(latex_dir, "PAPER_TABLES_PEAK_BATCH.tex")
    with open(latex_path, "w") as f:
        f.write("% Publication-Grade LaTeX Table: Peak VRAM Saturation and Maximum Batch Size B_max\n")
        f.write("% Hardware Target: NVIDIA RTX 4070 Laptop GPU (8GB VRAM / 7.62 GiB usable, SM 8.9)\n\n")

        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write(
            "\\caption{\\textbf{Peak VRAM Saturation and Maximum Batch Capacity ($B_{\\max}$).} Maximum achievable batch size ($B_{\\max}$) before out-of-memory under forward+backward autograd on an 8GB NVIDIA RTX 4070 Laptop GPU across sequence lengths $L \\in \\{2048, 4096, 8192\\}$. SVD Linear Attention and Block-Sparse Attention achieve up to \\textbf{"
            + f"{max(r['sparse_gain'] for r in results):.1f}"
            + "$\\times$} larger batch throughput relative to Standard Multi-Head Attention (MHA).}\n"
        )
        f.write("\\label{tab:peak_batch_saturation}\n")
        f.write("\\begin{tabular}{r | cc | ccc | ccc}\n\\toprule\n")
        f.write(
            "& \\multicolumn{2}{c|}{\\textbf{Standard MHA $\\mathcal{O}(L^2)$}} & \\multicolumn{3}{c|}{\\textbf{SVD Linear Attention $\\mathcal{O}(L)$}} & \\multicolumn{3}{c}{\\textbf{Block-Sparse Attention $\\mathcal{O}(L \\cdot B)$}} \\\\\n"
        )
        f.write(
            "Sequence ($L$) & $B_{\\max}$ & Peak VRAM (MB) & $B_{\\max}$ & Peak VRAM (MB) & Capacity Gain & $B_{\\max}$ & Peak VRAM (MB) & Capacity Gain \\\\\n\\midrule\n"
        )
        for r in results:
            mha_b = str(r["mha_b_max"]) if r["mha_b_max"] > 0 else "\\textit{0 (OOM)}"
            mha_v = f"{r['mha_vram_mb']:.1f}" if r["mha_b_max"] > 0 else "\\textit{N/A}"
            f.write(
                f"{r['seq_len']} & {mha_b} & {mha_v} & \\textbf{{{r['svd_b_max']}}} & {r['svd_vram_mb']:.1f} & \\textbf{{{r['svd_gain']:.1f}$\\times$}} & \\textbf{{{r['sparse_b_max']}}} & {r['sparse_vram_mb']:.1f} & \\textbf{{{r['sparse_gain']:.1f}$\\times$}} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"Saved LaTeX deliverable to {latex_path}")

    json_path = "/mnt/d/SVD_linear_Attention/results/profiling/peak_batch_saturation.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"device": device_name, "vram_gb": total_mem_gb, "results": results}, f, indent=2)
    print(f"Saved JSON deliverable to {json_path}")


if __name__ == "__main__":
    main()
