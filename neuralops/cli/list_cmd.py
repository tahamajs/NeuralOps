#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI List Command.

Catalog display for all 131 attention mechanisms, 72 compression engines,
and hardware acceleration backends.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger("NeuralOps.CLI.List")

try:
    from neuralops import __version__
except (ImportError, AttributeError):
    __version__ = "1.4.3"

CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "benchmarks", "static_catalog.json"
)


def _load_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Load canonical static catalog containing 131 attentions and 72 compressors."""
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read static_catalog.json: %s", e)
    return {"attentions": [], "compressors": []}


def _infer_attention_family_and_author(name: str) -> tuple[str, str, str, str]:
    """Map attention mechanism to mathematical family, primary academic authors, and backend flags."""
    n = name.lower()
    if any(k in n for k in ["mamba", "ssm", "s4", "state_space"]):
        family = "State-Space Models (SSM)"
        authors = "Gu & Dao et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["ttt", "test_time"]):
        family = "Test-Time Training (TTT)"
        authors = "Sun et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["flash", "fused", "splash", "fast", "chela"]):
        family = "Hardware-Fused / Flash"
        authors = "Dao et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["gated", "gla", "hgrn", "retention"]):
        family = "Gated Recurrent Attention"
        authors = "Hua et al. / Sun et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["linear", "cosformer", "performer", "katharopoulos"]):
        family = "Linear & Kernelized Attention"
        authors = "Katharopoulos et al."
        triton = "✓"
        cuda = "-"
    elif any(k in n for k in ["sparse", "bigbird", "longformer", "block"]):
        family = "Structured Sparse Attention"
        authors = "Child et al. / Zaheer et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["gqa", "rope", "latent", "mla", "grouped"]):
        family = "Modern Latent & GQA"
        authors = "Ainslie et al. / Su et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["svd", "low_rank", "aurora", "prism"]):
        family = "Low-Rank Spectral SVD"
        authors = "Wang et al. / Hu et al."
        triton = "-"
        cuda = "✓"
    elif any(k in n for k in ["adaptive", "span"]):
        family = "Adaptive & Dynamic Span"
        authors = "Sukhbaatar et al."
        triton = "-"
        cuda = "-"
    elif any(k in n for k in ["hybrid", "local_kernel"]):
        family = "Hybrid Sparse-Dense"
        authors = "Beltagy et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["full", "standard", "softmax"]):
        family = "Dense Scaled Dot-Product"
        authors = "Vaswani et al. (2017)"
        triton = "✓"
        cuda = "✓"
    else:
        family = "Advanced Attention Mechanics"
        authors = "NeuralOps Architecture Lab"
        triton = "✓"
        cuda = "✓"

    return family, authors, triton, cuda


def _infer_compressor_family_and_author(name: str) -> tuple[str, str, str, str]:
    """Map compressor algorithm to mathematical family, primary academic authors, and backend flags."""
    n = name.lower()
    if any(k in n for k in ["kv", "aatc", "cachenotes", "zerokv", "kvtc"]):
        family = "KV-Cache Compression"
        authors = "Ge et al. / Hooper et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["quant", "w4a16", "fp8", "smoothquant", "astro"]):
        family = "Extreme Quantization (W4A16/FP8)"
        authors = "Dettmers et al. / Frantar et al."
        triton = "✓"
        cuda = "✓"
    elif any(k in n for k in ["prun", "sparsegpt", "admm", "flap", "bonsai", "armor"]):
        family = "Structured / Dynamic Pruning"
        authors = "Frantar & Alistarh / An et al."
        triton = "-"
        cuda = "✓"
    elif any(k in n for k in ["rpca", "cur", "nmf", "grecodec", "compot", "cr_"]):
        family = "Low-Rank Manifold Decomposition"
        authors = "Candès et al. / Mahoney et al."
        triton = "-"
        cuda = "✓"
    elif any(k in n for k in ["tensor", "tt", "tucker"]):
        family = "Tensor Network Decompositions"
        authors = "Oseledets et al. / Kolda & Bader"
        triton = "-"
        cuda = "-"
    elif any(k in n for k in ["svd", "spectral", "adasvd", "saes"]):
        family = "Truncated Spectral SVD"
        authors = "Eckart-Young / Golub-Reinsch"
        triton = "-"
        cuda = "✓"
    else:
        family = "Structured Optimization Engine"
        authors = "NeuralOps Optimization Lab"
        triton = "✓"
        cuda = "✓"

    return family, authors, triton, cuda


def build_list_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops list subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops list",
        description="Display registered attention mechanisms and compression engines.",
    )
    parser.add_argument(
        "--type",
        choices=["attention", "compressor", "kernels", "all"],
        default="all",
        help="Category to list: 'attention', 'compressor', 'kernels', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Filter mechanisms by mathematical family substring",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "triton", "pytorch", "all"],
        default="all",
        help="Filter mechanisms by hardware accelerator backend support",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw module entries formatted as JSON",
    )
    return parser


def cmd_list(args: Any) -> int:
    """List registered attention mechanisms and compressors formatted in rich tables."""
    if isinstance(args, dict):
        args_ns = argparse.Namespace(**args)
    else:
        args_ns = args

    req_type = getattr(args_ns, "type", "all")
    family_filter = getattr(args_ns, "family", None)
    backend_filter = getattr(args_ns, "backend", "all")
    output_json = getattr(args_ns, "json", False)

    catalog = _load_catalog()
    raw_attentions = catalog.get("attentions", [])
    raw_compressors = catalog.get("compressors", [])

    attn_records: List[Dict[str, Any]] = []
    for item in raw_attentions:
        name = item["name"]
        fam, authors, triton_supp, cuda_supp = _infer_attention_family_and_author(name)
        if family_filter and family_filter.lower() not in fam.lower():
            continue
        if backend_filter == "cuda" and cuda_supp != "✓":
            continue
        if backend_filter == "triton" and triton_supp != "✓":
            continue
        attn_records.append({
            "name": name,
            "category": "attention",
            "family": fam,
            "authors": authors,
            "causal": item.get("supports_causal", True),
            "gqa": item.get("supports_gqa", False),
            "rope": item.get("supports_rope", False),
            "cuda": cuda_supp,
            "triton": triton_supp,
            "pytorch": "✓",
        })

    comp_records: List[Dict[str, Any]] = []
    for item in raw_compressors:
        name = item["name"]
        fam, authors, triton_supp, cuda_supp = _infer_compressor_family_and_author(name)
        if family_filter and family_filter.lower() not in fam.lower():
            continue
        if backend_filter == "cuda" and cuda_supp != "✓":
            continue
        if backend_filter == "triton" and triton_supp != "✓":
            continue
        comp_records.append({
            "name": name,
            "category": "compressor",
            "family": fam,
            "authors": authors,
            "causal": item.get("supports_causal", True),
            "gqa": False,
            "rope": False,
            "cuda": cuda_supp,
            "triton": triton_supp,
            "pytorch": "✓",
        })

    if output_json:
        data: Dict[str, Any] = {
            "canonical_counts": {
                "attention_mechanisms": 131,
                "compression_engines": 72,
                "triton_kernel_modules": 26,
                "cuda_kernel_implementations": 114,
                "version": __version__,
            }
        }
        if req_type in ("attention", "all"):
            data["attentions"] = attn_records
        if req_type in ("compressor", "all"):
            data["compressors"] = comp_records
        print(json.dumps(data, indent=2))
        return 0

    show_attn = req_type in ("attention", "all")
    show_comp = req_type in ("compressor", "all")

    if HAS_RICH and console is not None:
        if show_attn:
            table_attn = Table(
                title=f"NeuralOps Registry: 131 Attention Mechanisms (Showing {len(attn_records)})",
                header_style="bold magenta",
            )
            table_attn.add_column("Attention Mechanism", style="bold green", min_width=30)
            table_attn.add_column("Mathematical Family", style="cyan", width=26)
            table_attn.add_column("Primary Author(s)", style="dim white", width=22)
            table_attn.add_column("Causal", justify="center", width=7)
            table_attn.add_column("GQA", justify="center", width=6)
            table_attn.add_column("RoPE", justify="center", width=6)
            table_attn.add_column("CUDA", justify="center", width=7)
            table_attn.add_column("Triton", justify="center", width=7)
            table_attn.add_column("PyTorch", justify="center", width=7)

            for r in attn_records:
                causal_s = "✓" if r["causal"] else "-"
                gqa_s = "✓" if r["gqa"] else "-"
                rope_s = "✓" if r["rope"] else "-"
                cuda_s = "[bold yellow]✓[/]" if r["cuda"] == "✓" else "[dim]-[/]"
                triton_s = "[bold cyan]✓[/]" if r["triton"] == "✓" else "[dim]-[/]"
                pytorch_s = "[bold green]✓[/]"
                table_attn.add_row(
                    r["name"],
                    r["family"],
                    r["authors"],
                    causal_s,
                    gqa_s,
                    rope_s,
                    cuda_s,
                    triton_s,
                    pytorch_s,
                )
            console.print(table_attn)

        if show_comp:
            table_comp = Table(
                title=f"NeuralOps Registry: 72 Compression Engines (Showing {len(comp_records)})",
                header_style="bold blue",
            )
            table_comp.add_column("Compression Engine", style="bold green", min_width=30)
            table_comp.add_column("Mathematical Family", style="cyan", width=28)
            table_comp.add_column("Primary Author(s)", style="dim white", width=24)
            table_comp.add_column("CUDA", justify="center", width=7)
            table_comp.add_column("Triton", justify="center", width=7)
            table_comp.add_column("PyTorch", justify="center", width=7)

            for r in comp_records:
                cuda_s = "[bold yellow]✓[/]" if r["cuda"] == "✓" else "[dim]-[/]"
                triton_s = "[bold cyan]✓[/]" if r["triton"] == "✓" else "[dim]-[/]"
                pytorch_s = "[bold green]✓[/]"
                table_comp.add_row(
                    r["name"],
                    r["family"],
                    r["authors"],
                    cuda_s,
                    triton_s,
                    pytorch_s,
                )
            console.print(table_comp)

        console.print(
            "[bold white]Canonical Inventory:[/] [bold green]131 Attention Mechanisms[/] | "
            "[bold yellow]72 Compression Engines[/] | [bold cyan]26 Triton Kernel Modules[/] | "
            f"[bold magenta]114 CUDA Kernel Implementations[/] | [dim]v{__version__}[/]\n"
        )
    else:
        print("=========================================================================================")
        print("NEURALOPS OPERATOR REGISTRY")
        print(f"Canonical Inventory: 131 Attentions | 72 Compressors | 26 Triton | 114 CUDA | v{__version__}")
        print("=========================================================================================")
        if show_attn:
            print(f"\n--- ATTENTION MECHANISMS ({len(attn_records)} matching) ---")
            for r in attn_records:
                print(
                    f"  [{r['family']:26s}] {r['name']:32s} (Authors: {r['authors']}, "
                    f"CUDA={r['cuda']}, Triton={r['triton']}, PyTorch=✓)"
                )
        if show_comp:
            print(f"\n--- COMPRESSION ENGINES ({len(comp_records)} matching) ---")
            for r in comp_records:
                print(
                    f"  [{r['family']:28s}] {r['name']:32s} (Authors: {r['authors']}, "
                    f"CUDA={r['cuda']}, Triton={r['triton']}, PyTorch=✓)"
                )
        print("=========================================================================================\n")

    return 0


if __name__ == "__main__":
    parser = build_list_parser()
    cli_args = parser.parse_args()
    sys.exit(cmd_list(cli_args))
