"""
NeuralOps Rich Terminal Interface (huggingface-cli / gh style).
Provides colorized banner, interactive module explorer, grouped help menus,
progress meters, and formatted tables for NeuralOps.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.markdown import Markdown
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

BANNER_ART = r"""
  _   _                      _  ____             
 | \ | | ___ _   _ _ __ __ _| |/ __ \ _ __  ___ 
 |  \| |/ _ \ | | | '__/ _` | | |  | | '_ \/ __|
 | |\  |  __/ |_| | | | (_| | | |__| | |_) \__ \
 |_| \_|\___|\__,_|_|  \__,_|_|\____/| .__/|___/
                                     |_|         
"""

SUBCOMMAND_GROUPS = [
    {
        "title": "Model Surgery & Compression",
        "commands": [
            ("compress", "Compress any model via SVD, Hadamard Incoherent SVD, W4A16, or registered engines"),
            ("search", "AutoSVD Pareto optimal rank allocation via Fisher & Gram sensitivity"),
            ("export", "Export compressed/surgerized models to safetensors, ONNX, or TorchScript"),
        ],
    },
    {
        "title": "Inference Acceleration & Long-Context Evaluation",
        "commands": [
            ("generate", "Run lossless generation with self-speculative draft-and-verify decoding"),
            ("niah", "Execute Needle-In-A-Haystack passkey benchmark & generate 2D heatmaps"),
            ("benchmark-fused", "Benchmark fused Triton kernels (W4A16 dequant-GEMM, O(1) state step)"),
        ],
    },
    {
        "title": "Benchmarking & Training Suite",
        "commands": [
            ("run", "Train, fine-tune, or benchmark any of the 131 attentions or 72 compressors"),
            ("plan", "Dry-run and compute estimated compute hours, memory, and combinations"),
            ("results", "Inspect, summarize, and display benchmark results tables"),
            ("resume", "Resume an interrupted empirical benchmark sweep from state journal"),
        ],
    },
    {
        "title": "System Diagnostics & Module Registry",
        "commands": [
            ("list", "List and filter 131 attention mechanisms and 72 compressors"),
            ("doctor", "Run comprehensive health diagnostics on GPU, Triton, and packages"),
            ("info", "Print detailed GPU hardware telemetry and registry metrics"),
            ("verify", "Fast environment sanity check (PyTorch, CUDA, MPS)"),
            ("ui", "Launch full interactive terminal interface (TUI mode)"),
        ],
    },
]


def print_banner() -> None:
    """Prints the styled NeuralOps CLI banner."""
    if not HAS_RICH or console is None:
        print("================================================================")
        print("                   NeuralOps CLI (v1.3.0)                       ")
        print("     High-Performance Attention Zoo, SVD Compression & Triton   ")
        print("================================================================")
        return

    banner_text = Text(BANNER_ART, style="bold cyan")
    tagline = Text("NeuralOps v1.3.0 — High-Performance Attention Zoo, SVD Compression & Fused Triton Kernels\n", style="bold white")
    tagline.append("https://github.com/tahamajs/SVD_linear_Attention", style="dim underline blue")

    panel = Panel(
        tagline,
        title="[bold green]⚡ NeuralOps Engine[/bold green]",
        subtitle="[dim]Pair Programming & Distributed ML Systems CLI[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(banner_text)
    console.print(panel)


def print_hf_style_help() -> None:
    """Prints a structured help menu formatted identically to huggingface-cli."""
    print_banner()

    if not HAS_RICH or console is None:
        print("\nUsage: neuralops [OPTIONS] <command> [ARGS]...\n")
        print("Commands:")
        for grp in SUBCOMMAND_GROUPS:
            print(f"\n  {grp['title']}:")
            for cmd, desc in grp["commands"]:
                print(f"    {cmd:18s} {desc}")
        print("\nRun 'neuralops <command> --help' for command-specific flags.\n")
        return

    console.print("\n[bold yellow]Usage:[/bold yellow] [bold green]neuralops[/bold green] [dim][OPTIONS][/dim] [bold cyan]<command>[/bold cyan] [dim][ARGS]...[/dim]\n")

    for grp in SUBCOMMAND_GROUPS:
        table = Table(title=f"[bold magenta]{grp['title']}[/bold magenta]", show_header=True, header_style="bold blue", border_style="dim", expand=True)
        table.add_column("Command", style="bold cyan", width=20)
        table.add_column("Description", style="white")

        for cmd, desc in grp["commands"]:
            table.add_row(cmd, desc)
        console.print(table)
        console.print()

    # Examples & hints footer
    examples = (
        "[bold green]$[/bold green] neuralops compress --model gpt2 --method hadamard_incoherent_svd --rank 32\n"
        "[bold green]$[/bold green] neuralops generate --model gpt2 --prompt 'NeuralOps accelerates' --draft-k 4\n"
        "[bold green]$[/bold green] neuralops niah --model gpt2 --context-lengths 512,1024,2048\n"
        "[bold green]$[/bold green] neuralops benchmark-fused --kernel all\n"
        "[bold green]$[/bold green] neuralops run --module svd_linear_attention --model gpt2 --dataset sst2"
    )
    console.print(Panel(examples, title="[bold yellow]Quickstart Examples[/bold yellow]", border_style="yellow"))
    console.print("[dim]Use [bold cyan]neuralops <command> --help[/bold cyan] for full arguments and options on each subcommand.[/dim]\n")


def run_interactive_ui() -> None:
    """Interactive terminal UI for exploring modules, compression, and benchmarks."""
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

    print_banner()
    if not HAS_RICH or console is None:
        print("Interactive mode requires 'rich'. Falling back to standard help.")
        print_hf_style_help()
        return

    menu = Table(title="[bold green]NeuralOps Interactive Terminal Control[/bold green]", show_header=True, header_style="bold magenta", border_style="cyan")
    menu.add_column("Option", style="bold yellow", width=10)
    menu.add_column("Action", style="bold white")

    menu.add_row("1", "Browse Attention Zoo (131 modules)")
    menu.add_row("2", "Browse Compression Engines (72 modules)")
    menu.add_row("3", "Run Fused Triton Kernel Profiler")
    menu.add_row("4", "Run System & Hardware Diagnostics (Doctor)")
    menu.add_row("5", "Display Benchmark Sweeps Summary")
    menu.add_row("q", "Exit")

    console.print(menu)
    try:
        choice = input("\nSelect an action [1-5, q]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "1":
        table = Table(title="[bold cyan]NeuralOps Attention Mechanisms Zoo (131 Modules)[/bold cyan]", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Attention Module", style="bold green")
        table.add_column("Causal", justify="center")
        table.add_column("GQA", justify="center")
        table.add_column("RoPE", justify="center")

        for idx, (name, cls) in enumerate(sorted(ATTENTION_REGISTRY.items()), 1):
            causal = "✓" if getattr(cls, "supports_causal", True) else "✗"
            gqa = "✓" if getattr(cls, "supports_gqa", False) else "✗"
            rope = "✓" if getattr(cls, "supports_rope", False) else "✗"
            table.add_row(str(idx), name, causal, gqa, rope)
        console.print(table)

    elif choice == "2":
        table = Table(title="[bold cyan]NeuralOps Compression Engines (72 Modules)[/bold cyan]", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Compression Engine", style="bold green")
        table.add_column("Type", style="cyan")

        for idx, (name, cls) in enumerate(sorted(COMPRESSOR_REGISTRY.items()), 1):
            c_type = "SVD / Low-Rank" if "svd" in name or "low_rank" in name else "Quant / Pruning"
            table.add_row(str(idx), name, c_type)
        console.print(table)

    elif choice == "3":
        from neuralops.benchmarks.cli import handle_benchmark_fused
        import argparse
        handle_benchmark_fused(argparse.Namespace(kernel="all", batch_size=4, seq_len=128, dim=768))

    elif choice == "4":
        from neuralops.cli.doctor import cmd_doctor
        cmd_doctor()

    elif choice == "5":
        from neuralops.benchmarks.cli import handle_results
        import argparse
        handle_results(argparse.Namespace(output_dir="results/roberta_benchmark_sweep"))

    elif choice == "q":
        console.print("[dim]Exiting NeuralOps CLI. Happy hacking![/dim]")
        return
