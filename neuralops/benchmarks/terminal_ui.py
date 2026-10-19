# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Rich Terminal Interface & Interactive Control Center (huggingface-cli / gh style)

Provides colorized banner, interactive module explorer, grouped help menus, progress meters, interactive module inspector, guided wizard, search, and formatted tables for NeuralOps.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional
from neuralops.benchmarks.discovery import CATALOG_PATH

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
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
        "title": "Model Surgery, Composition & Compression",
        "commands": [
            ("surgery", "1-line surgical attention replacement, Ridge calibration, and micro-healing (alias: transplant)"),
            ("compose", "Declarative algebraic builder for custom attention via ComposableAttention"),
            ("recommend", "Hardware-aware advisor recommending optimal attention & compression recipes"),
            ("calibrate", "Closed-form layerwise Ridge/subspace output alignment for surgical models"),
            ("compress", "Compress any model via SVD, Hadamard Incoherent SVD, W4A16, or registered engines"),
            ("search", "AutoSVD Pareto optimal rank allocation via Fisher & Gram sensitivity"),
            ("export", "Export compressed/surgerized models to safetensors, ONNX, or TorchScript"),
        ],
    },
    {
        "title": "Inference Acceleration & Serving Runtime",
        "commands": [
            ("serve", "Launch high-throughput ContinuousBatcher & PagedKVCache serving runtime"),
            ("generate", "Run lossless generation with self-speculative draft-and-verify decoding"),
            ("niah", "Execute Needle-In-A-Haystack passkey benchmark & generate 2D heatmaps"),
            ("benchmark-fused", "Benchmark fused Triton kernels (W4A16 dequant-GEMM, O(1) state step)"),
        ],
    },
    {
        "title": "Distillation, Benchmarking & Training Suite",
        "commands": [
            ("distill", "Execute 6-pillar multi-level attention & logit knowledge distillation"),
            ("benchmark", "Run empirical benchmark suite with Rich telemetry across models & datasets (alias: run)"),
            ("plan", "Dry-run and compute estimated compute hours, memory, and combinations"),
            ("results", "Inspect, summarize, and display benchmark results tables (alias: ledger)"),
            ("resume", "Resume an interrupted empirical benchmark sweep from state journal"),
        ],
    },
    {
        "title": "System Diagnostics & Telemetry Studio",
        "commands": [
            ("monitor", "Launch real-time telemetry, benchmark database & Web Documentation Studio (:8080)"),
            ("doctor", "Run comprehensive health diagnostics on GPU, Triton, and certified registry counts"),
            ("info", "Print detailed GPU hardware telemetry and registry metrics"),
            ("inspect", "UNIX-style architectural specification sheet for any registered module"),
            ("roofline", "Hardware Roofline & Arithmetic Intensity Analyzer for compiled Triton kernels"),
            ("validate", "Automated 6-point contract diagnostic check for any module"),
            ("scaffold", "Generate production-ready boilerplate for new custom attention or compressor"),
            ("list", "List and filter 131 attention mechanisms and 72 compressors"),
            ("verify", "Fast environment sanity check (PyTorch, CUDA, MPS)"),
            ("ui", "Launch minimalist flat two-pane terminal inspector (fzf-style)"),
        ],
    },
]

HELP_TOPICS = {
    "quickstart": """# NeuralOps Quickstart Guide
NeuralOps is a modular, high-performance library for Subquadratic Attention and Deep SVD Compression.

1. **Explore Modules**: `neuralops list --type attention --supports causal`
2. **Compress a Model**: `neuralops compress --model gpt2 --method hadamard_incoherent_svd --rank 32`
3. **Run Speculative Generation**: `neuralops generate --model gpt2 --prompt "NeuralOps is" --draft-k 4`
4. **Benchmark on Datasets**: `neuralops run --module svd_linear_attention --model gpt2 --dataset sst2`
""",
    "compression": """# Model Compression Guide
NeuralOps features **72 Compression Engines** spanning:
- **Hadamard Incoherent SVD**: Orthogonal Fast Walsh-Hadamard Transform eliminates activation/weight outliers before low-rank truncation.
- **Bit-Packed W4A16**: Packs two 4-bit weights per byte in uint8, dynamically dequantized in-SRAM via fused Triton GEMM.
- **AutoSVD Pareto Search**: Evaluates Fisher & Gram information matrices to allocate non-uniform layer ranks optimally under a target compression ratio.
""",
    "attention": r"""# Attention Zoo (131 Mechanisms)
Categorized by mathematical formulation:
- **Linear Attention**: $O(N)$ recurrence via kernel feature maps ($\phi(q) \sum \phi(k)^T v$).
- **Gated State-Space & Delta Rule**: Selective scans, Gated DeltaNet, HGRN2, and Mamba-1/2 blocks.
- **Hybrid & Local**: Multi-scale sliding window + low-rank residual routing.
- **Decay & Recurrence**: Hardware-aligned fused `@triton.jit` recurrent state updates.
""",
    "triton": """# Triton Kernels (Comprehensive Fused Hardware Kernels)
- **low_rank.py**: Bit-packed W4A16 GEMM and low-rank factorized linear forward.
- **state_update.py**: $O(1)$ single-token autoregressive recurrent state update in-SRAM.
- **selective_scan.py**: Mamba selective continuous-time discretization scan.
- **fused_losses.py**: CrossEntropy + KL divergence fused loss kernel avoiding logit materialization.
""",
}


def print_banner() -> None:
    """Prints the styled NeuralOps CLI banner."""
    if not HAS_RICH or console is None:
        print("================================================================")
        print("                   NeuralOps CLI (v1.4.0)                       ")
        print("     High-Performance Attention Zoo, SVD Compression & Triton   ")
        print("================================================================")
        return

    banner_text = Text(BANNER_ART, style="bold cyan")
    tagline = Text("NeuralOps v1.4.0 — High-Performance Attention Zoo, SVD Compression & Fused Triton Kernels\n", style="bold white")
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

    examples = (
        "[bold green]$[/bold green] neuralops compress --model gpt2 --method hadamard_incoherent_svd --rank 32\n"
        "[bold green]$[/bold green] neuralops generate --model gpt2 --prompt 'NeuralOps accelerates' --draft-k 4\n"
        "[bold green]$[/bold green] neuralops niah --model gpt2 --context-lengths 512,1024,2048\n"
        "[bold green]$[/bold green] neuralops benchmark-fused --kernel all\n"
        "[bold green]$[/bold green] neuralops run --module svd_linear_attention --model gpt2 --dataset sst2\n"
        "[bold green]$[/bold green] neuralops ui    [dim]# Launch full interactive TUI mode[/dim]"
    )
    console.print(Panel(examples, title="[bold yellow]Quickstart Examples[/bold yellow]", border_style="yellow"))
    console.print("[dim]Use [bold cyan]neuralops <command> --help[/bold cyan] for full arguments and options on each subcommand.[/dim]\n")


def _interactive_module_inspector(mod_name: str, mod_type: str = "attention") -> None:
    """Detailed visual inspector for a single module."""
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

    reg = ATTENTION_REGISTRY if mod_type == "attention" else COMPRESSOR_REGISTRY
    cls = reg.get(mod_name)
    if cls is None:
        console.print(f"[bold red]Module '{mod_name}' not found in {mod_type} registry.[/bold red]")
        return

    import inspect
    doc = inspect.getdoc(cls) or "No docstring provided."
    src_file = inspect.getfile(cls)

    details = Text()
    details.append(f"Name:          {mod_name}\n", style="bold cyan")
    details.append(f"Type:          {mod_type}\n", style="bold yellow")
    details.append(f"Class:         {cls.__name__}\n", style="bold green")
    details.append(f"File Location: {src_file}\n\n", style="dim")
    details.append("Documentation:\n", style="bold white")
    details.append(doc)

    console.print(Panel(details, title=f"[bold green]Inspector: {mod_name}[/bold green]", border_style="cyan"))


def _guided_wizard() -> None:
    """Step-by-step guided wizard for common operations."""
    console.print(Panel("[bold cyan]Welcome to the NeuralOps Guided Workflow Wizard![/bold cyan]\nAnswer a few prompts and NeuralOps will configure and run your task.", border_style="cyan"))

    task = Prompt.ask("\nWhat task would you like to perform?", choices=["compress", "generate", "benchmark", "diagnostics", "exit"], default="compress")
    if task == "exit":
        return

    if task == "compress":
        model_name = Prompt.ask("Enter base model name or path", default="gpt2")
        method = Prompt.ask("Choose compression engine", choices=["hadamard_incoherent_svd", "svd_linear_attention", "rotational_slicegpt_int4", "asvd", "fisher_svd"], default="hadamard_incoherent_svd")
        rank = IntPrompt.ask("Target rank per linear layer", default=32)
        save_dir = Prompt.ask("Save directory (leave empty to skip disk save)", default="")
        save_arg = save_dir if save_dir else None

        from neuralops.benchmarks.cli import handle_compress
        import argparse
        handle_compress(argparse.Namespace(model=model_name, method=method, rank=rank, save_dir=save_arg))

    elif task == "generate":
        model_name = Prompt.ask("Enter model name or path", default="gpt2")
        prompt_text = Prompt.ask("Enter prompt text", default="NeuralOps accelerates deep learning models by")
        draft_k = IntPrompt.ask("Draft candidate tokens per step (K)", default=4)
        temp = float(Prompt.ask("Sampling temperature (0.0 for greedy)", default="0.0"))

        from neuralops.benchmarks.cli import handle_generate
        import argparse
        handle_generate(argparse.Namespace(model=model_name, prompt=prompt_text, max_new_tokens=32, speculative=True, draft_k=draft_k, temperature=temp))

    elif task == "benchmark":
        mod = Prompt.ask("Enter attention or compressor module name", default="svd_linear_attention")
        model = Prompt.ask("Target model", default="gpt2")
        dataset = Prompt.ask("Target dataset", default="sst2")
        epochs = IntPrompt.ask("Fine-tuning epochs", default=1)
        samples = IntPrompt.ask("Max training samples", default=200)

        from neuralops.benchmarks.cli import handle_run
        import argparse
        handle_run(argparse.Namespace(
            module=mod, modules=None, no_compression=False,
            model=model, models=None, all_models=False,
            dataset=dataset, datasets=None, all_datasets=False,
            rank=64, ranks=None, device="cuda", epochs=epochs,
            batch_size=8, lr=2e-5, seq_len=64, parallel=1,
            output_dir="results/wizard_runs", timeout_s=1800,
            training_type="finetune", kd_teacher=None, kd_temperature=2.0, kd_alpha=0.5,
            recovery_steps=50, max_samples=samples, max_eval_samples=50, grad_accum_steps=1,
            tracker="none", project_name="neuralops", run_name=None, log_interval=10,
            distributed_backend="none", mixed_precision="no", no_trust_remote_code=False,
            hf_token=None, language=None, text_column=None, label_column=None, id_column=None,
            preprocess_script=None, config=None, auto_commit=False, no_auto_commit=True,
            commit_interval=1, push_to_hf=None, no_deduplicate=True, force_rerun=True,
        ))

    elif task == "diagnostics":
        from neuralops.cli.doctor import cmd_doctor
        cmd_doctor()


def _get_gpu_telemetry_panel() -> Optional[Panel]:
    """Retrieve fast live GPU telemetry via nvidia-smi if available."""
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.5,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 5:
                gpu_name, total_mem, used_mem, util, temp = parts[0], parts[1], parts[2], parts[3], parts[4]
                pct = int((float(used_mem) / float(total_mem)) * 100) if float(total_mem) > 0 else 0
                telemetry = Text()
                telemetry.append("GPU: ", style="bold cyan")
                telemetry.append(f"{gpu_name}   ", style="bold white")
                telemetry.append("VRAM: ", style="bold green")
                telemetry.append(f"{used_mem}MB / {total_mem}MB ({pct}%)   ", style="bold yellow")
                telemetry.append("Util: ", style="bold magenta")
                telemetry.append(f"{util}%   ", style="bold white")
                telemetry.append("Temp: ", style="bold red")
                telemetry.append(f"{temp}°C", style="bold white")
                return Panel(telemetry, title="[bold green]⚡ Live Hardware Telemetry[/bold green]", border_style="green", padding=(0, 1))
    except Exception:
        pass
    return None


def run_interactive_ui(inspect_target: Optional[str] = None, snapshot: bool = False) -> None:
    """Minimalist, flat UNIX-style terminal inspector and interactive control center."""
    from neuralops.core.inspector import inspect_architecture, render_architecture_card

    if snapshot:
        import torch
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Host"
        w = 118
        target = inspect_target or "gated_delta_net"
        try:
            spec = inspect_architecture(target)
            card_lines = render_architecture_card(spec).split("\n")
        except Exception as e:
            print(f"[-] Could not inspect '{target}': {e}")
            return

        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        all_attns = list(ATTENTION_REGISTRY.keys())
        all_comps = list(COMPRESSOR_REGISTRY.keys())
        all_modules = all_attns + all_comps

        target_idx = all_modules.index(target) if target in all_modules else 0
        left_pane_width = 34
        right_pane_width = 80
        total_rows = max(len(card_lines), 26)

        start_idx = max(0, min(target_idx - 10, len(all_modules) - total_rows))
        visible_modules = all_modules[start_idx : start_idx + total_rows]

        print("=" * w)
        print(f"  🏛️ NEURALOPS MINIMALIST TWO-PANE ARCHITECTURE EXPLORER")
        print(f"  Hardware Node: {dev_name}  │  131 Attentions · 72 Compressors · Unified Triton & CUDA Kernels")
        print("=" * w)
        print(f"  {'REGISTRY EXPLORER':<{left_pane_width}} │ {'14-FIELD ARCHITECTURE SPECIFICATION SHEET':<{right_pane_width}}")
        print("─" * (left_pane_width + 2) + "┼─" + "─" * right_pane_width)

        for row_idx in range(total_rows):
            if row_idx < len(visible_modules):
                m_name = visible_modules[row_idx]
                prefix = "> " if m_name == target else "  "
                tag = "attn" if m_name in ATTENTION_REGISTRY else "comp"
                left_entry = f"{prefix}{m_name} ({tag})"[:left_pane_width]
            else:
                left_entry = ""
            left_str = f"  {left_entry:<{left_pane_width}}"

            right_entry = card_lines[row_idx] if row_idx < len(card_lines) else ""
            right_str = right_entry[:right_pane_width]

            print(f"{left_str} │ {right_str}")

        print("=" * w)
        return

    if inspect_target:
        try:
            spec = inspect_architecture(inspect_target)
            card = render_architecture_card(spec)
            print(card)
        except Exception as e:
            print(f"[-] Could not inspect '{inspect_target}': {e}")
        return

    print_banner()
    if not HAS_RICH or console is None:
        print("Interactive mode requires 'rich'. Falling back to standard help.")
        print_hf_style_help()
        return

    while True:
        gpu_panel = _get_gpu_telemetry_panel()
        if gpu_panel is not None:
            console.print(gpu_panel)

        menu = Table(title="[bold green]NeuralOps Interactive Terminal Control Center[/bold green]", show_header=True, header_style="bold magenta", border_style="cyan", expand=True)
        menu.add_column("Key", style="bold yellow", width=6, justify="center")
        menu.add_column("Action", style="bold white", width=36)
        menu.add_column("Description", style="dim")

        menu.add_row("1", "Attention Zoo Explorer (131)", "Browse, filter by Causal/GQA/RoPE, and inspect attention modules")
        menu.add_row("2", "Compression Engines (72)", "Explore SVD, Hadamard Incoherent SVD, W4A16, and Quantizers")
        menu.add_row("3", "Search & Module Inspector", "Search any module by name or keyword and inspect its architecture")
        menu.add_row("4", "Run Guided Workflow Wizard", "Interactive step-by-step assistant for compression, generation, and benchmarks")
        menu.add_row("5", "Run Fused Triton Profiler", "Measure in-SRAM W4A16 dequant-GEMM and O(1) state step latency")
        menu.add_row("6", "Run System Diagnostics (Doctor)", "Deep verification of CUDA, SM capability, Triton cache, and VRAM")
        menu.add_row("7", "Inspect Benchmark Results", "View empirical benchmark metrics, accuracy gains, and loss curves")
        menu.add_row("8", "Help & Architecture Guides", "In-terminal documentation on SVD, Hadamard rotations, and Triton")
        menu.add_row("9", "Composable Attention Wizard", "Build custom attention via declarative algebraic primitives and generate Python code")
        menu.add_row("q", "Exit", "Close the NeuralOps Control Center")

        console.print(menu)
        try:
            choice = input("\nSelect an action [1-9, q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            table = Table(title="[bold cyan]NeuralOps Attention Mechanisms Zoo (131 Modules)[/bold cyan]", show_header=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Attention Module", style="bold green")
            table.add_column("Causal", justify="center")
            table.add_column("GQA", justify="center")
            table.add_column("RoPE", justify="center")

            import json
            attns = []
            if CATALOG_PATH.exists():
                try:
                    with open(CATALOG_PATH, "r", encoding="utf-8") as fp:
                        cat = json.load(fp)
                    attns = cat.get("attentions", [])
                except Exception:
                    pass

            if attns:
                for idx, item in enumerate(attns, 1):
                    causal = "[green]✓[/green]" if item.get("supports_causal", True) else "[red]✗[/red]"
                    gqa = "[green]✓[/green]" if item.get("supports_gqa", False) else "[dim]✗[/dim]"
                    rope = "[green]✓[/green]" if item.get("supports_rope", False) else "[dim]✗[/dim]"
                    table.add_row(str(idx), item["name"], causal, gqa, rope)
            else:
                from neuralops.core.registry import ATTENTION_REGISTRY
                for idx, (name, cls) in enumerate(sorted(ATTENTION_REGISTRY.items()), 1):
                    causal = "[green]✓[/green]" if getattr(cls, "supports_causal", True) else "[red]✗[/red]"
                    gqa = "[green]✓[/green]" if getattr(cls, "supports_gqa", False) else "[dim]✗[/dim]"
                    rope = "[green]✓[/green]" if getattr(cls, "supports_rope", False) else "[dim]✗[/dim]"
                    table.add_row(str(idx), name, causal, gqa, rope)

            console.print(table)
            input("\nPress Enter to return to main menu...")

        elif choice == "2":
            table = Table(title="[bold cyan]NeuralOps Compression Engines (72 Modules)[/bold cyan]", show_header=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Compression Engine", style="bold green")
            table.add_column("Category", style="cyan")

            import json
            comps = []
            if CATALOG_PATH.exists():
                try:
                    with open(CATALOG_PATH, "r", encoding="utf-8") as fp:
                        cat = json.load(fp)
                    comps = cat.get("compressors", [])
                except Exception:
                    pass

            if comps:
                for idx, item in enumerate(comps, 1):
                    name = item["name"]
                    c_type = "SVD / Low-Rank Factorization" if "svd" in name or "low_rank" in name else "Quantization / Pruning"
                    table.add_row(str(idx), name, c_type)
            else:
                from neuralops.core.registry import COMPRESSOR_REGISTRY
                for idx, (name, cls) in enumerate(sorted(COMPRESSOR_REGISTRY.items()), 1):
                    c_type = "SVD / Low-Rank Factorization" if "svd" in name or "low_rank" in name else "Quantization / Pruning"
                    table.add_row(str(idx), name, c_type)

            console.print(table)
            input("\nPress Enter to return to main menu...")

        elif choice == "3":
            query = Prompt.ask("\nEnter module name or keyword to search").strip().lower()
            import json
            attns, comps = [], []
            if CATALOG_PATH.exists():
                try:
                    with open(CATALOG_PATH, "r", encoding="utf-8") as fp:
                        cat = json.load(fp)
                    attns = [item["name"] for item in cat.get("attentions", [])]
                    comps = [item["name"] for item in cat.get("compressors", [])]
                except Exception:
                    pass
            if not attns:
                from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
                attns = list(ATTENTION_REGISTRY.keys())
                comps = list(COMPRESSOR_REGISTRY.keys())

            matches_attn = [k for k in attns if query in k.lower()]
            matches_comp = [k for k in comps if query in k.lower()]

            if not matches_attn and not matches_comp:
                console.print(f"[bold red]No modules found matching '{query}'.[/bold red]")
            else:
                res_table = Table(title=f"Search Results for '{query}'", show_header=True)
                res_table.add_column("Type", style="cyan", width=12)
                res_table.add_column("Module Name", style="bold green")
                for k in matches_attn:
                    res_table.add_row("attention", k)
                for k in matches_comp:
                    res_table.add_row("compressor", k)
                console.print(res_table)

                inspect_choice = Prompt.ask("\nEnter exact module name to inspect (or press Enter to skip)", default="").strip()
                if inspect_choice:
                    m_type = "attention" if inspect_choice in attns else "compressor"
                    _interactive_module_inspector(inspect_choice, mod_type=m_type)
            input("\nPress Enter to return to main menu...")

        elif choice == "4":
            _guided_wizard()
            input("\nPress Enter to return to main menu...")

        elif choice == "5":
            from neuralops.benchmarks.cli import handle_benchmark_fused
            import argparse
            handle_benchmark_fused(argparse.Namespace(kernel="all", batch_size=4, seq_len=128, dim=768))
            input("\nPress Enter to return to main menu...")

        elif choice == "6":
            from neuralops.cli.doctor import cmd_doctor
            cmd_doctor()
            input("\nPress Enter to return to main menu...")

        elif choice == "7":
            from neuralops.benchmarks.cli import handle_results
            import argparse
            dir_choice = Prompt.ask("Choose results directory", choices=["results/roberta_benchmark_sweep", "results/full_queue_all_modules", "results/roberta_compressors_sweep"], default="results/roberta_benchmark_sweep")
            handle_results(argparse.Namespace(output_dir=dir_choice))
            input("\nPress Enter to return to main menu...")

        elif choice == "8":
            console.print("\n[bold cyan]Available Architecture Documentation Topics:[/bold cyan]")
            for k in HELP_TOPICS.keys():
                console.print(f"  • [bold yellow]{k}[/bold yellow]")
            topic = Prompt.ask("\nChoose topic to read", choices=list(HELP_TOPICS.keys()) + ["back"], default="quickstart")
            if topic in HELP_TOPICS:
                console.print(Markdown(HELP_TOPICS[topic]))
            input("\nPress Enter to return to main menu...")

        elif choice == "9":
            console.print(Panel("[bold green]Composable Attention Architecture Wizard[/bold green]\nInteractively assemble custom attention layers and export ready-to-use Python code.", border_style="green"))
            proj = Prompt.ask("Choose projection (P)", choices=["dense", "svd(rank=32)", "svd(rank=64)", "conv1d(kernel_size=4)"], default="svd(rank=64)")
            pos = Prompt.ask("Choose positional embedding", choices=["none", "rope", "alibi"], default="rope")
            kmap = Prompt.ask("Choose feature map (phi)", choices=["softmax", "elu+1", "cosformer"], default="elu+1")
            msk = Prompt.ask("Choose masking (M)", choices=["causal", "sliding_window(128)", "sliding_window(128)+sink(4)", "bidirectional"], default="sliding_window(128)+sink(4)")
            eng = Prompt.ask("Choose execution engine (E)", choices=["sdpa", "chunked_scan", "delta_rule"], default="chunked_scan")

            from neuralops import compose_attention
            attn = compose_attention(projection=proj, positional=pos, kernel_map=kmap, mask=msk, engine=eng, embed_dim=768, num_heads=12)
            console.print("\n[bold cyan]Assembled Architecture Specification:[/bold cyan]")
            console.print(attn.summary())

            code = (
                "import torch\n"
                "from neuralops import compose_attention\n\n"
                f"custom_attn = compose_attention(\n"
                f"    embed_dim=768,\n"
                f"    num_heads=12,\n"
                f"    projection='{proj}',\n"
                f"    positional='{pos}',\n"
                f"    kernel_map='{kmap}',\n"
                f"    mask='{msk}',\n"
                f"    engine='{eng}',\n"
                ")\n"
                "# Forward pass\n"
                "x = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)\n"
                "out = custom_attn(x)\n"
            )
            console.print("\n[bold yellow]Generated Python Snippet:[/bold yellow]")
            if HAS_RICH and console is not None:
                console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
            else:
                print(code)
            input("\nPress Enter to return to main menu...")

        elif choice == "q":
            console.print("[dim]Exiting NeuralOps Control Center. Happy hacking![/dim]")
            break
