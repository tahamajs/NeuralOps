"""
NeuralOps Rich Terminal Interface & Interactive Control Center (huggingface-cli / gh style).
Provides colorized banner, interactive module explorer, grouped help menus,
progress meters, interactive module inspector, guided wizard, search, and formatted tables for NeuralOps.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional

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

HELP_TOPICS = {
    "quickstart": """# NeuralOps Quickstart Guide
NeuralOps is a modular, high-performance library for Subquadratic Attention and Deep SVD Compression.

### Basic Workflow
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
    "attention": """# Attention Zoo (131 Mechanisms)
Categorized by mathematical formulation:
- **Linear Attention**: $O(N)$ recurrence via kernel feature maps ($\phi(q) \sum \phi(k)^T v$).
- **Gated State-Space & Delta Rule**: Selective scans, Gated DeltaNet, HGRN2, and Mamba-1/2 blocks.
- **Hybrid & Local**: Multi-scale sliding window + low-rank residual routing.
- **Decay & Recurrence**: Hardware-aligned fused `@triton.jit` recurrent state updates.
""",
    "triton": """# Triton Kernels (26 Fused Kernels)
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


def run_interactive_ui() -> None:
    """Interactive terminal UI for exploring modules, compression, benchmarks, and help documentation."""
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

    print_banner()
    if not HAS_RICH or console is None:
        print("Interactive mode requires 'rich'. Falling back to standard help.")
        print_hf_style_help()
        return

    while True:
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
        menu.add_row("q", "Exit", "Close the NeuralOps Control Center")

        console.print(menu)
        try:
            choice = input("\nSelect an action [1-8, q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            table = Table(title="[bold cyan]NeuralOps Attention Mechanisms Zoo (131 Modules)[/bold cyan]", show_header=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Attention Module", style="bold green")
            table.add_column("Causal", justify="center")
            table.add_column("GQA", justify="center")
            table.add_column("RoPE", justify="center")

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

            for idx, (name, cls) in enumerate(sorted(COMPRESSOR_REGISTRY.items()), 1):
                c_type = "SVD / Low-Rank Factorization" if "svd" in name or "low_rank" in name else "Quantization / Pruning"
                table.add_row(str(idx), name, c_type)
            console.print(table)
            input("\nPress Enter to return to main menu...")

        elif choice == "3":
            query = Prompt.ask("\nEnter module name or keyword to search").strip().lower()
            matches_attn = [k for k in ATTENTION_REGISTRY.keys() if query in k.lower()]
            matches_comp = [k for k in COMPRESSOR_REGISTRY.keys() if query in k.lower()]

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
                    m_type = "attention" if inspect_choice in ATTENTION_REGISTRY else "compressor"
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

        elif choice == "q":
            console.print("[dim]Exiting NeuralOps Control Center. Happy hacking![/dim]")
            break
