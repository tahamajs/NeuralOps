#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""CLI command for an OpenAI-compatible Hugging Face generation server."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def build_serve_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops serve subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops serve",
        description="Launch a basic OpenAI-compatible server backed by a Hugging Face causal language model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Target model identifier or local checkpoint path to serve",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host network interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Network port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--engine",
        choices=["fastapi"],
        default="fastapi",
        help="OpenAI-compatible FastAPI server (the only currently implemented HTTP engine)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for client authentication. Can also be set via NEURALOPS_API_KEY env var.",
    )
    parser.add_argument(
        "--kv-cache-budget-gb",
        type=float,
        default=4.0,
        help="KV-cache virtual memory budget in gigabytes (default: 4.0)",
    )
    return parser


def _load_serving_model(model_name_or_path: str, device: str) -> Any:
    """Load the requested pretrained causal LM without substituting random weights."""
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pretrained causal language model {model_name_or_path!r}; "
            "the serving command will not start with random placeholder weights."
        ) from exc


def cmd_serve(args: Any) -> int:
    """Load a pretrained causal LM and run its OpenAI-compatible HTTP adapter."""
    if isinstance(args, dict):
        args_ns = argparse.Namespace(**args)
    else:
        args_ns = args

    model_id = getattr(args_ns, "model", "gpt2")
    host = getattr(args_ns, "host", "127.0.0.1")
    port = getattr(args_ns, "port", 8000)
    engine_type = getattr(args_ns, "engine", "fastapi")
    api_key = getattr(args_ns, "api_key", None) or os.environ.get("NEURALOPS_API_KEY")
    kv_cache_budget_gb = getattr(args_ns, "kv_cache_budget_gb", 4.0)

    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("--port must be an integer between 1 and 65535")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("--host must be a non-empty network interface name or address")
    if engine_type != "fastapi":
        raise ValueError("Only the FastAPI serving engine is implemented by this command.")
    if kv_cache_budget_gb <= 0:
        raise ValueError("--kv-cache-budget-gb must be a positive float")

    if host not in ("127.0.0.1", "localhost") and not api_key:
        raise ValueError(
            "Refusing to bind serving to non-localhost address without an API key configured (set NEURALOPS_API_KEY)."
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    try:
        import uvicorn
        from transformers import AutoTokenizer

        from neuralops.integrations.serving import create_openai_serving_app
    except ImportError as exc:
        raise RuntimeError(
            "Serving requires the optional dependencies FastAPI, Uvicorn, and Transformers. "
            "Install them before running `neuralops serve`."
        ) from exc

    # Banner
    if HAS_RICH and console is not None:
        table = Table(title="NeuralOps OpenAI-Compatible Server", header_style="bold green")
        table.add_column("Component", style="cyan", width=26)
        table.add_column("Configuration", style="bold yellow", min_width=35)
        table.add_row("Primary Model", str(model_id))
        table.add_row("Serving Engine", "FastAPI / Hugging Face generate")
        table.add_row("Listening Host / Port", f"{host}:{port}")
        table.add_row("Execution Hardware", device.upper())
        table.add_row("KV Cache Budget", f"{kv_cache_budget_gb:.1f} GB")
        table.add_row("Telemetry Endpoints", "/health, /metrics, /v1/telemetry")
        console.print(table)
    else:
        print("=================================================================")
        print(f"NeuralOps OpenAI-Compatible Server: {model_id}")
        print(f"Engine: FastAPI / Hugging Face generate | Device: {device}")
        print(f"Host: {host}:{port} | KV Cache Budget: {kv_cache_budget_gb:.1f} GB")
        print("Telemetry Endpoints: /health, /metrics, /v1/telemetry")
        print("=================================================================")

    # Load the model and tokenizer before binding the listening socket.
    print(f"[*] Loading primary model '{model_id}' on {device}...")
    primary_model = _load_serving_model(model_id, device=device)
    primary_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    app = create_openai_serving_app(
        primary_model,
        tokenizer,
        model_name=model_id,
        api_key=api_key,
        kv_cache_budget_gb=kv_cache_budget_gb,
    )
    print(f"[*] Starting HTTP listener at http://{host}:{port} (Ctrl+C to stop).")
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    parser = build_serve_parser()
    cli_args = parser.parse_args()
    sys.exit(cmd_serve(cli_args))
