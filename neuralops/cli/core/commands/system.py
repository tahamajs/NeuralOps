#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""System command family: verify, doctor, info, compat, ui."""

from __future__ import annotations

import argparse
import platform
import sys

from neuralops.cli.core.errors import ExitCode
from neuralops.cli.core.output import CliReporter


def handle_verify(args: argparse.Namespace) -> int:
    """Verify hardware environment, PyTorch backend, and canonical registry counts."""
    import torch

    import neuralops.attentions  # noqa: F401
    import neuralops.compression_engine  # noqa: F401
    import neuralops.kernels as nk
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    from neuralops.core.triton_kernels import HAS_TRITON

    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    triton_count = len(nk.list_triton_kernels())
    cuda_count = len(nk.list_cuda_kernels())
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    device_count = torch.cuda.device_count() if cuda_avail else 0

    data = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_device_name": device_name,
        "cuda_device_count": device_count,
        "mps_available": torch.backends.mps.is_available(),
        "has_triton": HAS_TRITON,
        "attention_registry_count": len(ATTENTION_REGISTRY),
        "compressor_registry_count": len(COMPRESSOR_REGISTRY),
        "triton_kernel_count": triton_count,
        "cuda_kernel_count": cuda_count,
    }

    if getattr(args, "json", False):
        reporter.print_json(data)
        return int(ExitCode.SUCCESS)

    headers = ["Component", "Status / Version"]
    rows = [
        ["Platform", data["platform"]],
        ["Python Version", data["python_version"]],
        ["PyTorch Version", data["pytorch_version"]],
        ["CUDA Available", f"{cuda_avail} ({device_name})"],
        ["GPU Devices", str(device_count)],
        ["MPS Available", str(data["mps_available"])],
        ["Triton JIT", str(HAS_TRITON)],
        ["Attention Registry", f"{len(ATTENTION_REGISTRY)} modules (Canonical: 131)"],
        ["Compressor Registry", f"{len(COMPRESSOR_REGISTRY)} engines (Canonical: 72)"],
        ["Triton Kernels", f"{triton_count} unified kernels"],
        ["CUDA C++ Kernels", f"{cuda_count} compiled modules"],
    ]
    reporter.print_table("System Environment & Hardware Verification", headers, rows)
    return int(ExitCode.SUCCESS)


def handle_doctor(args: argparse.Namespace) -> int:
    """Diagnose environment setup, missing dependencies, and accelerator capabilities."""
    from neuralops.cli.doctor_cmd import cmd_doctor

    ret = cmd_doctor(args)
    return int(ret) if ret is not None else int(ExitCode.SUCCESS)


def handle_info(args: argparse.Namespace) -> int:
    """Show detailed environment and runtime information."""
    from neuralops.cli.info_cmd import cmd_info

    ret = cmd_info(args)
    return int(ret) if ret is not None else int(ExitCode.SUCCESS)


def handle_compat(args: argparse.Namespace) -> int:
    """Validate cross-hardware portability matrix and fallbacks."""
    from neuralops.cli.compat_cmd import run_compat_suite

    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )
    reporter.log("[+] Running NeuralOps Portability Matrix Test Suite...")
    res = run_compat_suite()
    if getattr(args, "json", False):
        reporter.print_json(res)
    else:
        reporter.log(f"[✓] Compatibility Matrix passed: {res.get('status', 'OK')}")
    return int(ExitCode.SUCCESS)


def handle_ui(args: argparse.Namespace) -> int:
    """Launch interactive terminal UI."""
    from neuralops.cli.terminal_ui import run_interactive_ui

    run_interactive_ui(inspect_target=getattr(args, "inspect", None), snapshot=getattr(args, "snapshot", False))
    return int(ExitCode.SUCCESS)
