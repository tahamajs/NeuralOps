#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Typed Exit Codes and Error Handling Hierarchy for NeuralOps CLI.

Defines standard machine-readable exit codes and structured error cards.
"""

from __future__ import annotations

from enum import IntEnum
import json
import sys
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    HAS_RICH = True
    err_console = Console(stderr=True)
except ImportError:
    HAS_RICH = False
    err_console = None


class ExitCode(IntEnum):
    """Standardized POSIX process exit codes for NeuralOps CLI."""

    SUCCESS = 0
    OK = 0
    FAILURE = 1
    INVALID_CONFIG = 2
    USAGE = 2
    USAGE_ERROR = 2
    BACKEND_UNAVAILABLE = 3
    NUMERICAL_GATE_FAILED = 4
    NUMERICAL_FAILURE = 4
    ARCHITECTURE_MISMATCH = 5
    PARTIAL = 5
    FILE_LOCK_FAILED = 6
    OUTPUT_FAILURE = 6
    GATE_VERIFICATION_FAILED = 21
    MODULE_NOT_FOUND = 22
    RUN_RESUME_MISMATCH = 23
    USER_INTERRUPT = 130
    INTERNAL = 70
    INTERNAL_ERROR = 70


class CliError(Exception):
    """Base exception for expected user-facing errors with structured diagnostic metadata."""

    def __init__(
        self,
        message: str,
        code: ExitCode = ExitCode.FAILURE,
        exit_code: ExitCode | None = None,
        hint: str | None = None,
        title: str = "Execution Error",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = exit_code if exit_code is not None else code
        self.exit_code = self.code
        self.hint = hint
        self.title = title
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize error diagnostic into a machine-readable dictionary."""
        data: dict[str, Any] = {
            "error_code": self.code.name,
            "exit_code": int(self.code),
            "title": self.title,
            "message": self.message,
        }
        if self.hint:
            data["hint"] = self.hint
        if self.context:
            data["context"] = self.context
        return data


class BackendUnavailableError(CliError):
    """Raised when an explicitly requested hardware accelerator or compiler extension is missing."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.BACKEND_UNAVAILABLE,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=exit_code,
            exit_code=exit_code,
            hint=hint or "Check device driver, CUDA installation, or run 'neuralops system verify'.",
            title="Hardware Backend Unavailable",
            context=context,
        )


class NumericalGateError(CliError):
    """Raised when mathematical parity, finite checks, or autograd gradients fail validation."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.NUMERICAL_FAILURE,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=exit_code,
            exit_code=exit_code,
            hint=hint or "Inspect tolerance budget or verify FP32 accumulation.",
            title="Numerical Gate Violation",
            context=context,
        )


class ConfigurationError(CliError):
    """Raised when configuration files, CLI arguments, or environment variables are invalid."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.USAGE,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=exit_code,
            exit_code=exit_code,
            hint=hint or "Verify configuration schema against 'neuralops benchmark plan'.",
            title="Configuration Schema Error",
            context=context,
        )


class ExecutionError(CliError):
    """Raised when a benchmark or training execution step fails."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.FAILURE,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=exit_code,
            exit_code=exit_code,
            hint=hint,
            title="Execution Failure",
            context=context,
        )


class ArchitectureMismatchError(CliError):
    """Raised when a model architecture cannot accept surgical transplantation or compression."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.ARCHITECTURE_MISMATCH,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=exit_code,
            exit_code=exit_code,
            hint=hint or "Verify that layer projection names match the required target module.",
            title="Architecture Incompatibility",
            context=context,
        )


class RunDirectoryLockedError(CliError):
    """Raised when another runner holds an active lock on the target output directory."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode = ExitCode.FILE_LOCK_FAILED,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=exit_code,
            exit_code=exit_code,
            hint=hint or "Use a different --output-dir or wait for the active process to finish.",
            title="Run Directory Locked",
            context=context,
        )


def emit_cli_error(error: Exception | str, json_mode: bool = False, exit_code: ExitCode = ExitCode.FAILURE) -> None:
    """Format and print structured error messages to stderr."""
    if isinstance(error, CliError):
        err_obj = error
    else:
        err_obj = CliError(str(error), code=exit_code)

    if json_mode:
        print(json.dumps({"error": err_obj.to_dict()}), file=sys.stderr)
        return

    if HAS_RICH and err_console is not None:
        content = f"[bold red]✗ {err_obj.message}[/bold red]"
        if err_obj.hint:
            content += f"\n\n[dim cyan]Suggestion:[/] {err_obj.hint}"
        panel = Panel(
            content,
            title=f"[bold red]{err_obj.title}[/bold red]",
            subtitle=f"[dim]Exit Code: {int(err_obj.code)} ({err_obj.code.name})[/dim]",
            border_style="red",
            padding=(0, 2),
        )
        err_console.print(panel)
    else:
        print(f"✗ Error ({err_obj.code.name}): {err_obj.message}", file=sys.stderr)
        if err_obj.hint:
            print(f"  Suggestion: {err_obj.hint}", file=sys.stderr)
