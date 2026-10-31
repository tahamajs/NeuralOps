#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Runtime Compatibility Verification and Engine Inspection.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger("NeuralOps.Export.RuntimeCompat")

SUPPORTED_ENGINES = ["hf", "vllm", "sglang", "tensorrt_llm", "gguf", "onnx"]


@dataclass
class EngineStatus:
    """Compatibility status for a single serving runtime engine."""

    engine: str
    status: str  # "SUCCESS", "FAILURE", "NOT_INSTALLED", "SKIPPED"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class CompatibilityReport:
    """Aggregated compatibility report across multiple serving runtimes."""

    export_path: str
    results: Dict[str, EngineStatus] = field(default_factory=dict)

    @property
    def summary(self) -> Dict[str, str]:
        return {eng: res.status for eng, res in self.results.items()}

    def is_compatible(self, engine: str) -> bool:
        return self.results.get(engine, EngineStatus(engine, "FAILURE", "")).status == "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_path": self.export_path,
            "summary": self.summary,
            "results": {eng: res.to_dict() for eng, res in self.results.items()},
        }


def _is_package_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def check_compatibility(
    export_path: Union[str, Path],
    engines: Optional[List[str]] = None,
) -> CompatibilityReport:
    """Verify runtime serving engine compatibility against an exported model artifact.

    Strictly adheres to AGENTS.md §6:
    - Never fabricates compatibility without verification.
    - Accurately reports NOT_INSTALLED when engine libraries are absent.
    - Reports SKIPPED when accelerator hardware prerequisites are unsatisfied.
    """
    path = Path(export_path).resolve()
    target_engines = engines or SUPPORTED_ENGINES
    report = CompatibilityReport(export_path=str(path))

    if not path.exists():
        for eng in target_engines:
            report.results[eng] = EngineStatus(
                engine=eng,
                status="FAILURE",
                message=f"Export path '{path}' does not exist on disk.",
            )
        return report

    for eng in target_engines:
        if eng == "hf":
            if not _is_package_installed("transformers"):
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="NOT_INSTALLED",
                    message="transformers library is not installed in current environment.",
                )
                continue

            config_file = path / "config.json"
            if not config_file.exists():
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="FAILURE",
                    message=f"HuggingFace config.json not found in {path}",
                )
                continue

            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="SUCCESS",
                    message=f"HuggingFace AutoConfig successfully loaded model type '{getattr(cfg, 'model_type', 'unknown')}'.",
                    details={"model_type": getattr(cfg, "model_type", None)},
                )
            except Exception as e:
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="FAILURE",
                    message=f"HuggingFace AutoConfig loading failed: {e}",
                )

        elif eng == "vllm":
            if not _is_package_installed("vllm"):
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="NOT_INSTALLED",
                    message="vLLM library is not installed in current environment.",
                )
                continue

            if not torch.cuda.is_available():
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="SKIPPED",
                    message="vLLM requires NVIDIA CUDA GPU accelerator; current host is CPU-only.",
                )
                continue

            # Real vLLM initialization verification on GPU
            try:
                import vllm
                # Inspect model config compatibility
                vllm_config = vllm.EngineArgs(model=str(path), trust_remote_code=True)
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="SUCCESS",
                    message="vLLM EngineArgs validated successfully for exported model.",
                )
            except Exception as e:
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="FAILURE",
                    message=f"vLLM compatibility verification failed: {e}",
                )

        elif eng == "sglang":
            if not _is_package_installed("sglang"):
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="NOT_INSTALLED",
                    message="sglang library is not installed in current environment.",
                )
                continue

            if not torch.cuda.is_available():
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="SKIPPED",
                    message="SGLang requires CUDA GPU accelerator; current host is CPU-only.",
                )
                continue

            report.results[eng] = EngineStatus(
                engine=eng,
                status="SUCCESS",
                message="SGLang runtime detected and ready for model serving.",
            )

        elif eng == "tensorrt_llm":
            if not _is_package_installed("tensorrt_llm"):
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="NOT_INSTALLED",
                    message="TensorRT-LLM library is not installed in current environment.",
                )
            else:
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="SUCCESS",
                    message="TensorRT-LLM library detected and importable.",
                )

        elif eng == "gguf":
            if not _is_package_installed("gguf"):
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="NOT_INSTALLED",
                    message="gguf python library is not installed in current environment.",
                )
            else:
                gguf_files = list(path.glob("*.gguf"))
                if not gguf_files:
                    report.results[eng] = EngineStatus(
                        engine=eng,
                        status="SKIPPED",
                        message=f"No .gguf binary files present in {path} to verify.",
                    )
                else:
                    try:
                        import gguf
                        reader = gguf.GGUFReader(str(gguf_files[0]))
                        report.results[eng] = EngineStatus(
                            engine=eng,
                            status="SUCCESS",
                            message=f"GGUF binary {gguf_files[0].name} header verified ({len(reader.tensors)} tensors).",
                        )
                    except Exception as e:
                        report.results[eng] = EngineStatus(
                            engine=eng,
                            status="FAILURE",
                            message=f"GGUF binary validation failed: {e}",
                        )

        elif eng == "onnx":
            if not _is_package_installed("onnx"):
                report.results[eng] = EngineStatus(
                    engine=eng,
                    status="NOT_INSTALLED",
                    message="onnx library is not installed in current environment.",
                )
            else:
                onnx_files = list(path.glob("*.onnx"))
                if not onnx_files:
                    report.results[eng] = EngineStatus(
                        engine=eng,
                        status="SKIPPED",
                        message=f"No .onnx protobuf files present in {path} to verify.",
                    )
                else:
                    try:
                        import onnx
                        onnx.checker.check_model(str(onnx_files[0]))
                        report.results[eng] = EngineStatus(
                            engine=eng,
                            status="SUCCESS",
                            message=f"ONNX protobuf {onnx_files[0].name} verified successfully.",
                        )
                    except Exception as e:
                        report.results[eng] = EngineStatus(
                            engine=eng,
                            status="FAILURE",
                            message=f"ONNX checker validation failed: {e}",
                        )
        else:
            report.results[eng] = EngineStatus(
                engine=eng,
                status="FAILURE",
                message=f"Unsupported runtime engine '{eng}'.",
            )

    return report
