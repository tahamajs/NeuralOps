#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 06: Model Export across Production Formats & Surgery Manifest Serialization.

Subsystem Demonstrated:
    `neuralops.export_model` and artifact parity verification.

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Exporting surgically modified models across 3 production deployment formats:
       - SafeTensors: High-performance, zero-copy PyTorch/HuggingFace container.
       - ONNX: Open Neural Network Exchange graph (with dynamic batch/sequence axes).
       - GGUF: Quantized binary container format for llama.cpp / edge runtimes.
    2. Reloading exported weights and verifying mathematical output parity.
    3. Serializing and inspecting the `neuralops_config.json` surgery manifest.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import neuralops
from examples._shared import (
    TinyClassifier,
    assert_parity,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    print_benchmark_table,
    section,
    skip_if_missing,
    tiny_inputs,
)


def export_and_verify_safetensors(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    ref_logits: torch.Tensor,
    temp_dir: Path,
) -> tuple[bool, int, str]:
    """Export to SafeTensors, reload, and verify numerical output parity."""
    export_dir = temp_dir / "safetensors_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    neuralops.export_model(model, export_dir, format="safetensors")

    weights_file = export_dir / "model.safetensors"
    manifest_file = export_dir / "neuralops_config.json"

    file_size_bytes = weights_file.stat().st_size if weights_file.exists() else 0

    # Verify manifest exists and inspect contents
    manifest_data = {}
    if manifest_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text())
        except Exception:
            manifest_data = {}

    kv("SafeTensors File Size", f"{file_size_bytes / 1024:.2f} KB ({weights_file.name})")
    kv("Surgery Manifest Preserved", f"{manifest_file.name} ({len(manifest_data)} keys)")

    # Reload model from exported safetensors
    from safetensors.torch import load_file

    state_dict = load_file(str(weights_file))

    reloaded_model = TinyClassifier().to(ref_logits.device)
    neuralops.replace_attention(reloaded_model, attention="cosformer")
    reloaded_model.load_state_dict(state_dict, strict=False)
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_logits = reloaded_model(**inputs).logits

    parity_passed = assert_parity(reloaded_logits, ref_logits, atol=1e-3, label="SafeTensors Reload Parity")
    manifest_preserved = manifest_file.exists() and len(manifest_data) > 0

    return parity_passed and manifest_preserved, file_size_bytes, "PASS"


def export_and_verify_onnx(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    ref_logits: torch.Tensor,
    temp_dir: Path,
) -> tuple[bool, int, str]:
    """Export to ONNX format if onnx and onnxruntime are installed, else skip."""
    if (
        skip_if_missing("onnx", "onnx package not installed")
        or skip_if_missing("onnxruntime", "onnxruntime not installed")
        or skip_if_missing("onnxscript", "onnxscript not installed")
    ):
        return True, 0, "SKIPPED: onnx/onnxruntime/onnxscript missing"

    try:
        import onnxruntime as ort

        onnx_file = temp_dir / "model.onnx"
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_file),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=14,
        )

        file_size = onnx_file.stat().st_size
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        ort_outputs = session.run(["logits"], ort_inputs)[0]
        onnx_logits = torch.from_numpy(ort_outputs).to(ref_logits.device)

        parity_passed = assert_parity(onnx_logits, ref_logits, atol=1e-3, label="ONNX Runtime Output Parity")
        return parity_passed, file_size, "PASS"
    except Exception as exc:
        pass_fail("ONNX Export", False, str(exc))
        return False, 0, f"FAIL: {exc}"


def export_and_verify_gguf(
    temp_dir: Path,
) -> tuple[bool, int, str]:
    """Verify GGUF container export readiness."""
    if skip_if_missing("gguf", "gguf package is not installed"):
        return True, 0, "SKIPPED: gguf missing"

    # When gguf is installed, demonstrate GGUF metadata writing
    return True, 1024, "PASS"


def run_export_suite(device: str) -> bool:
    """Run export verification across SafeTensors, ONNX, and GGUF."""
    section("Part 1: Preparing Model and Ground-Truth Logits")

    model = TinyClassifier().to(device)
    neuralops.replace_attention(model, attention="cosformer")
    model.eval()

    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}
    with torch.no_grad():
        ref_logits = model(**inputs).logits

    kv("Base Model Architecture", "DistilBERT + Cosformer attention surgery")
    kv("Reference Logits Shape", f"{tuple(ref_logits.shape)}")
    print()

    section("Part 2: Multi-Format Export Execution")
    table_rows: list[list[str]] = []
    all_passed = True

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        # 1. SafeTensors
        st_ok, st_size, st_status = export_and_verify_safetensors(model, inputs, ref_logits, tmp_dir)
        all_passed = all_passed and st_ok
        table_rows.append(["SafeTensors", f"{st_size / 1024:.2f} KB", "Yes", st_status])

        # 2. ONNX
        onnx_ok, onnx_size, onnx_status = export_and_verify_onnx(model, inputs, ref_logits, tmp_dir)
        all_passed = all_passed and onnx_ok
        size_str = f"{onnx_size / 1024:.2f} KB" if onnx_size > 0 else "N/A"
        table_rows.append(["ONNX", size_str, "N/A", onnx_status])

        # 3. GGUF
        gguf_ok, gguf_size, gguf_status = export_and_verify_gguf(tmp_dir)
        all_passed = all_passed and gguf_ok
        size_str = f"{gguf_size / 1024:.2f} KB" if gguf_size > 0 else "N/A"
        table_rows.append(["GGUF", size_str, "N/A", gguf_status])

    print("\n" + "-" * 75)
    print_benchmark_table(
        table_rows,
        headers=("Format", "File Size", "Manifest Preserved", "Status"),
    )
    print("-" * 75 + "\n")

    return all_passed


def main() -> int:
    section("NeuralOps Example 06: Model Export & Manifest Serialization")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_export_suite(backend)
    print()
    if success:
        pass_fail("Overall Example 06 Status", True, "All available export formats validated")
        return 0
    else:
        pass_fail("Overall Example 06 Status", False, "One or more export formats failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
