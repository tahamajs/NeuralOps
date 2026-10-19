#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Execution test suite for all NeuralOps interactive Jupyter notebooks.

Verifies:
1. notebooks/01_quickstart_surgery_and_healing.ipynb
2. notebooks/02_roofline_and_triton_kernels.ipynb
3. notebooks/03_custom_plugins_and_extensibility.ipynb
"""

import json
import os
from pathlib import Path
import pytest
import torch


def _execute_notebook_cells(nb_path: Path):
    """Parse and execute code cells from an .ipynb file in sequence."""
    assert nb_path.exists(), f"Notebook not found: {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    exec_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        lines = cell.get("source", [])
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("!"):
                continue
            code_lines.append(line)

        code_str = "".join(code_lines)
        if not code_str.strip():
            continue

        try:
            exec(code_str, exec_globals)
        except Exception as e:
            pytest.fail(f"Execution failed in {nb_path.name} cell #{idx}:\nCode:\n{code_str}\nError: {e}")


def test_notebook_01_quickstart_surgery_and_healing():
    """Verify end-to-end execution of Notebook 01."""
    nb_path = Path(__file__).parent.parent / "notebooks" / "01_quickstart_surgery_and_healing.ipynb"
    _execute_notebook_cells(nb_path)


def test_notebook_02_roofline_and_triton_kernels():
    """Verify end-to-end execution of Notebook 02."""
    nb_path = Path(__file__).parent.parent / "notebooks" / "02_roofline_and_triton_kernels.ipynb"
    _execute_notebook_cells(nb_path)


def test_notebook_03_custom_plugins_and_extensibility():
    """Verify end-to-end execution of Notebook 03 and restore registry purity."""
    import neuralops
    from neuralops.core.extensibility import unregister_custom_module
    from neuralops.core.registry import ATTENTION_REGISTRY

    nb_path = Path(__file__).parent.parent / "notebooks" / "03_custom_plugins_and_extensibility.ipynb"
    try:
        _execute_notebook_cells(nb_path)
    finally:
        if "my_research_attn" in ATTENTION_REGISTRY:
            unregister_custom_module("my_research_attn")
        assert len(ATTENTION_REGISTRY) == 131, f"Expected 131 attentions, got {len(ATTENTION_REGISTRY)}"
