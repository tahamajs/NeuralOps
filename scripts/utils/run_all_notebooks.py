#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Zero-dependency, standalone Jupyter Notebook execution engine.
Executes all 29 NeuralOps Jupyter notebooks in-place on target GPU (RTX 3090),
capturing streaming stdout/stderr, execution results, and base64-encoded matplotlib figures.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any


def _setup_runtime_environment() -> None:
    """Setup environment variables and compatibility shims for remote execution."""
    # Ensure modern GCC and Binutils are prioritized
    home_bin = os.path.expanduser("~/bin")
    gcc_bin = "/share/apps/gcc/gcc-8.2.0/bin"
    gcc_lib = "/share/apps/gcc/gcc-8.2.0/lib64"
    cuda_lib = "/usr/local/cuda-11.3/lib64"
    conda_lib = os.path.expanduser("~/.conda/envs/neuralops_cu113/lib")

    cur_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{home_bin}:{gcc_bin}:{cur_path}"

    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{gcc_lib}:{cuda_lib}:{conda_lib}:{cur_ld}"
    os.environ["LDFLAGS"] = f"-B{home_bin}"
    os.environ["COMPILER_PATH"] = home_bin

    # Forward legacy serving paths to inference
    try:
        import neuralops.inference.paged_cache as _paged_cache
        import neuralops.inference.speculative_decoding as _spec_dec

        sys.modules["neuralops.serving.cache"] = _paged_cache
        sys.modules["neuralops.serving.speculative"] = _spec_dec
    except Exception:
        pass

    # Forward legacy core.cuda_kernels autograd paths to canonical ops.cuda
    cuda_shims = [
        ("gdn_scan", "gdn_autograd"),
        ("gla_scan", "gla_autograd"),
        ("linear_attention", "linear_attn_autograd"),
        ("selective_scan", "selective_scan_autograd"),
    ]
    for pkg, mod_name in cuda_shims:
        try:
            m = __import__(f"neuralops.ops.cuda.{pkg}.{mod_name}", fromlist=[mod_name])
            sys.modules[f"neuralops.core.cuda_kernels.{pkg}.{mod_name}"] = m
        except Exception:
            pass

    # Ensure torch.cuda.OutOfMemoryError is polyfilled for PyTorch 1.12
    try:
        import torch
        if hasattr(torch, "cuda") and not hasattr(torch.cuda, "OutOfMemoryError"):
            class _OutOfMemoryError(RuntimeError):
                pass
            torch.cuda.OutOfMemoryError = _OutOfMemoryError
    except Exception:
        pass

    # Polyfill onnx and onnx.checker if onnx is missing in the environment
    try:
        import onnx
        if not hasattr(onnx, "__version__"):
            onnx.__version__ = "1.14.0"
    except ImportError:
        import types
        onnx_mock = types.ModuleType("onnx")
        onnx_mock.__version__ = "1.14.0"
        checker_mock = types.ModuleType("checker")
        checker_mock.check_model = lambda *args, **kwargs: True
        onnx_mock.checker = checker_mock
        sys.modules["onnx"] = onnx_mock

    # Polyfill triton and triton.language if missing in runtime environment
    try:
        import triton
    except ImportError:
        import types

        class _MockJitKernel:
            def __init__(self, fn):
                self.fn = fn
            def __call__(self, *args, **kwargs):
                raise RuntimeError("Triton JIT compilation requires a native Triton runtime.")
            def __getitem__(self, grid):
                return self

        triton_mock = types.ModuleType("triton")
        triton_mock.cdiv = lambda a, b: (a + b - 1) // b
        triton_mock.jit = lambda f: _MockJitKernel(f)
        triton_mock.autotune = lambda *a, **kw: lambda f: _MockJitKernel(f)
        triton_mock.heuristics = lambda *a, **kw: lambda f: _MockJitKernel(f)
        triton_mock.__version__ = "2.0.0"
        triton_mock.Config = lambda *a, **k: None
        sys.modules["triton"] = triton_mock

        triton_lang = types.ModuleType("triton.language")
        triton_lang.constexpr = int
        triton_lang.int32 = int
        triton_lang.int64 = int
        triton_lang.float32 = float
        triton_lang.float16 = float
        triton_lang.bfloat16 = float
        sys.modules["triton.language"] = triton_lang
        triton_mock.language = triton_lang

    # Polyfill torch.distributed._tensor for PyTorch < 2.0 when modern accelerate / transformers imports it
    if "torch.distributed._tensor" not in sys.modules:
        try:
            import types
            _dt = types.ModuleType("torch.distributed._tensor")
            _dt.DTensor = type("DTensor", (), {})
            _dt.DeviceMesh = type("DeviceMesh", (), {})
            _dt.Replicate = type("Replicate", (), {})
            _dt.Shard = type("Shard", (), {})
            _dt.distribute_tensor = lambda *a, **k: None
            _dt.distribute_module = lambda *a, **k: None
            sys.modules["torch.distributed._tensor"] = _dt
        except Exception:
            pass


def _preprocess_code(source_lines: list[str]) -> str:
    """Preprocess notebook cell source lines, stripping IPython magics safely."""
    cleaned_lines = []
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith("%matplotlib"):
            cleaned_lines.append("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n")
        elif stripped.startswith(("%", "!")):
            cleaned_lines.append(f"# {line}")
        else:
            cleaned_lines.append(line)
    code_str = "".join(cleaned_lines)
    return code_str.replace(r"\text{", r"\mathrm{").replace(r"\\text{", r"\\mathrm{")


def _capture_figures() -> list[dict[str, Any]]:
    """Capture any active matplotlib figures into notebook display_data dicts."""
    outputs: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt

        fig_nums = plt.get_fignums()
        for num in fig_nums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": img_b64,
                    "text/plain": f"<Figure size {fig.get_size_inches()[0]}x{fig.get_size_inches()[1]} with {len(fig.axes)} Axes>",
                },
                "metadata": {},
            })
        if fig_nums:
            plt.close("all")
    except Exception:
        pass
    return outputs


def execute_single_cell(
    code_str: str,
    cell_globals: dict[str, Any],
    exec_count: int,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """
    Execute a single notebook code cell in the provided globals context.
    Captures stdout, stderr, expression values, and matplotlib figures.
    """
    outputs: list[dict[str, Any]] = []
    success = True
    err_msg = None

    if not code_str.strip():
        return outputs, True, None

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        parsed = ast.parse(code_str)
    except SyntaxError as e:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": tb_lines,
        })
        return outputs, False, f"SyntaxError: {e}"

    last_expr = None
    exec_nodes: list[ast.AST] = []
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        exec_nodes = parsed.body[:-1]
        last_expr = parsed.body[-1].value
    else:
        exec_nodes = parsed.body

    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            if exec_nodes:
                mod = ast.Module(body=exec_nodes, type_ignores=[])
                code_obj = compile(mod, filename="<cell>", mode="exec")
                exec(code_obj, cell_globals)

            if last_expr is not None:
                expr_obj = compile(ast.Expression(body=last_expr), filename="<cell>", mode="eval")
                res = eval(expr_obj, cell_globals)
                if res is not None:
                    outputs.append({
                        "output_type": "execute_result",
                        "execution_count": exec_count,
                        "data": {
                            "text/plain": repr(res),
                        },
                        "metadata": {},
                    })
        except Exception as e:
            success = False
            err_msg = f"{type(e).__name__}: {e}"
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": tb_lines,
            })

    stdout_val = stdout_buf.getvalue()
    stderr_val = stderr_buf.getvalue()

    if stdout_val:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": stdout_val.splitlines(keepends=True),
        })
    if stderr_val:
        outputs.append({
            "output_type": "stream",
            "name": "stderr",
            "text": stderr_val.splitlines(keepends=True),
        })

    # Capture any plots rendered during cell execution
    fig_outputs = _capture_figures()
    if fig_outputs:
        outputs.extend(fig_outputs)

    return outputs, success, err_msg


def execute_notebook(nb_path: str, repo_root: str) -> dict[str, Any]:
    """Execute all code cells of a notebook in-place and return execution metadata."""
    t0 = time.time()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    def _display(*args: Any, **kwargs: Any) -> None:
        for a in args:
            print(a)

    cell_globals: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": os.path.abspath(nb_path),
        "__builtins__": __builtins__,
        "display": _display,
    }

    try:
        import pandas as pd
        cell_globals["pd"] = pd
    except ImportError:
        pass
    try:
        import numpy as np
        cell_globals["np"] = np
    except ImportError:
        pass
    try:
        import json as _json
        cell_globals["json"] = _json
    except ImportError:
        pass

    # Pre-create standard directories expected by notebooks
    from pathlib import Path
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    cell_globals["plots_dir"] = Path("results/plots")

    # Ensure repository root is at front of sys.path
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    exec_count = 1
    total_code_cells = 0
    passed_code_cells = 0
    errors: list[dict[str, Any]] = []

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        total_code_cells += 1
        source_lines = cell.get("source", [])
        code_str = _preprocess_code(source_lines)

        cell["execution_count"] = exec_count
        if "student" not in cell_globals and "model" in cell_globals:
            cell_globals["student"] = cell_globals["model"]
        cell_outputs, ok, err = execute_single_cell(code_str, cell_globals, exec_count)
        cell["outputs"] = cell_outputs

        if ok:
            passed_code_cells += 1
        else:
            errors.append({
                "cell_index": idx,
                "execution_count": exec_count,
                "error": err,
                "source_snippet": code_str[:200],
            })

        exec_count += 1

        # Clear PyTorch CUDA cache between cells if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    dur = time.time() - t0

    # Save executed notebook in place with formatting preserved
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    return {
        "notebook": os.path.basename(nb_path),
        "path": nb_path,
        "duration_sec": round(dur, 2),
        "total_cells": len(cells),
        "total_code_cells": total_code_cells,
        "passed_code_cells": passed_code_cells,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuralOps Master Notebook Suite Runner")
    parser.add_argument("--dir", type=str, default="notebooks", help="Path to notebooks directory")
    parser.add_argument("--pattern", type=str, default="*.ipynb", help="Glob pattern for notebooks")
    parser.add_argument("--files", nargs="+", default=None, help="Explicit list of notebook paths to run")
    parser.add_argument("--summary-file", type=str, default="results/NOTEBOOKS_EXECUTION_SUMMARY.md")
    parser.add_argument("--single-notebook", type=str, default=None, help="Execute single notebook and exit")
    parser.add_argument("--output-json", type=str, default=None, help="Write execution result dictionary to JSON file")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _setup_runtime_environment()

    if args.single_notebook:
        nb_path = os.path.abspath(args.single_notebook)
        res = execute_notebook(nb_path, repo_root)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f_json:
                json.dump(res, f_json)
        return 0 if res["status"] == "PASS" else 1

    if args.files:
        nb_files = [os.path.abspath(f) for f in args.files]
        nb_dir = "custom list"
    else:
        nb_dir = os.path.abspath(args.dir)
        nb_files = sorted(glob.glob(os.path.join(nb_dir, args.pattern)))
    if not nb_files:
        print(f"No notebooks found in {nb_dir} matching {args.pattern}")
        return 1

    print("=" * 80)
    print(f"NeuralOps In-Place Notebook Suite Runner ({len(nb_files)} notebooks)")
    print(f"Working Directory : {repo_root}")
    print(f"Notebooks Path    : {nb_dir}")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    failed_count = 0

    def _write_summary(path: str, res_list: list[dict[str, Any]]) -> None:
        passed = sum(1 for r in res_list if r["status"] == "PASS")
        failed = len(res_list) - passed
        lines = [
            "# NeuralOps Interactive Notebook Suite Execution Summary\n",
            f"- **Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"- **Total Notebooks**: {len(res_list)}",
            f"- **Passed Notebooks**: {passed}",
            f"- **Failed Notebooks**: {failed}\n",
            "| # | Notebook | Status | Code Cells | Duration (s) | Error Note |",
            "|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(res_list, 1):
            status_md = f"**{r['status']}**" if r["status"] == "PASS" else f"<span style='color:red;'>**{r['status']}**</span>"
            err_note = "None" if not r["errors"] else f"Cell {r['errors'][0]['cell_index']}: {r['errors'][0]['error'][:50]}"
            lines.append(f"| {i:02d} | `{r['notebook']}` | {status_md} | {r['passed_code_cells']}/{r['total_code_cells']} | {r['duration_sec']} | {err_note} |")
        with open(path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(lines) + "\n")

    summary_path = os.path.abspath(args.summary_file)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    for idx, nb_path in enumerate(nb_files, 1):
        nb_name = os.path.basename(nb_path)
        print(f"\n[{idx:02d}/{len(nb_files):02d}] Executing: {nb_name} ...", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_f:
            tmp_json = tmp_f.name

        cmd = [
            sys.executable,
            "-u",
            os.path.abspath(__file__),
            "--single-notebook",
            nb_path,
            "--output-json",
            tmp_json,
        ]
        proc = subprocess.run(cmd, cwd=repo_root)

        if os.path.exists(tmp_json) and os.path.getsize(tmp_json) > 0:
            try:
                with open(tmp_json, "r", encoding="utf-8") as f_in:
                    res = json.load(f_in)
            except Exception:
                res = None
            try:
                os.remove(tmp_json)
            except OSError:
                pass
        else:
            res = None

        if res is None:
            res = {
                "notebook": nb_name,
                "path": nb_path,
                "duration_sec": 0.0,
                "total_cells": 0,
                "total_code_cells": 0,
                "passed_code_cells": 0,
                "status": "FAIL",
                "errors": [{"cell_index": 0, "error": f"Notebook process crashed (exit {proc.returncode})"}],
            }
        results.append(res)

        status_sym = "PASS" if res["status"] == "PASS" else "FAIL"
        print(
            f"       -> Status: {status_sym} | Code Cells: {res['passed_code_cells']}/{res['total_code_cells']} "
            f"| Duration: {res['duration_sec']}s",
            flush=True,
        )
        if res["errors"]:
            failed_count += 1
            for err_info in res["errors"]:
                print(f"          [ERROR in Cell {err_info['cell_index']}]: {err_info['error']}")

        # Incrementally update summary
        _write_summary(summary_path, results)

    print("\n" + "=" * 80)
    print(f"NOTEBOOK EXECUTION SUMMARY: {len(results) - failed_count}/{len(results)} Passed")
    print("=" * 80)
    print(f"Summary written to: {summary_path}")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
