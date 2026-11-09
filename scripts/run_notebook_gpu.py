#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Stand-alone GPU Notebook Runner without jupyter/nbconvert dependencies.
Executes code cells in sequence on GPU-3090-1, collects outputs/errors,
and saves the executed notebook into results/notebooks_executed/.
"""
import sys
import os
import io
import json
import csv
import math
import itertools
import time
import types
import functools
import re
import linecache
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Ensure terminal encoding safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup project root and sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

# Set headless matplotlib backend before importing pyplot
os.environ["MPLBACKEND"] = "Agg"
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
except Exception:
    pass

# 1. Dependency Shims for cluster environments lacking propcache/aiohappyeyeballs
try:
    import propcache
    import propcache.api
except ImportError:
    mod_pc = types.ModuleType("propcache")
    mod_pc_api = types.ModuleType("propcache.api")
    mod_pc.cached_property = functools.cached_property
    mod_pc.under_cached_property = functools.cached_property
    mod_pc_api.cached_property = functools.cached_property
    mod_pc_api.under_cached_property = functools.cached_property
    sys.modules["propcache"] = mod_pc
    sys.modules["propcache.api"] = mod_pc_api

try:
    import aiohappyeyeballs
except ImportError:
    mod_ahe = types.ModuleType("aiohappyeyeballs")
    sys.modules["aiohappyeyeballs"] = mod_ahe

# 2. PyTorch & Matplotlib Compatibility Monkey-Patches
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Patch F.scaled_dot_product_attention if missing (PyTorch < 2.0)
    if not hasattr(F, "scaled_dot_product_attention"):
        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += attn_mask
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            if dropout_p > 0.0:
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value
        F.scaled_dot_product_attention = scaled_dot_product_attention

    # Patch nn.Transformer.generate_square_subsequent_mask for device kwarg
    _orig_gen_mask = nn.Transformer.generate_square_subsequent_mask
    def _safe_gen_mask(sz, device=None, dtype=None):
        try:
            if device is not None and dtype is not None:
                return _orig_gen_mask(sz, device=device, dtype=dtype)
            elif device is not None:
                return _orig_gen_mask(sz, device=device)
            return _orig_gen_mask(sz)
        except TypeError:
            m = _orig_gen_mask(sz)
            if device is not None:
                m = m.to(device)
            if dtype is not None:
                m = m.to(dtype)
            return m
    nn.Transformer.generate_square_subsequent_mask = staticmethod(_safe_gen_mask)

    # Patch torch.load for weights_only kwarg in older PyTorch
    _orig_load = torch.load
    def _safe_load(*args, **kwargs):
        try:
            return _orig_load(*args, **kwargs)
        except TypeError as e:
            if "weights_only" in str(e):
                kwargs.pop("weights_only", None)
                return _orig_load(*args, **kwargs)
            raise
    torch.load = _safe_load

except ImportError:
    pass

# Patch matplotlib mathtext parser to prevent ParseFatalException
try:
    import matplotlib
    import matplotlib.mathtext as mathtext

    _orig_parse = mathtext.MathTextParser.parse

    def _clean_latex_math(s):
        if not isinstance(s, str):
            return s
        s = re.sub(r'\\(?:text|mbox|mathrm|mathbf)\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\ge\b', r'\\geq', s)
        s = re.sub(r'\\le\b', r'\\leq', s)
        s = re.sub(r'\\[,:;!]', r' ', s)
        return s

    def _safe_parse(self, s, *args, **kwargs):
        try:
            return _orig_parse(self, s, *args, **kwargs)
        except Exception:
            clean_s = _clean_latex_math(s)
            try:
                return _orig_parse(self, clean_s, *args, **kwargs)
            except Exception:
                plain_s = s.replace("$", "").replace("\\", "")
                return _orig_parse(self, plain_s, *args, **kwargs)

    mathtext.MathTextParser.parse = _safe_parse
except ImportError:
    pass

class DummyIPython:
    def run_line_magic(self, *args, **kwargs):
        return None
    def system(self, *args, **kwargs):
        return None

def execute_notebook(nb_path, out_path):
    print("=" * 80)
    print("Executing Notebook: {}".format(os.path.basename(nb_path)))
    print("=" * 80)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    exec_globals = {
        "__name__": "__main__",
        "__file__": nb_path,
        "display": lambda *args, **kwargs: None,
        "get_ipython": lambda: DummyIPython(),
        "json": json,
        "csv": csv,
        "math": math,
        "itertools": itertools,
        "os": os,
        "sys": sys,
        "time": time,
        "cells": cells,
    }

    try:
        import matplotlib.pyplot as plt
        exec_globals["plt"] = plt
    except Exception:
        pass

    try:
        import numpy as np
        exec_globals["np"] = np
    except Exception:
        pass

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        exec_globals["torch"] = torch
        exec_globals["nn"] = nn
        exec_globals["F"] = F
        exec_globals["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        pass

    passed_cells = 0
    failed_cells = 0
    total_code_cells = 0

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        total_code_cells += 1
        source_lines = cell.get("source", [])
        if isinstance(source_lines, list):
            code_str = "".join(source_lines)
        else:
            code_str = str(source_lines)

        # Filter IPython magic commands
        clean_lines = []
        for line in code_str.split("\n"):
            line_strip = line.strip()
            if line_strip.startswith(("!", "%")):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        exec_code = "\n".join(clean_lines)

        # Register code in linecache so inspect.getsource works for Triton JIT & reflection
        cell_filename = f"<notebook_{os.path.basename(nb_path)}_cell_{idx+1}>"
        linecache.cache[cell_filename] = (
            len(exec_code),
            None,
            [line + "\n" for line in exec_code.splitlines()],
            cell_filename
        )

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cell_outputs = []
        status = "PASSED"

        t0 = time.perf_counter()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                code_obj = compile(exec_code, cell_filename, "exec")
                exec(code_obj, exec_globals)
            except Exception as e:
                status = "FAILED"
                failed_cells += 1
                err_msg = traceback.format_exc()
                stderr_buf.write(err_msg)
            else:
                passed_cells += 1
        dt = time.perf_counter() - t0

        out_txt = stdout_buf.getvalue()
        err_txt = stderr_buf.getvalue()

        if out_txt:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": out_txt.splitlines(True)
            })
        if err_txt:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": err_txt.splitlines(True)
            })

        cell["outputs"] = cell_outputs
        cell["execution_count"] = total_code_cells

        print("  Cell {:02d}/{} [{:6.2f}s] -> {}".format(idx + 1, len(cells), dt, status))
        if status == "FAILED":
            first_err_line = err_txt.strip().splitlines()[-1] if err_txt.strip() else "Unknown Error"
            first_err_line = first_err_line.encode("ascii", errors="replace").decode("ascii")
            print("     Error: {}".format(first_err_line))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

    print("-" * 80)
    print("Completed: {} | Code Cells: {} (Passed: {}, Failed: {})".format(
        os.path.basename(nb_path), total_code_cells, passed_cells, failed_cells
    ))
    print("Saved to: {}\n".format(out_path))
    return passed_cells, failed_cells

def main():
    if len(sys.argv) < 2:
        print("Usage: run_notebook_gpu.py <path_to_notebook.ipynb> [output_path]")
        sys.exit(1)

    nb_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "notebooks_executed")
        out_path = os.path.join(out_dir, os.path.basename(nb_path))

    execute_notebook(nb_path, out_path)

if __name__ == "__main__":
    main()
