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
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

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
    }
    
    # Pre-populate sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
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
            if line_strip.startswith("!") or line_strip.startswith("%"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        exec_code = "\n".join(clean_lines)
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cell_outputs = []
        status = "PASSED"
        
        t0 = time.perf_counter()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                exec(exec_code, exec_globals)
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
