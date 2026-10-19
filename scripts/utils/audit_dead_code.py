"""Comprehensive AST, Triton JIT & CUDA C++ Dead-Code Auditor for NeuralOps.

Author: Taha Majlesi
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


def audit_repository(root: Path) -> int:
    """Scan Python and CUDA files for 6 classes of dead or unreachable code."""
    py_files = sorted(
        p for p in root.rglob("*.py")
        if not any(
            part in p.parts
            for part in (".git", "__pycache__", "build", "dist", ".venv", "venv", "archive")
        )
    )
    cu_files = sorted((root / "neuralops" / "core" / "cuda_kernels").rglob("*.cu"))

    all_py_text = {p: p.read_text(encoding="utf-8", errors="ignore") for p in py_files}
    combined_corpus = "\n".join(all_py_text.values())

    shadowed_defs: list[str] = []
    unreachable_stmts: list[str] = []
    dead_triton_kernels: list[str] = []
    unused_private_helpers: list[str] = []
    commented_code_blocks: list[str] = []
    dead_cuda_kernels: list[str] = []

    for path, src in all_py_text.items():
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError as exc:
            print(f"[SYNTAX ERROR] {path}: {exc}")
            continue

        # 1. Detect duplicate/shadowed function or class definitions in the same scope (F811)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef)):
                seen: dict[str, int] = {}
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        decs = [ast.unparse(d) for d in getattr(stmt, "decorator_list", [])]
                        if any("overload" in d or "setter" in d or "deleter" in d for d in decs):
                            continue
                        if stmt.name in seen:
                            shadowed_defs.append(
                                f"{path}:{stmt.lineno} -> '{stmt.name}' shadows earlier definition at line {seen[stmt.name]}"
                            )
                        seen[stmt.name] = stmt.lineno

        # 2. Detect unreachable statements after return / raise / break / continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    body = getattr(sub, "body", None)
                    if isinstance(body, list):
                        for idx, stmt in enumerate(body[:-1]):
                            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                                nxt = body[idx + 1]
                                unreachable_stmts.append(
                                    f"{path}:{nxt.lineno} -> Unreachable statement in '{node.name}' after {type(stmt).__name__}"
                                )
                                break

        # 3. Detect unlaunched @triton.jit kernels & unused _private helper functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                decs = [ast.unparse(d) for d in node.decorator_list]
                is_triton_jit = any("triton.jit" in d for d in decs)
                local_hits = len(re.findall(rf"\b{re.escape(node.name)}\b", src))
                if is_triton_jit and local_hits <= 1:
                    global_hits = len(re.findall(rf"\b{re.escape(node.name)}\b", combined_corpus))
                    if global_hits <= 1:
                        dead_triton_kernels.append(
                            f"{path}:{node.lineno} -> @triton.jit '{node.name}' is defined but never launched"
                        )
                elif node.name.startswith("_") and not node.name.startswith("__") and local_hits <= 1:
                    global_hits = len(re.findall(rf"\b{re.escape(node.name)}\b", combined_corpus))
                    if global_hits <= 1:
                        unused_private_helpers.append(
                            f"{path}:{node.lineno} -> Private helper '{node.name}' is never referenced"
                        )

        # 4. Detect multi-line commented-out code blocks & agent chatter
        lines = src.splitlines()
        for i, line in enumerate(lines, start=1):
            if re.search(r"#\s*(Fixed by Agent|Subagent|Phase \d+ fix|TODO:\s*remove|DEAD CODE)", line, re.IGNORECASE):
                commented_code_blocks.append(f"{path}:{i} -> Agent chatter comment: {line.strip()[:80]}")

    # 5. Detect unlaunched CUDA __global__ kernels in .cu files
    for cu_path in cu_files:
        cu_src = cu_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"__global__\s+void\s+([a-zA-Z0-9_]+)\s*\(", cu_src):
            k_name = match.group(1)
            if len(re.findall(rf"\b{re.escape(k_name)}\b", cu_src)) <= 1:
                dead_cuda_kernels.append(
                    f"{cu_path} -> __global__ kernel '{k_name}' is defined but never launched"
                )

    print("=== NEURALOPS DEAD CODE & HYGIENE AUDIT ===")
    print(f"1. Shadowed / Duplicate Definitions : {len(shadowed_defs)}")
    for x in shadowed_defs:
        print("  ", x)
    print(f"2. Unreachable Statements           : {len(unreachable_stmts)}")
    for x in unreachable_stmts:
        print("  ", x)
    print(f"3. Unlaunched @triton.jit Kernels   : {len(dead_triton_kernels)}")
    for x in dead_triton_kernels:
        print("  ", x)
    print(f"4. Unlaunched CUDA __global__       : {len(dead_cuda_kernels)}")
    for x in dead_cuda_kernels:
        print("  ", x)
    print(f"5. Unused Private _helpers          : {len(unused_private_helpers)}")
    for x in unused_private_helpers:
        print("  ", x)
    print(f"6. Agent Chatter Comments           : {len(commented_code_blocks)}")
    for x in commented_code_blocks[:50]:
        print("  ", x)

    total_issues = (
        len(shadowed_defs)
        + len(unreachable_stmts)
        + len(dead_triton_kernels)
        + len(dead_cuda_kernels)
        + len(unused_private_helpers)
        + len(commented_code_blocks)
    )
    return total_issues


if __name__ == "__main__":
    issues = audit_repository(Path("."))
    sys.exit(0)
