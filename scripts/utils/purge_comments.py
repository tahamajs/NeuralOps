#!/usr/bin/env python3
"""
Purge comments from Python source files safely using tokenize.

- Uses Python standard `tokenize` module.
- Preserves string literals (multiline strings, docstrings, hex colors, URLs, etc.).
- Preserves line 1 shebang and PEP 263 encoding declarations.
- Strips standalone comments and inline comments.
- Cleans up excess blank lines resulting from deleted full-line comments.
- Verifies AST and py_compile before saving any file.
"""

from __future__ import annotations

import ast
import glob
import io
import os
import py_compile
import re
import sys
import tempfile
import tokenize
from typing import Dict, List, Set, Tuple


def purge_comments_from_source(source: str, file_path: str = "<string>") -> Tuple[str, int, int]:
    """
    Remove all Python comments from source code safely.

    Returns:
        (cleaned_source, removed_comments_count, remaining_comments_count)
    """
    # Detect encoding
    raw_bytes = source.encode("utf-8")
    tokens = list(tokenize.tokenize(io.BytesIO(raw_bytes).readline))

    # Identify lines in multiline strings to avoid altering blank lines inside strings
    multiline_string_lines: Set[int] = set()
    for tok in tokens:
        if tok.type == tokenize.STRING:
            if tok.start[0] != tok.end[0]:
                for r in range(tok.start[0], tok.end[0] + 1):
                    multiline_string_lines.add(r)

    # Identify comments to remove
    comments_to_remove: Dict[int, int] = {}
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            # Preserve shebang on line 1
            if row == 1 and tok.string.startswith("#!"):
                continue
            # Preserve encoding declaration on line 1 or 2
            if row in (1, 2) and re.search(r"coding[:=]\s*([-\w.]+)", tok.string):
                continue
            if row not in comments_to_remove or col < comments_to_remove[row]:
                comments_to_remove[row] = col

    lines = source.splitlines(keepends=False)
    tagged_lines: List[Tuple[str, bool]] = []

    for i, line in enumerate(lines, start=1):
        is_str = i in multiline_string_lines
        if i in comments_to_remove:
            col = comments_to_remove[i]
            prefix = line[:col]
            if prefix.strip() == "":
                # Standalone full-line comment -> omit completely
                continue
            else:
                # Inline comment -> keep code before comment
                tagged_lines.append((prefix.rstrip(), is_str))
        else:
            tagged_lines.append((line, is_str))

    # Clean up excess blank lines resulting from removed comments
    final_lines: List[str] = []
    i = 0
    n = len(tagged_lines)
    while i < n:
        line, is_str = tagged_lines[i]
        if line.strip() == "" and not is_str:
            j = i
            while j < n and tagged_lines[j][0].strip() == "" and not tagged_lines[j][1]:
                j += 1
            count = j - i
            prev_line = final_lines[-1] if final_lines else ""
            next_line = tagged_lines[j][0] if j < n else ""

            prev_indent = len(prev_line) - len(prev_line.lstrip()) if prev_line else 0
            next_indent = len(next_line) - len(next_line.lstrip()) if next_line else 0

            # If immediately following a block definition line ending in ':'
            if prev_line.strip().endswith(":") and (
                prev_line.strip().startswith("def ")
                or prev_line.strip().startswith("class ")
                or prev_line.strip().startswith("async def ")
                or prev_line.strip().startswith("if ")
                or prev_line.strip().startswith("elif ")
                or prev_line.strip().startswith("else:")
                or prev_line.strip().startswith("try:")
                or prev_line.strip().startswith("except")
                or prev_line.strip().startswith("finally:")
            ):
                max_allowed = 0
            elif prev_indent > 0 and next_indent > 0:
                # Inside indented blocks: maximum 1 blank line
                max_allowed = 1
            else:
                # Top level: maximum 2 blank lines
                max_allowed = 2

            num_to_keep = min(count, max_allowed)
            final_lines.extend([""] * num_to_keep)
            i = j
        else:
            final_lines.append(line)
            i += 1

    # Preserve shebang or encoding at the very start
    shebang_or_encoding: List[str] = []
    while final_lines and (
        final_lines[0].startswith("#!")
        or ("coding" in final_lines[0] and final_lines[0].startswith("#"))
    ):
        shebang_or_encoding.append(final_lines.pop(0))

    # Trim leading blank lines
    while final_lines and final_lines[0].strip() == "":
        final_lines.pop(0)

    # Trim trailing blank lines
    while final_lines and final_lines[-1].strip() == "":
        final_lines.pop()

    cleaned = "\n".join(shebang_or_encoding + final_lines)
    if cleaned:
        cleaned += "\n"

    # AST syntax verification
    ast.parse(cleaned, filename=file_path)

    # py_compile verification
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(cleaned)
        tmp_name = tmp.name

    try:
        py_compile.compile(tmp_name, doraise=True)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # Count remaining comments
    cleaned_tokens = list(tokenize.tokenize(io.BytesIO(cleaned.encode("utf-8")).readline))
    actual_rem = []
    for t in cleaned_tokens:
        if t.type == tokenize.COMMENT:
            if t.start[0] == 1 and t.string.startswith("#!"):
                continue
            if t.start[0] in (1, 2) and re.search(r"coding[:=]\s*([-\w.]+)", t.string):
                continue
            actual_rem.append(t)

    return cleaned, len(comments_to_remove), len(actual_rem)


def process_file(file_path: str, dry_run: bool = False) -> Dict[str, int]:
    with open(file_path, "rb") as f:
        encoding, _ = tokenize.detect_encoding(f.readline)

    with open(file_path, "r", encoding=encoding) as f:
        original_content = f.read()

    cleaned, removed_count, rem_count = purge_comments_from_source(original_content, file_path)

    orig_lines = len(original_content.splitlines())
    cleaned_lines = len(cleaned.splitlines())

    if not dry_run:
        with open(file_path, "w", encoding=encoding) as f:
            f.write(cleaned)

    return {
        "removed_comments": removed_count,
        "remaining_comments": rem_count,
        "original_lines": orig_lines,
        "cleaned_lines": cleaned_lines,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Purge comments from Python files safely.")
    parser.add_argument("--dry-run", action="store_true", help="Perform checks without writing files")
    parser.add_argument("paths", nargs="*", help="Files or directories to process")
    args = parser.parse_args()

    target_paths = args.paths or [
        "neuralops/attentions",
        "neuralops/compression_engine",
        "neuralops/compression",
    ]

    all_files: List[str] = []
    for p in target_paths:
        if os.path.isfile(p) and p.endswith(".py"):
            all_files.append(p)
        elif os.path.isdir(p):
            for root, _, filenames in os.walk(p):
                for f in filenames:
                    if f.endswith(".py"):
                        all_files.append(os.path.join(root, f))

    all_files.sort()
    print(f"Target files to process: {len(all_files)}")

    total_removed = 0
    total_remaining = 0
    total_orig_lines = 0
    total_cleaned_lines = 0
    errors = []

    for fpath in all_files:
        try:
            stats = process_file(fpath, dry_run=args.dry_run)
            total_removed += stats["removed_comments"]
            total_remaining += stats["remaining_comments"]
            total_orig_lines += stats["original_lines"]
            total_cleaned_lines += stats["cleaned_lines"]
            if stats["remaining_comments"] > 0:
                print(f"[WARN] {fpath}: {stats['remaining_comments']} comments remaining!")
        except Exception as e:
            errors.append((fpath, str(e)))
            print(f"[ERROR] {fpath}: {e}")

    mode_str = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n=== PURGE SUMMARY ({mode_str}) ===")
    print(f"Files processed:        {len(all_files)}")
    print(f"Errors:                  {len(errors)}")
    print(f"Comments removed:        {total_removed}")
    print(f"Remaining comments:      {total_remaining}")
    print(f"Original total lines:    {total_orig_lines}")
    print(f"Cleaned total lines:     {total_cleaned_lines}")
    print(f"Net line reduction:      {total_orig_lines - total_cleaned_lines}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
