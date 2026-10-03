#!/usr/bin/env python3
"""Canonical paper builder for PerModulePaper.

Concatenates modular markdown sections into unified paper drafts or PDFs.
Usage:
    python paper/build_paper.py --module vspec
    python paper/build_paper.py --all
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md",
    "11_conclusion.md", "12_reproducibility.md", "13_references.md",
    "A_math.md", "B_implementation.md", "C_data.md",
    "D_negative_results.md", "E_ethics.md",
]


def build_module(module_dir: Path, output_file: Path | None = None, to_pdf: bool = False) -> bool:
    if not module_dir.is_dir():
        return False
    parts = []
    missing = []
    for section in SECTIONS:
        path = module_dir / section
        if not path.exists():
            missing.append(section)
            continue
        parts.append(path.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if not parts:
        return False

    output = output_file or (module_dir / "paper_full.md")
    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"[{module_dir.name}] Wrote {len(full):,} chars to {output}")

    if to_pdf:
        try:
            pdf_path = output.with_suffix(".pdf")
            subprocess.run([
                "pandoc", str(output),
                "-o", str(pdf_path),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
            ], check=True)
            print(f"[{module_dir.name}] Generated PDF: {pdf_path}")
        except Exception as e:
            print(f"[{module_dir.name}] PDF build skipped: {e}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified papers from modular sections.")
    parser.add_argument("--module", type=str, default=None, help="Module folder name inside PerModulePaper")
    parser.add_argument("--all", action="store_true", help="Build all modules inside PerModulePaper")
    parser.add_argument("--output", type=str, default=None, help="Explicit output path")
    parser.add_argument("--pdf", action="store_true", help="Compile to PDF via pandoc")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent / "PerModulePaper"
    if args.all:
        built = 0
        for mod_dir in sorted(base_dir.iterdir()):
            if mod_dir.is_dir() and not mod_dir.name.startswith(("_", ".")):
                if build_module(mod_dir, to_pdf=args.pdf):
                    built += 1
        print(f"Successfully built {built} module papers.")
    elif args.module:
        mod_dir = base_dir / args.module
        out = Path(args.output) if args.output else None
        if not build_module(mod_dir, output_file=out, to_pdf=args.pdf):
            print(f"Module paper directory not found or empty: {mod_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
