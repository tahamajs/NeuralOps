"""Concatenate all sections into paper_full.md."""

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


def build(md: Path, out: Path, pdf: bool = False):
    parts, missing = [], []
    for s in SECTIONS:
        p = md / s
        if not p.exists():
            missing.append(s)
            continue
        parts.append(p.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if missing:
        print(f"Missing: {missing}")

    full = "".join(parts)
    out.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {out}")

    if pdf:
        try:
            subprocess.run([
                "pandoc", str(out),
                "-o", str(out.with_suffix(".pdf")),
                "--pdf-engine=xelatex",
            ], check=True)
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--pdf", action="store_true")
    args = p.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return

    out = Path(args.output) if args.output else md / "paper_full.md"
    build(md, out, args.pdf)


if __name__ == "__main__":
    main()
```

---