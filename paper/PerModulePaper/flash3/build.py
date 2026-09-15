"""Build the A3 module paper from its section files."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SECTIONS = [
    "00_meta.md",
    "01_intro.md",
    "02_related.md",
    "03_background.md",
    "04_method.md",
    "05_architecture.md",
    "06_setup.md",
    "07_results.md",
    "08_analysis.md",
    "09_discussion.md",
    "10_limitations.md",
    "11_conclusion.md",
    "12_reproducibility.md",
    "13_references.md",
    "A_math.md",
    "B_implementation.md",
    "C_data.md",
    "D_negative_results.md",
    "E_ethics.md",
]


def build(output: Path, to_pdf: bool = False) -> None:
    missing = [name for name in SECTIONS if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing paper sections: " + ", ".join(missing)
        )

    parts = []
    for name in SECTIONS:
        parts.append((ROOT / name).read_text(encoding="utf-8").rstrip())
    output.write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {len(output.read_text(encoding='utf-8')):,} chars to {output}")

    if to_pdf:
        pdf = output.with_suffix(".pdf")
        try:
            subprocess.run(
                [
                    "pandoc",
                    str(output),
                    "-o",
                    str(pdf),
                    "--pdf-engine=xelatex",
                    "-V",
                    "geometry:margin=1in",
                ],
                check=True,
            )
        except FileNotFoundError:
            print("pandoc is not installed; Markdown build completed without PDF.")
        except subprocess.CalledProcessError as exc:
            print(f"PDF generation failed with exit code {exc.returncode}.")
        else:
            print(f"Wrote PDF to {pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_full.md",
        help="output Markdown path (default: paper_full.md)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="also generate a PDF when pandoc and XeLaTeX are available",
    )
    args = parser.parse_args()
    build(args.output, to_pdf=args.pdf)


if __name__ == "__main__":
    main()
