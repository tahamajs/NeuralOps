"""Dual-Conference Submission Packaging & Anonymization Engine.

Supports:
- NeurIPS 2026: Public metadata, open-source repository references, full LaTeX packages.
- ICLR 2026: Strict double-blind compliance, scrubbed URLs, anonymized repository links,
  and scrubbed compute/author metadata.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import sys

PROJECT_DIR = Path(__file__).resolve().parent


def scrub_anonymity(text: str) -> str:
    """Scrubs identifying metadata for double-blind peer review."""
    # Scrub GitHub repository URLs
    text = re.sub(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "[ANONYMIZED_REPOSITORY]", text)
    text = re.sub(r"git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "[ANONYMIZED_REPOSITORY]", text)
    
    # Scrub author and host names
    text = re.sub(r"Hadi Schumacher", "[ANONYMIZED_AUTHOR]", text, flags=re.IGNORECASE)
    text = re.sub(r"tahamajs", "[ANONYMIZED_USER]", text, flags=re.IGNORECASE)
    text = re.sub(r"192\.168\.1\.\d+", "[ANONYMIZED_IP]", text)
    text = re.sub(r"D:\\SVD_Project", "[WORKSPACE]", text)
    text = re.sub(r"/Users/[^/]+/", "/[ANONYMIZED_PATH]/", text)
    
    return text


def build_submission_package(target: str = "neurips", output_base: str = "submissions"):
    target = target.lower().strip()
    assert target in {"neurips", "iclr"}, "Target must be 'neurips' or 'iclr'"
    
    bundle_dir = PROJECT_DIR / output_base / f"{target}_submission"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Packaging submission artifact for target: {target.upper()}")
    print(f"[*] Target Directory: {bundle_dir}")

    # 1. Copy LaTeX tables & Markdown reports
    results_dir = PROJECT_DIR / "results"
    quality_runs = sorted(list(results_dir.glob("run_quality_*")))
    
    source_run = quality_runs[-1] if quality_runs else None
    if source_run and source_run.is_dir():
        print(f"[*] Sourcing results from: {source_run.name}")
        shutil.copytree(source_run, bundle_dir / "results", dirs_exist_ok=True)
    
    # 2. Copy Code & Documentation
    shutil.copytree(PROJECT_DIR / "Codes", bundle_dir / "Codes", dirs_exist_ok=True)
    shutil.copy(PROJECT_DIR / "README.md", bundle_dir / "README.md")
    shutil.copy(PROJECT_DIR / "Dockerfile", bundle_dir / "Dockerfile")
    shutil.copy(PROJECT_DIR / "requirements.txt", bundle_dir / "requirements.txt")
    shutil.copy(PROJECT_DIR / "run_full_benchmark.sh", bundle_dir / "run_full_benchmark.sh")
    shutil.copy(PROJECT_DIR / "run_full_benchmark.bat", bundle_dir / "run_full_benchmark.bat")

    # 3. Apply Double-Blind Anonymization if ICLR
    if target == "iclr":
        print("[*] Applying strict double-blind anonymization...")
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                if file.endswith((".py", ".md", ".tex", ".txt", ".json", ".csv", ".sh", ".bat")):
                    p = Path(root) / file
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        scrubbed = scrub_anonymity(content)
                        with open(p, "w", encoding="utf-8") as f:
                            f.write(scrubbed)
                    except Exception as e:
                        print(f"Warning: Could not scrub {p}: {e}")

    # 4. Create ZIP Archive
    archive_path = PROJECT_DIR / output_base / f"{target}_supplementary_material"
    shutil.make_archive(str(archive_path), "zip", bundle_dir)
    print(f"[✓] Successfully generated submission package: {archive_path}.zip")
    return str(bundle_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeurIPS / ICLR Submission Packaging Engine")
    parser.add_argument("--target", type=str, default="neurips", choices=["neurips", "iclr"], help="Target conference")
    args = parser.parse_args()
    build_submission_package(args.target)
