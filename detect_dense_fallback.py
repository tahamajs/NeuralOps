#!/usr/bin/env python3
"""
Detect attention modules that fall back to DENSE softmax for decoder/causal mode.

This script scans all .py files in neuralops/attentions/ and identifies
modules where the causal/decoder path uses dense attention instead of the
intended sparse/linear/spherical mechanism.

Patterns to detect:
  1. if self.is_decoder or self.is_causal: ... F.softmax(scores) ...
  2. if is_decoder: ... dense attention ...
  3. use_dense = is_causal or ...
  4. Decoder branches that use torch.matmul(Q, K^T) with softmax

Output: 
  - Table of all modules with their causal-path status
  - Categorized as: SPARSE-AWARE / DENSE-FALLBACK / NO-CAUSAL-DETECTED / ERROR
"""

import ast
import re
import sys
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================
ATTENTIONS_DIR = Path(__file__).resolve().parent / "neuralops" / "attentions"
OUTPUT_CSV = "dense_fallback_audit.csv"

# Patterns that indicate dense fallback in decoder path
DENSE_FALLBACK_PATTERNS = [
    # Common forms of dense softmax in forward
    (r"if\s+self\.is_decoder\s+or\s+self\.is_causal\s*:", "is_decoder or is_causal check"),
    (r"if\s+self\.is_decoder\s*:", "is_decoder check"),
    (r"if\s+self\.is_causal\s*:", "is_causal check"),
    (r"use_dense\s*=\s*is_causal", "use_dense = is_causal"),
    (r"use_dense\s*=\s*.*is_decoder", "use_dense with is_decoder"),
    (r"if\s+is_causal\s+or\s+is_cross", "is_causal or is_cross check"),
    (r"if\s+is_decoder\s+or\s+is_cross", "is_decoder or is_cross check"),
]

# Patterns that indicate TRUE sparse/linear causal handling
SPARSE_CAUSAL_PATTERNS = [
    (r"_chunked_causal", "chunked causal method"),
    (r"_causal_linear_attention", "causal linear attention"),
    (r"chunked_delta_rule", "chunked delta rule"),
    (r"triton.*causal", "triton causal kernel"),
    (r"def\s+_causal", "custom causal method"),
    (r"causal_scan", "causal scan"),
]

# Patterns that indicate dense softmax usage
DENSE_SOFTMAX_PATTERNS = [
    (r"F\.softmax\(\s*scores", "F.softmax(scores)"),
    (r"torch\.softmax\(\s*scores", "torch.softmax(scores)"),
    (r"softmax\(\s*dense_scores", "softmax(dense_scores)"),
]


# ============================================================================
# Scanner
# ============================================================================
def read_file_safe(path: Path) -> Optional[str]:
    """Read file with UTF-8 encoding, fallback to latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def find_forward_method(content: str) -> Optional[str]:
    """Extract the `def forward(...)` body from the file."""
    # Find "def forward" and grab until next top-level def or class
    match = re.search(
        r"(\n    def forward\s*\(.*?)(?=\n    def |\nclass |\n@|\Z)",
        content,
        re.DOTALL,
    )
    if match:
        return match.group(1)
    # Fallback: find first forward anywhere
    match = re.search(r"(def forward\s*\(.*?)(?=\ndef |\nclass |\Z)", content, re.DOTALL)
    return match.group(1) if match else None


def detect_patterns(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    """Return list of pattern descriptions that match in text."""
    found = []
    for regex, desc in patterns:
        if re.search(regex, text, re.IGNORECASE):
            found.append(desc)
    return found


def classify_module(filepath: Path, content: str) -> dict:
    """Classify a module's causal path."""
    forward_body = find_forward_method(content)
    if forward_body is None:
        return {
            "status": "NO_FORWARD",
            "reason": "no forward method found",
            "dense_patterns": [],
            "sparse_patterns": [],
            "has_softmax": False,
        }

    # Check for dense fallback patterns
    dense_matches = detect_patterns(forward_body, DENSE_FALLBACK_PATTERNS)

    # Check for TRUE sparse causal patterns
    sparse_matches = detect_patterns(content, SPARSE_CAUSAL_PATTERNS)

    # Check for softmax usage
    softmax_matches = detect_patterns(forward_body, DENSE_SOFTMAX_PATTERNS)

    # Classify
    if sparse_matches:
        status = "SPARSE_AWARE"
        reason = f"has sparse causal: {', '.join(sparse_matches)}"
    elif dense_matches:
        status = "DENSE_FALLBACK"
        reason = f"dense fallback in forward: {', '.join(dense_matches)}"
    elif softmax_matches:
        status = "LIKELY_DENSE"
        reason = f"softmax in forward but no explicit causal check: {', '.join(softmax_matches)}"
    else:
        status = "UNKNOWN"
        reason = "no clear causal handling pattern"

    return {
        "status": status,
        "reason": reason,
        "dense_patterns": dense_matches,
        "sparse_patterns": sparse_matches,
        "has_softmax": len(softmax_matches) > 0,
    }


def main():
    if not ATTENTIONS_DIR.exists():
        print(f"ERROR: {ATTENTIONS_DIR} does not exist")
        sys.exit(1)

    py_files = sorted(ATTENTIONS_DIR.glob("*.py"))
    py_files = [p for p in py_files if not p.stem.startswith("_")]

    print("=" * 100)
    print(f"SCANNING {len(py_files)} ATTENTION MODULES")
    print("=" * 100)

    results = []
    for path in py_files:
        content = read_file_safe(path)
        if content is None:
            results.append({
                "module": path.stem,
                "status": "READ_ERROR",
                "reason": "could not read file",
                "dense_patterns": "",
                "sparse_patterns": "",
                "has_softmax": False,
            })
            continue

        try:
            info = classify_module(path, content)
            results.append({
                "module": path.stem,
                "status": info["status"],
                "reason": info["reason"],
                "dense_patterns": "; ".join(info["dense_patterns"]),
                "sparse_patterns": "; ".join(info["sparse_patterns"]),
                "has_softmax": info["has_softmax"],
            })
        except Exception as e:
            results.append({
                "module": path.stem,
                "status": "SCAN_ERROR",
                "reason": str(e),
                "dense_patterns": "",
                "sparse_patterns": "",
                "has_softmax": False,
            })

    # Categorize
    categories = {
        "SPARSE_AWARE": [],
        "DENSE_FALLBACK": [],
        "LIKELY_DENSE": [],
        "UNKNOWN": [],
        "NO_FORWARD": [],
        "READ_ERROR": [],
        "SCAN_ERROR": [],
    }
    for r in results:
        categories.setdefault(r["status"], []).append(r)

    # Print summary
    print()
    print("-" * 100)
    print("SUMMARY")
    print("-" * 100)
    for cat in ["SPARSE_AWARE", "DENSE_FALLBACK", "LIKELY_DENSE", "UNKNOWN", "NO_FORWARD", "READ_ERROR", "SCAN_ERROR"]:
        n = len(categories.get(cat, []))
        if n > 0:
            print(f"  {cat:20s}: {n:3d} modules")

    # Print DENSE_FALLBACK modules (most important)
    print()
    print("=" * 100)
    print("MODULES WITH DENSE FALLBACK (need chunked causal fix):")
    print("=" * 100)
    for r in categories.get("DENSE_FALLBACK", []):
        print(f"  {r['module']:40s} | {r['reason']}")

    # Print LIKELY_DENSE
    print()
    print("=" * 100)
    print("MODULES LIKELY DENSE (no explicit causal check, has softmax):")
    print("=" * 100)
    for r in categories.get("LIKELY_DENSE", []):
        print(f"  {r['module']:40s} | {r['reason']}")

    # Print SPARSE_AWARE
    print()
    print("=" * 100)
    print("MODULES WITH TRUE SPARSE CAUSAL:")
    print("=" * 100)
    for r in categories.get("SPARSE_AWARE", []):
        print(f"  {r['module']:40s} | {r['reason']}")

    # Print UNKNOWN
    print()
    print("=" * 100)
    print("MODULES WITH UNKNOWN CAUSAL HANDLING:")
    print("=" * 100)
    for r in categories.get("UNKNOWN", []):
        print(f"  {r['module']:40s}")

    # Save CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["module", "status", "reason", "dense_patterns", "sparse_patterns", "has_softmax"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print()
    print(f"Full results saved to: {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()