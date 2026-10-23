#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated code quality, static hygiene, and security regression test suite.

Covers:
- test_no_hardcoded_pass_in_except: Verifies zero 'except Exception: pass' occurrences in neuralops/.
- test_no_unwanted_files_tracked: Verifies no heavy binary files (.onnx, .gguf, .trt, .pt, .bin) > 500KB tracked in git.
- test_shebang_consistency: Verifies CLI scripts in scripts/ start with '#!/usr/bin/env python3'.
- test_no_secret_leaks: Verifies no secret keys / tokens (sk-, ghp_, AKIA, BEGIN PRIVATE KEY) exist in git-tracked files.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


def test_no_hardcoded_pass_in_except():
    """Verify grep -rn 'except Exception:\\s*pass' neuralops/ returns zero results."""
    repo_root = Path(__file__).resolve().parent.parent
    neuralops_dir = repo_root / "neuralops"
    assert neuralops_dir.is_dir(), f"neuralops directory not found at {neuralops_dir}"

    single_line_pass = re.compile(r"except\s+Exception\s*:\s*pass\b")
    violations: list[str] = []

    for py_file in sorted(neuralops_dir.rglob("*.py")):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            if single_line_pass.search(line):
                rel_path = py_file.relative_to(repo_root)
                violations.append(f"{rel_path}:{line_no}: {line.strip()}")

    # Also verify via subprocess grep matching the exact gate command
    try:
        proc = subprocess.run(
            ["grep", "-rnE", r"except Exception:\s*pass", str(neuralops_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for match_line in proc.stdout.splitlines():
                if match_line.strip() and match_line not in violations:
                    violations.append(match_line.strip())
    except FileNotFoundError:
        pass

    assert not violations, (
        f"Found {len(violations)} occurrences of 'except Exception: pass' in neuralops/:\n"
        + "\n".join(violations)
    )


def test_no_unwanted_files_tracked():
    """Verify no .onnx, .gguf, .trt, .pt, .bin files over 500KB are tracked in git."""
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    banned_extensions = (".onnx", ".gguf", ".trt", ".pt", ".bin")
    max_size_bytes = 500 * 1024  # 500 KB

    violating_files: list[str] = []
    for rel_path in tracked_files:
        if rel_path.lower().endswith(banned_extensions):
            file_path = repo_root / rel_path
            if file_path.is_file():
                size = file_path.stat().st_size
                if size > max_size_bytes:
                    violating_files.append(f"{rel_path} ({size / 1024:.1f} KB > 500 KB)")
            else:
                cat_proc = subprocess.run(
                    ["git", "cat-file", "-s", f":{rel_path}"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                )
                if cat_proc.returncode == 0:
                    size = int(cat_proc.stdout.strip())
                    if size > max_size_bytes:
                        violating_files.append(
                            f"{rel_path} (git object: {size / 1024:.1f} KB > 500 KB)"
                        )

    assert not violating_files, (
        f"Found {len(violating_files)} tracked binary files exceeding 500KB limit:\n"
        + "\n".join(violating_files)
    )


def test_shebang_consistency():
    """Verify executable scripts in scripts/ start with #!/usr/bin/env python3."""
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    assert scripts_dir.is_dir(), f"scripts directory not found at {scripts_dir}"

    violations: list[str] = []
    found_any = False

    for script_path in sorted(scripts_dir.rglob("*")):
        if not script_path.is_file():
            continue

        rel_path = script_path.relative_to(repo_root)

        # Check Python scripts in scripts/
        if script_path.suffix == ".py":
            found_any = True
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
            except Exception as exc:
                violations.append(f"{rel_path}: failed to read ({exc})")
                continue

            if first_line != "#!/usr/bin/env python3":
                violations.append(
                    f"{rel_path}: expected '#!/usr/bin/env python3', got {first_line!r}"
                )
        elif script_path.suffix == ".sh":
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
            except Exception as exc:
                violations.append(f"{rel_path}: failed to read ({exc})")
                continue

            if not first_line.startswith("#!"):
                violations.append(f"{rel_path}: shell script missing shebang, got {first_line!r}")

    assert found_any, "No Python scripts found in scripts/"
    assert not violations, (
        f"Found {len(violations)} shebang inconsistencies in scripts/:\n"
        + "\n".join(violations)
    )


def test_no_secret_leaks():
    """Check for API key patterns (e.g. sk-, ghp_, AKIA, BEGIN PRIVATE KEY) across git tracked files."""
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    # Patterns for actual secrets/tokens (not regex pattern definitions)
    secret_patterns: dict[str, re.Pattern[str]] = {
        "openai_api_key": re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"),
        "github_pat": re.compile(r"\bghp_[a-zA-Z0-9]{30,}\b"),
        "github_fine_grained_pat": re.compile(r"\bgithub_pat_[a-zA-Z0-9_]{30,}\b"),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "private_key": re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"),
    }

    # Files that define security scanners or this test suite itself
    exempt_files = {
        "tests/test_code_quality.py",
        "neuralops/models/hub.py",
        "neuralops/benchmarks/results/hf_sync.py",
    }

    leaks: list[str] = []
    for rel_path in tracked_files:
        if rel_path in exempt_files:
            continue
        file_path = repo_root / rel_path
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for secret_name, pattern in secret_patterns.items():
            matches = pattern.findall(content)
            if matches:
                masked = [m[:8] + "..." for m in matches]
                leaks.append(f"{rel_path}: {secret_name} detected ({', '.join(masked)})")

    assert not leaks, (
        f"Found {len(leaks)} potential secret leak(s) in tracked repository files:\n"
        + "\n".join(leaks)
    )


def test_secret_patterns_synthetic_detection():
    """Verify the secret detection patterns catch synthetic tokens."""
    patterns = {
        "openai_api_key": re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"),
        "github_pat": re.compile(r"\bghp_[a-zA-Z0-9]{30,}\b"),
        "github_fine_grained_pat": re.compile(r"\bgithub_pat_[a-zA-Z0-9_]{30,}\b"),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "private_key": re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"),
    }

    # Dynamically constructed to prevent static scanners from flagging this test file
    sample_openai = "".join(["s", "k", "-", "a" * 25])
    sample_ghp = "".join(["g", "h", "p", "_", "b" * 32])
    sample_gh_fine = "".join(["g", "i", "t", "h", "u", "b", "_", "p", "a", "t", "_", "c" * 32])
    sample_aws = "".join(["A", "K", "I", "A", "1" * 16])
    sample_key = "".join(["-", "-", "-", "-", "-", "BEGIN RSA PRIVATE KEY-----"])

    assert patterns["openai_api_key"].search(sample_openai) is not None
    assert patterns["github_pat"].search(sample_ghp) is not None
    assert patterns["github_fine_grained_pat"].search(sample_gh_fine) is not None
    assert patterns["aws_access_key"].search(sample_aws) is not None
    assert patterns["private_key"].search(sample_key) is not None


def test_no_hardcoded_pass_in_except_synthetic():
    """Verify regex correctly catches single-line except Exception: pass."""
    pattern = re.compile(r"except\s+Exception\s*:\s*pass\b")
    assert pattern.search("except Exception: pass") is not None
    assert pattern.search("except Exception:   pass") is not None
    assert pattern.search("except Exception as exc: pass") is None
    assert pattern.search("except KeyError: pass") is None
