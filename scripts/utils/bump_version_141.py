#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Bump version to 1.4.1 across the entire NeuralOps repository.

"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(".")


def bump_file(path: Path, old: str, new: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if old in text:
        new_text = text.replace(old, new)
        path.write_text(new_text, encoding="utf-8")
        print(f"Bumped {path}: {old} -> {new}")
        return True
    return False


def main() -> None:
    # 1. pyproject.toml
    bump_file(ROOT / "pyproject.toml", 'version = "1.4.0"', 'version = "1.4.1"')

    # 2. neuralops/__init__.py
    bump_file(ROOT / "neuralops" / "__init__.py", '__version__ = "1.4.0"', '__version__ = "1.4.1"')

    # 3. scripts/utils/verify_readme_counts.py
    bump_file(ROOT / "scripts" / "utils" / "verify_readme_counts.py", '"1.4.0"', '"1.4.1"')

    # 4. scripts/utils/publish_pypi_direct.sh
    bump_file(ROOT / "scripts" / "utils" / "publish_pypi_direct.sh", "1.4.0", "1.4.1")

    # 5. README.md
    bump_file(ROOT / "README.md", "v1.4.0", "v1.4.1")
    bump_file(ROOT / "README.md", "1.4.0", "1.4.1")

    # 6. Test files
    test_files = [
        ROOT / "tests" / "test_library_health.py",
        ROOT / "tests" / "test_installation_and_doctor.py",
        ROOT / "tests" / "test_integrations.py",
        ROOT / "tests" / "test_public_api.py",
        ROOT / "tests" / "test_master_end_to_end_framework_integration.py",
        ROOT / "tests" / "test_master_exhaustive_ecosystem_integration.py",
        ROOT / "tests" / "test_production_kernel_ecosystem_exhaustive.py",
        ROOT / "tests" / "test_triton_flagship_extended.py",
    ]
    for tf in test_files:
        bump_file(tf, "1.4.0", "1.4.1")

    # 7. Notebooks
    for nb in (ROOT / "notebooks").glob("*.ipynb"):
        bump_file(nb, "1.4.0", "1.4.1")

    print("Version bump to 1.4.1 complete!")


if __name__ == "__main__":
    main()
