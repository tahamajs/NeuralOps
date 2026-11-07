# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for component family taxonomy and README consistency (F-21)."""

from collections import defaultdict
from pathlib import Path
import re

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def test_every_registered_class_has_family():
    """Verify no registered class has FAMILY == 'uncategorized' (except experimental/)."""
    ATTENTION_REGISTRY._load_all()
    COMPRESSOR_REGISTRY._load_all()

    for name, reg in [("ATTENTION", ATTENTION_REGISTRY), ("COMPRESSOR", COMPRESSOR_REGISTRY)]:
        for k in reg.canonical_keys():
            obj = reg._dict[k]
            fam = getattr(obj, "FAMILY", None)
            assert fam is not None, f"{name} entry '{k}' has no FAMILY attribute"
            if "experimental" not in getattr(obj, "__module__", ""):
                assert fam != "uncategorized", (
                    f"{name} entry '{k}' in module '{getattr(obj, '__module__', '')}' "
                    f"has FAMILY == 'uncategorized'"
                )


def test_readme_family_counts_match_registry():
    """Verify each family's count in README.md matches the registry canonical count."""
    ATTENTION_REGISTRY._load_all()
    COMPRESSOR_REGISTRY._load_all()

    # Calculate registry counts per family
    expected_attn_counts: dict[str, int] = defaultdict(int)
    for k in ATTENTION_REGISTRY.canonical_keys():
        fam = getattr(ATTENTION_REGISTRY._dict[k], "FAMILY", "uncategorized")
        expected_attn_counts[fam] += 1

    expected_comp_counts: dict[str, int] = defaultdict(int)
    for k in COMPRESSOR_REGISTRY.canonical_keys():
        fam = getattr(COMPRESSOR_REGISTRY._dict[k], "FAMILY", "uncategorized")
        expected_comp_counts[fam] += 1

    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md not found"
    readme_text = readme_path.read_text(encoding="utf-8")

    # Split README by section
    sec13_text = readme_text.split("## 13. Attention Zoo")[1].split("## 14. Compression Engines")[0]
    sec14_text = readme_text.split("## 14. Compression Engines")[1]

    # Parse markdown table rows: | family | count | members |
    table_row_regex = re.compile(r"^\|\s*([a-zA-Z0-9_]+)\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|$", re.MULTILINE)
    readme_attn_families = {m[0]: int(m[1]) for m in table_row_regex.findall(sec13_text)}
    readme_comp_families = {m[0]: int(m[1]) for m in table_row_regex.findall(sec14_text)}

    # Verify attention families
    for fam, expected_count in expected_attn_counts.items():
        assert fam in readme_attn_families, f"Attention family '{fam}' not found in README §13 table"
        assert readme_attn_families[fam] == expected_count, (
            f"Attention family '{fam}' count mismatch: README has {readme_attn_families[fam]}, "
            f"registry has {expected_count}"
        )

    # Verify compressor families
    for fam, expected_count in expected_comp_counts.items():
        assert fam in readme_comp_families, f"Compressor family '{fam}' not found in README §14 table"
        assert readme_comp_families[fam] == expected_count, (
            f"Compressor family '{fam}' count mismatch: README has {readme_comp_families[fam]}, "
            f"registry has {expected_count}"
        )
