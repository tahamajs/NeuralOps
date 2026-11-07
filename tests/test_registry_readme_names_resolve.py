#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying all attention mechanisms advertised in README resolve via registry.

Ensures no dead-end aliases, phantom entries, or unresolvable names in README tables
or deprecated alias definitions (AGT-11 / Finding F-19).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
DEPRECATED_ALIASES_PATH = REPO_ROOT / "neuralops" / "core" / "deprecated_aliases.json"


def test_every_readme_attention_name_resolves():
    """Verify that every attention mechanism name listed in README Section 13 resolves."""
    assert README_PATH.exists(), f"README.md not found at {README_PATH}"
    content = README_PATH.read_text(encoding="utf-8")

    m = re.search(r"## 13\..*?(?=## 14\.)", content, re.DOTALL)
    assert m, "Section 13 (Attention Zoo) not found in README.md"
    sec13 = m.group(0)

    # 1. Names listed in the 13.3 Hardware Audit Table (| # | `name` | ...)
    table_keys = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", sec13, re.MULTILINE)
    assert len(table_keys) > 0, "No attention module entries found in README §13 table"

    unresolved = []
    for name in table_keys:
        obj = ATTENTION_REGISTRY.get(name)
        if obj is None:
            unresolved.append(name)

    assert not unresolved, (
        f"The following {len(unresolved)} attention mechanism names from README §13 "
        f"failed to resolve via ATTENTION_REGISTRY.get(): {unresolved}"
    )


def test_no_unresolvable_alias_chain():
    """Verify that no alias in deprecated_aliases.json dead-ends."""
    assert DEPRECATED_ALIASES_PATH.exists(), f"Alias file not found: {DEPRECATED_ALIASES_PATH}"
    with open(DEPRECATED_ALIASES_PATH, encoding="utf-8") as f:
        aliases: dict[str, str] = json.load(f)

    assert len(aliases) > 0, "No deprecated aliases found in deprecated_aliases.json"

    dead_ends = []
    for alias, target in aliases.items():
        # Both the target and the alias itself must resolve to a valid registered entity
        res_target = ATTENTION_REGISTRY.get(target) or COMPRESSOR_REGISTRY.get(target)
        if res_target is None:
            dead_ends.append((alias, target, "target_unresolved"))
            continue

        res_alias = ATTENTION_REGISTRY.get(alias) or COMPRESSOR_REGISTRY.get(alias)
        if res_alias is None:
            dead_ends.append((alias, target, "alias_unresolved"))

    assert not dead_ends, (
        f"Found {len(dead_ends)} dead-end alias chain(s) in deprecated_aliases.json:\n"
        + "\n".join(f"  {a} -> {t} ({reason})" for a, t, reason in dead_ends[:20])
    )


def test_every_readme_compressor_name_resolves():
    """Verify that every compression engine name listed in README Section 14 resolves."""
    assert README_PATH.exists(), f"README.md not found at {README_PATH}"
    content = README_PATH.read_text(encoding="utf-8")

    m = re.search(r"## 14\..*?(?=## 15\.)", content, re.DOTALL)
    assert m, "Section 14 (Compression Engines) not found in README.md"
    sec14 = m.group(0)

    # Names listed in the 14.3 Hardware Audit Table (| # | `name` | ...)
    table_keys = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", sec14, re.MULTILINE)
    assert len(table_keys) > 0, "No compressor entries found in README §14 table"

    unresolved = []
    for name in table_keys:
        obj = COMPRESSOR_REGISTRY.get(name)
        if obj is None:
            unresolved.append(name)

    assert not unresolved, (
        f"The following {len(unresolved)} compression engine names from README §14 "
        f"failed to resolve via COMPRESSOR_REGISTRY.get(): {unresolved}"
    )
