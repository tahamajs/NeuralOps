"""Verification: Hardcoded String / Shape Scan."""
from pathlib import Path

import pytest


def test_no_hardcoded_roberta_in_core():
    root = Path(__file__).resolve().parent.parent / "core"
    for py_file in root.glob("*.py"):
        text = py_file.read_text()
        # Verify get_attention_modules does not use string matching on model family
        if py_file.name == "architecture.py":
            assert 'family == "roberta"' not in text
            assert 'family == "gpt2"' not in text
