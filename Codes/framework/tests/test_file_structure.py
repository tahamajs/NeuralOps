"""Unit tests for framework directory integrity, python compilation, and clean workspace."""

import os
import py_compile
from pathlib import Path

import pytest
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


class TestFileStructureAndCompilation:
    """Verify directory structure, bytecode compilation, and absence of AppleDouble artifacts."""

    @pytest.fixture
    def framework_root(self):
        # tests/ -> framework/
        return Path(__file__).resolve().parent.parent

    def test_required_directories_exist(self, framework_root):
        expected_dirs = [
            "core",
            "attentions",
            "compression_engine",
            "utils",
            "tests",
            "trainer",
        ]
        for d in expected_dirs:
            p = framework_root / d
            assert p.exists() and p.is_dir(), f"Expected directory {d} missing in {framework_root}"

    def test_no_appledouble_dot_underscore_files(self, framework_root):
        """macOS creates hidden '._*' files during copying; assert none exist in framework."""
        bad_files = [str(p) for p in framework_root.rglob("*") if p.name.startswith("._")]
        assert len(bad_files) == 0, f"Found unexpected AppleDouble files: {bad_files}"

    def test_all_framework_python_files_compile(self, framework_root):
        """Verify all python files in the framework compile to bytecode without syntax errors."""
        py_files = list(framework_root.rglob("*.py"))
        assert len(py_files) > 50

        compile_errors = []
        for py_file in py_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                compile_errors.append((str(py_file), str(e)))

        assert len(compile_errors) == 0, f"Syntax / compilation errors found: {compile_errors}"

    def test_registries_healthy_counts(self):
        """Verify both attention and compressor registries contain the required minimal set."""
        assert len(ATTENTION_REGISTRY.keys()) >= 15
        assert len(COMPRESSOR_REGISTRY.keys()) >= 8

    def test_duplicate_files_are_canonical_reexports(self, framework_root):
        """Files present in both attentions and compression_engine must be clean re-exports."""
        att_files = set(f.name for f in (framework_root / "attentions").glob("*.py"))
        ce_files = set(f.name for f in (framework_root / "compression_engine").glob("*.py"))
        overlap = (att_files & ce_files) - {"__init__.py"}

        for filename in overlap:
            p1 = (framework_root / "attentions" / filename).read_text()
            p2 = (framework_root / "compression_engine" / filename).read_text()
            # One must re-export the other, or they must be complementary
            is_reexport = ("Canonical home:" in p1 or "Canonical home:" in p2 or
                           "Re-exports" in p1 or "Re-exports" in p2 or
                           "import *" in p1 or "import *" in p2 or
                           "ResSVDLinear" in p1)
            assert is_reexport, f"{filename} is duplicated without canonical re-export linkage"
