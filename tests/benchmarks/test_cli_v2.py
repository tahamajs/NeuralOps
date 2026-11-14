"""Comprehensive tests for NeuralOps CLI v2 commands and entrypoints.

Tests benchmark, run, plan, results, doctor, ui, and completion commands,
verifying parameter handling, output formats, and error conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from neuralops._version import __version__
from neuralops.benchmarks.cli_v2.main import create_parser, main
from neuralops.benchmarks.cli_v2.utils.completion import (
    CANONICAL_COMMANDS,
    SUPPORTED_SHELLS,
    generate_completion_script,
)
from neuralops.benchmarks.cli_v2.utils.rendering import (
    get_console,
    get_err_console,
    print_cli_error,
    render_badge,
    render_summary_card,
    render_table,
)


class TestCLIV2Commands:
    """Test suite for CLI v2 command handlers and routing."""

    def test_benchmark_basic_invocation(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify neuralops benchmark execution with basic parameters."""
        ret = main(["benchmark", "--modules", "all", "--device", "cpu", "--dry-run"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Benchmark dry-run" in captured.out
        assert "modules=all" in captured.out

    def test_benchmark_run_alias(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify neuralops run alias dispatches identically to benchmark."""
        ret = main(["run", "--modules", "attention:flash_attention_2", "--device", "cpu", "--dry-run"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Benchmark dry-run" in captured.out
        assert "attention:flash_attention_2" in captured.out

    def test_benchmark_missing_config_returns_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify providing a non-existent config file returns non-zero error code."""
        ret = main(["benchmark", "--config", "/nonexistent/config/path.yaml"])
        assert ret == 1
        captured = capsys.readouterr()
        err_out = captured.err if captured.err else captured.out
        assert "not found" in err_out.lower()

    def test_plan_command_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify plan command calculates and outputs human-readable summary."""
        ret = main(["plan", "--modules", "attention:full_attention", "--device", "cpu"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Benchmark Plan Summary (v2):" in captured.out
        assert "Modules:  1" in captured.out
        assert "Total planned runs:" in captured.out

    def test_plan_command_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify plan command with --json outputs valid parseable JSON."""
        ret = main(["plan", "--modules", "attention:full_attention,attention:cosformer", "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["modules"] == 2
        assert "total_runs" in data
        assert "estimated_hours" in data

    def test_results_command_inspection(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify results subcommand prints output directory and scans directory."""
        sample_json = tmp_path / "run_01.json"
        sample_json.write_text(json.dumps({"name": "run_01", "status": "PASS"}), encoding="utf-8")

        ret = main(["results", "--output-dir", str(tmp_path), "--list"])
        assert ret == 0
        captured = capsys.readouterr()
        assert f"Results output directory: {tmp_path}" in captured.out
        assert "Found 1 benchmark records" in captured.out

    def test_results_command_json_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify results subcommand outputs structured JSON when requested."""
        sample_json = tmp_path / "run_02.json"
        sample_json.write_text(json.dumps({"name": "test_kernel", "status": "SUCCESS"}), encoding="utf-8")

        ret = main(["results", "--output-dir", str(tmp_path), "--format", "json"])
        assert ret == 0
        captured = capsys.readouterr()
        # Find json block in output
        lines = captured.out.strip().split("\n")
        json_str = "\n".join(l for l in lines if not l.startswith("Results output directory:"))
        data = json.loads(json_str)
        assert data["record_count"] == 1
        assert data["records"][0]["name"] == "test_kernel"

    def test_doctor_command_delegation(self) -> None:
        """Verify doctor subcommand delegates to canonical doctor runner."""
        with patch("neuralops.benchmarks.doctor.cmd_doctor", return_value=0) as mock_doc:
            ret = main(["doctor", "--device", "cpu"])
            assert ret == 0
            mock_doc.assert_called_once()

    def test_ui_command_delegation(self) -> None:
        """Verify ui subcommand delegates to web UI runner."""
        with patch("neuralops.ui.cli.cmd_ui", return_value=0) as mock_ui:
            ret = main(["ui", "--port", "8888", "--device", "cpu"])
            assert ret == 0
            mock_ui.assert_called_once()

    def test_completion_command_shells(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify completion subcommand generates valid scripts for bash, zsh, fish."""
        for shell in ("bash", "zsh", "fish"):
            ret = main(["completion", shell])
            assert ret == 0
            captured = capsys.readouterr()
            assert "neuralops" in captured.out

    def test_install_completion_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --install-completion flag outputs completion script."""
        ret = main(["--install-completion", "bash"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "_neuralops_complete" in captured.out

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --version flag prints NeuralOps version and exits cleanly."""
        ret = main(["--version"])
        assert ret == 0
        captured = capsys.readouterr()
        out = captured.out if captured.out else captured.err
        assert __version__ in out


class TestCLIV2Utils:
    """Test suite for CLI v2 utilities (rendering and completion)."""

    def test_generate_completion_script_valid_shells(self) -> None:
        """Verify completion generation contains proper command lists."""
        for shell in SUPPORTED_SHELLS:
            script = generate_completion_script(shell)
            for cmd in CANONICAL_COMMANDS:
                assert cmd in script

    def test_generate_completion_script_unsupported_shell(self) -> None:
        """Verify ValueError is raised for unsupported shell names."""
        with pytest.raises(ValueError, match="Unsupported shell"):
            generate_completion_script("unknown_shell")

    def test_render_badge(self) -> None:
        """Verify status badge rendering produces non-empty representations."""
        for st in ("PASS", "FAIL", "SKIP", "WARN", "RUNNING"):
            badge = render_badge(st)
            assert st in badge

    def test_render_table_and_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify render_table and render_summary_card execute cleanly without errors."""
        render_table("Test Table", ["Col A", "Col B"], [["val1", "val2"]])
        render_summary_card("Test Summary", [("Field 1", "Value 1"), ("Field 2", "Value 2")])
        captured = capsys.readouterr()
        assert "Test Table" in captured.out or "Col A" in captured.out
        assert "Test Summary" in captured.out or "Field 1" in captured.out

    def test_print_cli_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify print_cli_error prints message and suggestion."""
        print_cli_error("Simulated failure", hint="Check settings", title="Error Card")
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Simulated failure" in combined
        assert "Check settings" in combined
