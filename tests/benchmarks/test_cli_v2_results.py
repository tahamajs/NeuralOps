import pytest
from neuralops.benchmarks.cli_v2.main import main


def test_cli_v2_results_command(tmp_path, capsys):
    ret = main(["results", "--output-dir", str(tmp_path), "--device", "cpu"])
    assert ret == 0
    captured = capsys.readouterr()
    assert f"Results output directory: {tmp_path}" in captured.out


def test_cli_v2_results_format(tmp_path, capsys):
    ret = main(["results", "--output-dir", str(tmp_path), "--format", "json"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Results output directory:" in captured.out
