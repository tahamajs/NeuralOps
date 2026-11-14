import pytest
from neuralops.benchmarks.cli_v2.main import main


def test_cli_v2_plan_command(capsys):
    ret = main(["plan", "--modules", "attention:flash_attention_2,attention:cosformer", "--device", "cpu"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Benchmark Plan Summary (v2):" in captured.out
    assert "Modules:  2" in captured.out


def test_cli_v2_plan_all_models_datasets(capsys):
    ret = main(["plan", "--all-models", "--all-datasets", "--device", "cpu"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Target Models:  10" in captured.out
    assert "Datasets:       22" in captured.out
