import pytest
from neuralops.benchmarks.cli_v2.main import main


def test_cli_v2_benchmark_command():
    ret = main(["benchmark", "--modules", "all", "--device", "cpu"])
    assert ret == 0
