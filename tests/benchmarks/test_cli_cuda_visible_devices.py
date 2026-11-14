import os
import pytest
from neuralops.benchmarks.cli_v2.main import main


def test_cli_cuda_visible_devices_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"

    ret = main(["benchmark", "--device", "cpu"])
    assert ret == 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_cli_cuda_visible_devices_plan_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    ret = main(["plan", "--device", "cpu"])
    assert ret == 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_cli_cuda_visible_devices_global_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    ret = main(["--device", "cpu", "results"])
    assert ret == 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
