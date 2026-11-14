import pytest
from neuralops.benchmarks.results.safetensors_io import load_surgerized_safetensors


def test_safetensors_strict_non_existent():
    with pytest.raises(FileNotFoundError):
        load_surgerized_safetensors("/non/existent/path.safetensors", strict=True)
