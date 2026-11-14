from pathlib import Path
import pytest
from neuralops.benchmarks.core.data_loader import _load_local_dataset


def test_load_local_unsupported_suffix(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported local dataset format"):
        _load_local_dataset(bad_file, streaming=False)
