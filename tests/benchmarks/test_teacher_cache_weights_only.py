import pytest
import torch
from pathlib import Path
from neuralops.benchmarks.kd.teacher_cache import _safe_torch_load


def test_safe_torch_load_weights_only(tmp_path: Path):
    model = torch.nn.Linear(4, 2)
    state = {"state_dict": model.state_dict(), "step": 10}
    ckpt_file = tmp_path / "model.pt"
    torch.save(state, ckpt_file)

    loaded = _safe_torch_load(ckpt_file)
    assert "state_dict" in loaded
    assert loaded["step"] == 10
