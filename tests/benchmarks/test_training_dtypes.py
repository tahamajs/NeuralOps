import torch
from neuralops.benchmarks.config import BenchmarkConfig


def test_training_dtypes_supported():
    for prec, expected in [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        cfg = BenchmarkConfig(device="cpu", precision=prec)
        assert cfg.resolve_torch_dtype() == expected
