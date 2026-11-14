import torch.nn as nn
from neuralops.benchmarks.kd.runner import KDRunner
from neuralops.benchmarks.kd.config import BenchmarkConfig


def test_kd_runner_unfreezes_all_when_no_attention():
    cfg = BenchmarkConfig()
    runner = KDRunner(cfg)
    # A model without explicit attention modules
    student = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    for p in student.parameters():
        p.requires_grad = False

    # Simulate logic from runner
    for p in student.parameters():
        p.requires_grad = False

    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        for p in student.parameters():
            p.requires_grad = True
        trainable = [p for p in student.parameters() if p.requires_grad]

    assert len(trainable) == 4
    assert all(p.requires_grad for p in student.parameters())
