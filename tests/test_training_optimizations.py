"""Tests for Phase 7 training optimizations."""
import torch
import torch.nn as nn
from neuralops.training.trainer import (
    setup_ddp_training,
    MixedPrecisionTrainer,
    LoRATrainer,
    CurriculumScheduler,
)


def test_setup_ddp_training_uninit():
    model = nn.Linear(8, 8)
    wrapped = setup_ddp_training(model)
    assert wrapped is model


def test_mixed_precision_trainer():
    model = nn.Linear(8, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    trainer = MixedPrecisionTrainer(model, opt, loss_fn, enabled=False)

    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    loss = trainer.train_step(x, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_lora_trainer():
    class DummyWithLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(8, 8)
            self.lora_a = nn.Linear(8, 2, bias=False)

        def forward(self, x):
            return self.base(x) + self.lora_a(x).sum(-1, keepdim=True)

    model = DummyWithLoRA()
    trainer = LoRATrainer(model)
    assert not model.base.weight.requires_grad
    assert model.lora_a.weight.requires_grad

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    loss = trainer.train_step(x, y, nn.MSELoss())
    assert isinstance(loss, float)


def test_curriculum_scheduler():
    scheduler = CurriculumScheduler(min_seq_len=128, max_seq_len=1024, total_steps=100)
    assert scheduler.get_seq_len(0) == 128
    assert scheduler.get_seq_len(50) == 576
    assert scheduler.get_seq_len(100) == 1024
    assert scheduler.get_seq_len(200) == 1024
