"""Tests for complete mixed precision trainer with gradient accumulation."""
import torch
import torch.nn as nn
from neuralops.training.trainer import MixedPrecisionTrainer


def test_mixed_precision_train_epoch_grad_accum():
    model = nn.Linear(16, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    trainer = MixedPrecisionTrainer(model, optimizer, loss_fn, enabled=False)

    data = [(torch.randn(4, 16), torch.randn(4, 4)) for _ in range(6)]
    avg_loss = trainer.train_epoch(data, grad_accum=2, max_steps=4)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0.0
