import torch
import torch.nn as nn
from neuralops.benchmarks.core.training import finetune_classification
from torch.utils.data import DataLoader, TensorDataset


def test_finetune_adamw_step():
    class DummySeq(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 2)
        def forward(self, input_ids, attention_mask=None, labels=None):
            logits = self.linear(input_ids.float())
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return type("Out", (), {"logits": logits, "loss": loss})()

    model = DummySeq()
    inputs = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, torch.ones_like(inputs), labels), batch_size=2)

    stats = finetune_classification(
        model=model,
        train_loader=loader,
        epochs=1,
        optimizer_name="adamw",
        device="cpu",
    )
    assert isinstance(stats, dict)
    assert stats["status"] == "success"
    assert stats["epochs_completed"] == 1
