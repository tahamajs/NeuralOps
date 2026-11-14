import torch
import torch.nn as nn
from neuralops.benchmarks.core.kd import KDLoss, kd_train_step


def test_kd_train_step_cpu():
    student = nn.Sequential(nn.Linear(8, 4), nn.Identity())
    # Mock logits output attribute
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)
        def forward(self, input_ids, **kw):
            out = self.linear(input_ids.float())
            return type("Out", (), {"logits": out})()

    student = DummyModel()
    teacher = DummyModel()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    kd_loss_fn = KDLoss()

    batch = (torch.randn(2, 8), torch.ones(2, 8), torch.tensor([0, 1]))
    loss_val = kd_train_step(
        student=student,
        teacher=teacher,
        batch=batch,
        optimizer=optimizer,
        kd_loss_fn=kd_loss_fn,
        device="cpu",
        mixed_precision="no",
    )
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
