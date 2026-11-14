import torch
from neuralops.benchmarks.core.kd import KDLoss


def test_kd_loss_forward():
    kd = KDLoss(temperature=2.0, alpha=0.5)
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 2, 3])

    loss = kd(student_logits=student_logits, teacher_logits=teacher_logits, labels=labels)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert float(loss) > 0.0
