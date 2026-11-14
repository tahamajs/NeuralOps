import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neuralops.benchmarks.kd.distill_engine import TwoStageDistillationEngine


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 10)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        class Output:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states
        h = self.linear(input_ids.float().unsqueeze(-1).expand(-1, -1, 8))
        logits = self.lm_head(h)
        return Output(logits, [h, h])


def test_distill_engine_budget_guard():
    engine = TwoStageDistillationEngine(
        train_secs=0.0001,
        min_steps_per_stage=2,
        device=torch.device("cpu"),
    )

    dataset = TensorDataset(
        torch.zeros((4, 8), dtype=torch.long),
        torch.ones((4, 8), dtype=torch.long),
    )
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
        }

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    student = DummyModel()
    teacher = DummyModel()

    def eval_fn(mod, dl, dev):
        return 1.0

    res = engine.run(
        student=student,
        train_loader=loader,
        val_loader=loader,
        teacher_model=teacher,
        eval_fn=eval_fn,
        train_secs=0.0001,
    )
    assert "pre_metric" in res
    assert "post_metric" in res
    assert "train_secs" in res
