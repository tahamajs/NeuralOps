import pytest
import torch
import torch.nn as nn
from framework.attentions.smart_weighted_ensemble import (
    SmartWeightedEnsemble, TaskAwareAttention)


class MockClassifier(nn.Module):
    def __init__(self, vocab_size=50, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.fc = nn.Linear(16, num_classes)
        self.last_attention_mask = None

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.last_attention_mask = attention_mask
        x = self.emb(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        pooled = x.mean(dim=1)
        return self.fc(pooled)


def test_smart_weighted_ensemble_all_methods():
    m1 = MockClassifier()
    m2 = MockClassifier()
    m3 = MockClassifier()

    ensemble = SmartWeightedEnsemble(models=[m1, m2, m3], num_tasks=3)

    B, L = 4, 8
    input_ids = torch.randint(0, 50, (B, L))
    mask = torch.ones(B, L)

    # 1. Simple average
    out_avg = ensemble(input_ids, attention_mask=mask, method="simple_average")
    assert out_avg.shape == (B, 2)
    assert not torch.isnan(out_avg).any()

    # 2. Weighted by performance
    out_w = ensemble(input_ids, attention_mask=mask, method="weighted")
    assert out_w.shape == (B, 2)
    assert not torch.isnan(out_w).any()

    # 3. Task-aware attention
    out_attn = ensemble(input_ids, attention_mask=mask, method="task_aware_attention", model_task_ids=[0, 1, 2])
    assert out_attn.shape == (B, 2)
    assert not torch.isnan(out_attn).any()

    # 4. Hybrid
    out_hyb = ensemble(input_ids, attention_mask=mask, method="hybrid", model_task_ids=[0, 1, 2])
    assert out_hyb.shape == (B, 2)
    assert not torch.isnan(out_hyb).any()


def test_attention_mask_propagation():
    m1 = MockClassifier()
    m2 = MockClassifier()

    ensemble = SmartWeightedEnsemble(models=[m1, m2], num_tasks=2)

    input_ids = torch.randint(0, 50, (2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    ensemble(input_ids, attention_mask=mask, method="hybrid")

    assert m1.last_attention_mask is not None
    assert torch.equal(m1.last_attention_mask, mask)
    assert m2.last_attention_mask is not None
    assert torch.equal(m2.last_attention_mask, mask)


def test_model_task_ids_validation():
    m1 = MockClassifier()
    m2 = MockClassifier()
    ensemble = SmartWeightedEnsemble(models=[m1, m2], num_tasks=2)

    input_ids = torch.randint(0, 50, (2, 4))
    # Passing 3 task IDs for 2 models must raise ValueError
    with pytest.raises(ValueError, match="model_task_ids length"):
        ensemble(input_ids, method="task_aware_attention", model_task_ids=[0, 1, 2])


def test_trainable_dynamic_gradients():
    m1 = MockClassifier()
    m2 = MockClassifier()
    ensemble = SmartWeightedEnsemble(models=[m1, m2], num_tasks=2, trainable_dynamic=True)

    input_ids = torch.randint(0, 50, (2, 4))
    mask = torch.ones(2, 4)

    out = ensemble(input_ids, attention_mask=mask, method="hybrid", model_task_ids=[0, 1])
    loss = out.sum()
    loss.backward()

    # Sub-models should receive gradients
    assert m1.fc.weight.grad is not None and m1.fc.weight.grad.abs().sum() > 0
    assert m2.fc.weight.grad is not None and m2.fc.weight.grad.abs().sum() > 0
    # Task-aware attention should also have gradients
    assert ensemble.attention.task_emb.weight.grad is not None


def test_update_performance():
    m1 = MockClassifier()
    m2 = MockClassifier()
    ensemble = SmartWeightedEnsemble(models=[m1, m2], num_tasks=2)

    logits1 = torch.tensor([[2.0, -1.0], [-1.0, 3.0]])
    logits2 = torch.tensor([[-1.0, 2.0], [2.0, -1.0]])
    targets = torch.tensor([0, 1])

    old_perf = ensemble.performance.clone()
    ensemble.update_performance(task_id=0, model_outputs=[logits1, logits2], target=targets)

    # Performance for task 0 should have updated
    assert not torch.allclose(ensemble.performance[:, 0], old_perf[:, 0])
    # Model 1 was 100% correct, so performance should increase
    assert ensemble.performance[0, 0] > old_perf[0, 0]
