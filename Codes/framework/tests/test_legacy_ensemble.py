import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, "Codes")


def test_legacy_import_works():
    """Legacy imports still work but emit deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from framework.attentions.legacy_ensemble import (
            SmartWeightedEnsemble, TaskAwareAttention)
        assert len(w) >= 1, "Expected DeprecationWarning"
    print("[OK] test_legacy_import_works")


def test_task_aware_attention_basic():
    """TaskAwareAttention construction and forward."""
    from framework.attentions.legacy_ensemble import TaskAwareAttention
    block = TaskAwareAttention(num_tasks=3, hidden_dim=32)
    all_logits = torch.randn(2, 4, 3)
    model_task_ids = [0, 1]
    out = block(0, all_logits, model_task_ids)
    assert out.shape == (4, 3)
    print("[OK] test_task_aware_attention_basic")


def test_smart_ensemble_basic():
    """SmartWeightedEnsemble with two dummy models."""
    from framework.attentions.legacy_ensemble import SmartWeightedEnsemble

    class DummyClassifier(nn.Module):
        def __init__(self, num_labels=2):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.cls = nn.Linear(16, num_labels)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            x = self.emb(input_ids).mean(dim=1)
            return self.cls(x)

    m1 = DummyClassifier()
    m2 = DummyClassifier()
    ens = SmartWeightedEnsemble([m1, m2], num_tasks=1)

    input_ids = torch.randint(0, 100, (4, 8))
    out = ens(input_ids, method="simple_average")
    assert out.shape == (4, 2)
    print("[OK] test_smart_ensemble_basic")


if __name__ == "__main__":
    tests = [
        test_legacy_import_works,
        test_task_aware_attention_basic,
        test_smart_ensemble_basic,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"PASSED: {passed}/{len(tests)}")
    print(f"FAILED: {failed}/{len(tests)}")
    sys.exit(0 if failed == 0 else 1)
