"""Performance baseline tests (sanity timing benchmarks)."""

import subprocess
import sys
import time
import pytest
import torch


@pytest.mark.performance
class TestPerformanceBaselines:
    """Performance sanity assertions ensuring sub-second lookups and execution."""

    def test_attention_forward_under_5s(self):
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401

        cls = ATTENTION_REGISTRY.get("svd_linear_attention")
        if cls is None:
            pytest.skip("Module not available")

        attn = cls(hidden_size=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        start = time.time()
        with torch.no_grad():
            _ = attn(x)
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_registry_lookup_is_fast(self):
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401

        start = time.time()
        for _ in range(1000):
            _ = ATTENTION_REGISTRY.get("svd_linear_attention")
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_import_time_acceptable(self):
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-c", "import neuralops"],
            capture_output=True,
            timeout=30,
        )
        elapsed = time.time() - start
        assert result.returncode == 0
        assert elapsed < 15.0
