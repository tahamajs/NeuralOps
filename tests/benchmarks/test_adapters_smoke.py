import pytest
import neuralops.benchmarks.adapters as ad


def test_adapters_canonical_lists():
    assert len(ad.CANONICAL_TRITON_MODULES) == 26
    assert len(ad.CANONICAL_CUDA_MODULES) > 0


def test_adapters_instantiation():
    adapters = [
        ad.AttentionBenchmarkAdapter,
        ad.ClassificationAdapter,
        ad.CompressorModuleAdapter,
        ad.PerplexityAdapter,
        ad.RulerAdapter,
        ad.ZeroShotAdapter,
    ]
    for cls in adapters:
        assert callable(cls)
