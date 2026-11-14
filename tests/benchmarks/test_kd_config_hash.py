from neuralops.benchmarks.kd.config import BenchmarkConfig


def test_kd_config_hash_covers_all_fields():
    cfg1 = BenchmarkConfig(seed=42)
    cfg2 = BenchmarkConfig(seed=999)  # seed was previously excluded in hardcoded dict

    h1 = cfg1.compute_hash()
    h2 = cfg2.compute_hash()
    assert h1 != h2
    assert len(h1) == 16
