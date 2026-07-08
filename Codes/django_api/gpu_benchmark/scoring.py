"""
DLPerf-style Composite Scoring Engine.

Normalizes raw benchmark results to a 0-100 scale using reference baselines
and computes a weighted composite score for cross-GPU comparison.

Reference baselines are derived from public MLPerf results and Vast.ai's
internal benchmarks, and can be updated as new hardware emerges.
"""

from .models import BenchmarkResult

# ═══════════════════════════════════════════════════════════════
# Reference Baselines (updated from MLPerf v4.0 / Vast.ai)
# These are the "100 points" reference — the best known result
# for each GPU model on each benchmark.
# ═══════════════════════════════════════════════════════════════
REFERENCE_BASELINES = {
    # (gpu_model, test_name, batch_size): metric_value
    # Training throughput (samples/sec)
    ("H100-SXM-80GB", "ResNet-50 Training", 256): 5000.0,
    ("A100-SXM4-80GB", "ResNet-50 Training", 256): 2800.0,
    ("RTX 4090", "ResNet-50 Training", 128): 1800.0,
    ("RTX 3090", "ResNet-50 Training", 128): 1100.0,
    ("RTX 3080", "ResNet-50 Training", 64): 700.0,
    ("H100-SXM-80GB", "BERT-Large Training", 64): 800.0,
    ("A100-SXM4-80GB", "BERT-Large Training", 64): 450.0,
    ("RTX 4090", "BERT-Large Training", 32): 280.0,
    # Inference throughput (samples/sec)
    ("H100-SXM-80GB", "ResNet-50 Inference", 256): 25000.0,
    ("A100-SXM4-80GB", "ResNet-50 Inference", 256): 14000.0,
    ("RTX 4090", "ResNet-50 Inference", 128): 9000.0,
    # LLM tokens/sec
    ("H100-SXM-80GB", "Llama-2-7B Inference", 32): 5000.0,
    ("A100-SXM4-80GB", "Llama-2-7B Inference", 32): 2800.0,
    ("RTX 4090", "Llama-2-7B Inference", 16): 1500.0,
}

# When no exact baseline exists, use a generic fallback per GPU tier
GPU_TIER_BASELINE = {
    "H100": 100.0,
    "H200": 120.0,
    "B200": 150.0,
    "A100": 80.0,
    "A6000": 60.0,
    "A5000": 40.0,
    "RTX 4090": 55.0,
    "RTX 3090": 35.0,
    "RTX 3080": 25.0,
    "RTX 3070": 18.0,
    "RTX 3060": 12.0,
}


def normalize_result(result: BenchmarkResult) -> float:
    """
    Normalize a raw benchmark result to a 0-100 scale.
    - Find the best reference baseline for this GPU model + test + batch_size.
    - If no exact match, use GPU tier baseline.
    - For 'maximize' metrics: score = (value / baseline) * 100
    - For 'minimize' metrics: score = (baseline / value) * 100
    - Clamp to [0, 200] to avoid extreme outliers.
    """
    gpu_model = result.listing.gpu_model
    test_name = result.test.name
    batch_size = result.batch_size
    metric_direction = result.test.metric_direction
    raw_value = result.metric_value

    if raw_value is None or raw_value <= 0:
        return 0.0

    # Try exact baseline match
    baseline = REFERENCE_BASELINES.get((gpu_model, test_name, batch_size))

    # Try partial match (any batch size for this GPU+test)
    if baseline is None:
        for (gm, tn, bs), val in REFERENCE_BASELINES.items():
            if gm == gpu_model and tn == test_name:
                baseline = val
                break

    # Fallback to GPU tier
    if baseline is None:
        for tier_prefix, tier_val in GPU_TIER_BASELINE.items():
            if gpu_model.startswith(tier_prefix):
                baseline = tier_val * 10  # scale to match
                break

    # Last resort
    if baseline is None or baseline <= 0:
        baseline = 100.0

    if metric_direction == "maximize":
        score = (raw_value / baseline) * 100.0
    else:
        score = (baseline / raw_value) * 100.0

    return max(0.0, min(200.0, round(score, 2)))


def compute_composite_score(results) -> float:
    """
    Compute a weighted composite score from a set of BenchmarkResults.
    Returns a score on a 0-100 scale.
    """
    if not results:
        return None

    total_weight = 0.0
    weighted_sum = 0.0

    for r in results:
        w = r.test.weight
        s = (
            r.normalized_score
            if r.normalized_score is not None
            else normalize_result(r)
        )
        weighted_sum += s * w
        total_weight += w

    if total_weight == 0:
        return None

    return round(weighted_sum / total_weight, 2)
