"""Celery tasks for running benchmarks and updating scores."""

import logging
import random

from celery import shared_task
from django.utils import timezone

from .models import BenchmarkResult, BenchmarkRun, BenchmarkTest, ListingBenchmarkScore
from .scoring import normalize_result

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def run_benchmark_suite(self, run_id: str):
    """
    Execute a full benchmark suite against a GPU listing.
    In production, this would:
    1. Connect to the provider's Docker agent
    2. Pull the benchmark Docker image
    3. Run each test with specified batch sizes and precisions
    4. Collect results
    5. Normalize and compute composite score

    For now, this uses simulated results with realistic values.
    """
    try:
        run = BenchmarkRun.objects.get(id=run_id)
    except BenchmarkRun.DoesNotExist:
        return

    run.status = "running"
    run.started_at = timezone.now()
    run.save(update_fields=["status", "started_at"])

    try:
        tests = run.suite.tests.filter(is_active=True)
        batch_sizes = run.batch_sizes or [8, 32, 128]
        precisions = run.precisions or ["fp32", "fp16"]

        for test in tests:
            for bs in batch_sizes:
                for prec in precisions:
                    # Simulate benchmark execution
                    result = _simulate_benchmark(test, run.listing, bs, prec)

                    # Normalize the score
                    normalized = normalize_result(result)
                    result.normalized_score = normalized
                    result.save(update_fields=["normalized_score"])

        run.status = "completed"
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "completed_at"])

        # Update composite score
        score, _ = ListingBenchmarkScore.objects.get_or_create(listing=run.listing)
        score.last_run = run
        score.total_runs = BenchmarkRun.objects.filter(
            listing=run.listing, status="completed"
        ).count()
        score.recalculate()

        # Update the listing's dlperf_score
        run.listing.dlperf_score = score.composite_score
        run.listing.save(update_fields=["dlperf_score"])

    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "error_message", "completed_at"])
        logger.exception("Benchmark run failed")


def _simulate_benchmark(
    test: BenchmarkTest, listing, batch_size: int, precision: str
) -> BenchmarkResult:
    """
    Simulate a benchmark result with realistic values based on GPU model.
    In production, this would actually run the benchmark in Docker.
    """
    gpu = listing.gpu_model

    # Rough performance tiers
    tiers = {
        "H200": 1.4,
        "B200": 1.6,
        "H100": 1.0,
        "A100": 0.65,
        "RTX 4090": 0.42,
        "RTX 3090": 0.28,
        "RTX 3080": 0.20,
        "RTX 3070": 0.13,
        "RTX 3060": 0.08,
    }
    multiplier = 1.0
    for prefix, factor in tiers.items():
        if gpu.startswith(prefix):
            multiplier = factor
            break

    # FP16 is ~2x throughput vs FP32 on most GPUs
    if precision in ("fp16", "bf16"):
        multiplier *= 1.8

    # Batch size scaling: larger batches → higher throughput up to VRAM limit
    vram = listing.vram_gb
    max_bs = min(batch_size * 4, int(vram * 8))
    bs_factor = min(1.0, batch_size / max(max_bs, 1)) * 1.2 + 0.3

    base = 100.0 * multiplier * bs_factor

    result = BenchmarkResult.objects.create(
        run=(
            BenchmarkRun.objects.get(id=test.suite.runs.last().id)
            if hasattr(test, "_run_id")
            else None
        ),
        run_id=_get_current_run_id(),
        test=test,
        listing=listing,
        batch_size=batch_size,
        precision=precision,
        metric_value=round(base + random.uniform(-10, 10), 2),
        metric_std_dev=round(random.uniform(1, 5), 2),
        peak_memory_mb=round(vram * 1024 * 0.7 + random.uniform(-500, 500), 0),
        gpu_utilization_pct=round(min(99, 85 + random.uniform(0, 14)), 1),
        cpu_utilization_pct=round(random.uniform(20, 60), 1),
        power_watts=round(200 * multiplier + random.uniform(-30, 30), 1),
        is_valid=True,
    )
    return result


def _get_current_run_id():
    """Helper to get the current benchmark run ID during simulation."""
    import inspect

    for frame in inspect.stack():
        if "run_id" in frame.frame.f_locals:
            return frame.frame.f_locals["run_id"]
    return None


@shared_task
def auto_benchmark_new_listings():
    """Automatically benchmark newly registered GPU listings."""
    from gpu_marketplace.models import GPUListing

    new_listings = GPUListing.objects.filter(
        status="online", benchmark_runs__isnull=True
    )[
        :10
    ]  # limit per cycle

    default_suite = BenchmarkSuite.objects.filter(
        name__icontains="DLPerf", is_active=True
    ).first()

    if not default_suite:
        return

    for listing in new_listings:
        run = BenchmarkRun.objects.create(
            listing=listing,
            suite=default_suite,
            batch_sizes=[8, 32, 128],
            precisions=["fp32", "fp16"],
        )
        run_benchmark_suite.delay(str(run.id))
