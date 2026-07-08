"""
GPU Benchmark REST API — MLPerf-style benchmarking for GPU listings.
"""

import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *
from .tasks import run_benchmark_suite

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# BENCHMARK SUITES & TESTS (Reference Data)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def suite_list(request):
    """List all available benchmark suites with their tests."""
    qs = BenchmarkSuite.objects.filter(is_active=True).prefetch_related("tests")
    suite_type = request.query_params.get("suite_type")
    if suite_type:
        qs = qs.filter(suite_type=suite_type)
    return Response(BenchmarkSuiteSerializer(qs, many=True).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def suite_detail(request, suite_id):
    """Get details of a specific benchmark suite."""
    suite = get_object_or_404(BenchmarkSuite, id=suite_id)
    return Response(BenchmarkSuiteSerializer(suite).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def test_list(request):
    """List all benchmark tests (filterable by suite and category)."""
    qs = BenchmarkTest.objects.filter(is_active=True).select_related("suite")
    suite_id = request.query_params.get("suite_id")
    if suite_id:
        qs = qs.filter(suite_id=suite_id)
    category = request.query_params.get("category")
    if category:
        qs = qs.filter(category=category)
    return Response(BenchmarkTestSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# BENCHMARK RUNS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_benchmark(request, listing_id):
    """Start a benchmark run for a specific GPU listing."""
    listing = get_object_or_404(GPUListing, id=listing_id)

    suite_id = request.data.get("suite_id")
    if not suite_id:
        return Response({"error": "suite_id required"}, status=400)

    suite = get_object_or_404(BenchmarkSuite, id=suite_id, is_active=True)

    # Check if listing is online
    if listing.status != "online":
        return Response({"error": "Listing must be online to benchmark"}, status=400)

    # Check if a run is already in progress
    if BenchmarkRun.objects.filter(
        listing=listing, status__in=["queued", "running"]
    ).exists():
        return Response(
            {"error": "A benchmark is already running for this listing"}, status=400
        )

    batch_sizes = request.data.get("batch_sizes", [])
    precisions = request.data.get("precisions", ["fp32", "fp16"])

    run = BenchmarkRun.objects.create(
        listing=listing,
        suite=suite,
        initiated_by=request.user,
        batch_sizes=batch_sizes,
        precisions=precisions,
    )

    # Trigger async execution
    run_benchmark_suite.delay(str(run.id))

    return Response(BenchmarkRunSerializer(run).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_list(request):
    """List benchmark runs for a listing or all runs."""
    qs = BenchmarkRun.objects.select_related("listing", "suite").order_by("-created_at")

    listing_id = request.query_params.get("listing_id")
    if listing_id:
        qs = qs.filter(listing_id=listing_id)

    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        BenchmarkRunSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_detail(request, run_id):
    """Get detailed results of a benchmark run."""
    run = get_object_or_404(BenchmarkRun, id=run_id)
    return Response(BenchmarkRunSerializer(run).data)


# ═══════════════════════════════════════════════════════════════
# BENCHMARK SCORES (Leaderboard-style)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def score_list(request):
    """List benchmark scores for all listings — GPU leaderboard."""
    qs = (
        ListingBenchmarkScore.objects.select_related("listing__provider")
        .filter(composite_score__isnull=False)
        .order_by("-composite_score")
    )

    gpu_model = request.query_params.get("gpu_model")
    if gpu_model:
        qs = qs.filter(listing__gpu_model__icontains=gpu_model)

    category = request.query_params.get("category")
    if category == "training":
        qs = qs.filter(training_score__isnull=False).order_by("-training_score")
    elif category == "inference":
        qs = qs.filter(inference_score__isnull=False).order_by("-inference_score")
    elif category == "llm":
        qs = qs.filter(llm_score__isnull=False).order_by("-llm_score")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ListingBenchmarkScoreSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def score_detail(request, listing_id):
    """Get the benchmark score for a specific listing."""
    score = get_object_or_404(ListingBenchmarkScore, listing_id=listing_id)
    return Response(ListingBenchmarkScoreSerializer(score).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def recalculate_score(request, listing_id):
    """Manually trigger score recalculation for a listing."""
    listing = get_object_or_404(GPUListing, id=listing_id)
    score, _ = ListingBenchmarkScore.objects.get_or_create(listing=listing)
    score.recalculate()
    return Response(ListingBenchmarkScoreSerializer(score).data)


# ═══════════════════════════════════════════════════════════════
# GPU LEADERBOARD (Cross-model comparison)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def gpu_leaderboard(request):
    """GPU performance leaderboard — ranked by DLPerf composite score."""
    qs = (
        ListingBenchmarkScore.objects.filter(composite_score__isnull=False)
        .select_related("listing__provider")
        .order_by("-composite_score")
    )

    gpu_model = request.query_params.get("gpu_model")
    if gpu_model:
        qs = qs.filter(listing__gpu_model__icontains=gpu_model)

    min_score = request.query_params.get("min_score")
    if min_score:
        qs = qs.filter(composite_score__gte=float(min_score))

    limit = min(int(request.query_params.get("limit", 50)), 100)

    results = []
    for i, score in enumerate(qs[:limit], 1):
        results.append(
            {
                "rank": i,
                "gpu_model": score.listing.gpu_model,
                "provider": score.listing.provider.username,
                "composite_score": score.composite_score,
                "training_score": score.training_score,
                "inference_score": score.inference_score,
                "llm_score": score.llm_score,
                "best_throughput": score.best_throughput,
                "price_per_hour": str(score.listing.current_price_per_hour),
                "price_performance_ratio": (
                    round(
                        score.composite_score
                        / float(score.listing.current_price_per_hour),
                        2,
                    )
                    if score.composite_score and score.listing.current_price_per_hour
                    else None
                ),
                "listing_id": str(score.listing.id),
            }
        )

    return Response({"results": results, "total": qs.count()})


# ═══════════════════════════════════════════════════════════════
# PROVIDER BENCHMARK DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_benchmark_dashboard(request):
    """Provider's benchmark dashboard showing scores across all their listings."""
    from gpu_marketplace.models import GPUListing

    listings = GPUListing.objects.filter(provider=request.user)
    scores = []
    for listing in listings:
        try:
            score = listing.benchmark_score
            scores.append(
                {
                    "listing_id": str(listing.id),
                    "gpu_model": listing.gpu_model,
                    "composite_score": score.composite_score,
                    "best_throughput": score.best_throughput,
                    "total_runs": score.total_runs,
                    "last_run_at": (
                        score.last_run.created_at if score.last_run else None
                    ),
                }
            )
        except ListingBenchmarkScore.DoesNotExist:
            scores.append(
                {
                    "listing_id": str(listing.id),
                    "gpu_model": listing.gpu_model,
                    "composite_score": None,
                    "best_throughput": None,
                    "total_runs": 0,
                    "last_run_at": None,
                }
            )
    return Response({"listings": scores})
