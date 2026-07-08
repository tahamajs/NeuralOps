# leaderboards/views.py
"""
Leaderboard REST API — full CRUD for benchmarks, submissions, scores, votes, badges.
"""

import logging

from django.db.models import Count, F, Q, Window
from django.db.models.functions import DenseRank
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# BENCHMARKS — List / Create
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def benchmark_list(request):
    if request.method == "GET":
        qs = Benchmark.objects.annotate(submission_count=Count("submissions"))
        # Filters
        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        visibility = request.query_params.get("visibility")
        if visibility:
            qs = qs.filter(visibility=visibility)
        category = request.query_params.get("category")
        if category:
            qs = qs.filter(tasks__category=category).distinct()
        search = request.query_params.get("q")
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))
        # Default: show public + user's own private
        if not request.user.is_staff:
            qs = qs.filter(
                Q(visibility="public")
                | Q(created_by=request.user)
                | Q(collaborators=request.user)
            ).distinct()
        qs = qs.order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            BenchmarkListSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = BenchmarkDetailSerializer(data=request.data)
        if serializer.is_valid():
            benchmark = serializer.save(created_by=request.user)
            return Response(BenchmarkDetailSerializer(benchmark).data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# BENCHMARK — Detail / Update / Delete
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def benchmark_detail(request, benchmark_id):
    benchmark = get_object_or_404(
        Benchmark.objects.annotate(submission_count=Count("submissions")),
        id=benchmark_id,
    )
    if request.method == "GET":
        return Response(BenchmarkDetailSerializer(benchmark).data)
    elif request.method == "PATCH":
        if benchmark.created_by != request.user and not request.user.is_staff:
            return Response({"error": "Only the owner can edit"}, status=403)
        serializer = BenchmarkDetailSerializer(
            benchmark, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if benchmark.created_by != request.user and not request.user.is_staff:
            return Response({"error": "Only the owner can delete"}, status=403)
        benchmark.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# LEADERBOARD — Get ranked submissions for a benchmark
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def leaderboard_view(request, benchmark_id):
    """Return the full leaderboard for a benchmark with rankings."""
    benchmark = get_object_or_404(Benchmark, id=benchmark_id)
    public_only = request.query_params.get("public", "false").lower() == "true"
    metric_slug = request.query_params.get("metric")
    page = int(request.query_params.get("page", 1))
    page_size = min(int(request.query_params.get("page_size", 25)), 100)

    submissions = benchmark.submissions.filter(status="completed")
    if public_only:
        submissions = submissions.filter(public_score__isnull=False)

    # Rank by composite_score (or by a specific metric)
    order_field = "composite_score"
    if metric_slug:
        submissions = submissions.filter(scores__metric__slug=metric_slug).distinct()
        order_field = "scores__value"

    submissions = submissions.annotate(
        rank=Window(expression=DenseRank(), order_by=F(order_field).desc())
    ).order_by("rank")

    # Compute total count for pagination
    total = submissions.count()
    start = (page - 1) * page_size
    end = start + page_size

    entries = []
    for s in submissions[start:end]:
        entries.append(
            {
                "rank": s.rank,
                "model_name": s.model_name,
                "username": s.user.username,
                "team_name": s.team_name or s.user.username,
                "composite_score": s.composite_score,
                "public_score": s.public_score,
                "private_score": (
                    s.private_score if benchmark.final_results_published else None
                ),
                "parameters_million": s.parameters_million,
                "submission_id": str(s.id),
                "created_at": s.created_at,
                "scores": {
                    sc.metric.slug: sc.value for sc in s.scores.filter(is_public=True)
                },
            }
        )

    # Get requesting user's own rank
    my_rank = None
    if request.user.is_authenticated:
        my_sub = submissions.filter(user=request.user).first()
        if my_sub:
            my_rank = my_sub.rank

    return Response(
        {
            "benchmark": benchmark.name,
            "total_submissions": total,
            "page": page,
            "page_size": page_size,
            "entries": entries,
            "my_rank": my_rank,
        }
    )


# ═══════════════════════════════════════════════════════════════
# SUBMISSIONS — Submit / List / Detail
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def submission_list(request, benchmark_id):
    benchmark = get_object_or_404(Benchmark, id=benchmark_id)
    if request.method == "GET":
        qs = benchmark.submissions.all().order_by("-created_at")
        # Filter by user
        user_filter = request.query_params.get("user")
        if user_filter == "me":
            qs = qs.filter(user=request.user)
        elif user_filter:
            qs = qs.filter(user__username=user_filter)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            SubmissionSerializer(page, many=True).data
        )

    elif request.method == "POST":
        # Check if benchmark is open
        if not benchmark.is_open and not request.user.is_staff:
            return Response(
                {"error": "Benchmark is not currently accepting submissions"},
                status=400,
            )
        # Check submission limit
        user_count = benchmark.submissions.filter(user=request.user).count()
        if (
            user_count >= benchmark.submission_limit_per_user
            and not request.user.is_staff
        ):
            return Response(
                {
                    "error": f"Submission limit ({benchmark.submission_limit_per_user}) reached"
                },
                status=400,
            )

        serializer = SubmissionSerializer(data=request.data)
        if serializer.is_valid():
            submission = serializer.save(benchmark=benchmark, user=request.user)

            # Save scores if provided
            scores_data = request.data.get("scores", [])
            for score_data in scores_data:
                Score.objects.create(
                    submission=submission,
                    metric_id=score_data.get("metric_id"),
                    dataset_id=score_data.get("dataset_id"),
                    value=score_data.get("value"),
                    is_public=score_data.get("is_public", True),
                )

            # Compute composite score
            _update_composite_score(submission)

            # Award badges
            _check_and_award_badges(benchmark, request.user, submission)

            return Response(SubmissionSerializer(submission).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def submission_detail(request, benchmark_id, submission_id):
    submission = get_object_or_404(
        Submission, id=submission_id, benchmark_id=benchmark_id
    )
    if request.method == "GET":
        return Response(SubmissionSerializer(submission).data)
    elif request.method == "PATCH":
        if submission.user != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        serializer = SubmissionSerializer(submission, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if submission.user != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        submission.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# MY RANK
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_rank(request, benchmark_id):
    """Get the authenticated user's rank and best submission for a benchmark."""
    benchmark = get_object_or_404(Benchmark, id=benchmark_id)
    best_submission = (
        benchmark.submissions.filter(user=request.user, status="completed")
        .order_by("-composite_score")
        .first()
    )
    if not best_submission:
        return Response({"rank": None, "has_submission": False})

    ranked = (
        benchmark.submissions.filter(status="completed")
        .annotate(
            rank=Window(expression=DenseRank(), order_by=F("composite_score").desc())
        )
        .filter(id=best_submission.id)
        .first()
    )

    return Response(
        {
            "rank": ranked.rank if ranked else None,
            "total_participants": benchmark.submissions.filter(
                status="completed"
            ).count(),
            "best_score": best_submission.composite_score,
            "submission_id": str(best_submission.id),
            "has_submission": True,
        }
    )


# ═══════════════════════════════════════════════════════════════
# VOTING
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def vote_submission(request, benchmark_id, submission_id):
    submission = get_object_or_404(
        Submission, id=submission_id, benchmark_id=benchmark_id
    )
    direction = request.data.get("direction")
    if direction not in ("up", "down"):
        return Response({"error": "direction must be 'up' or 'down'"}, status=400)

    vote, created = Vote.objects.update_or_create(
        submission=submission, user=request.user, defaults={"direction": direction}
    )
    return Response(
        {
            "status": "voted",
            "direction": direction,
            "upvotes": submission.votes.filter(direction="up").count(),
            "downvotes": submission.votes.filter(direction="down").count(),
        }
    )


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_vote(request, benchmark_id, submission_id):
    submission = get_object_or_404(
        Submission, id=submission_id, benchmark_id=benchmark_id
    )
    Vote.objects.filter(submission=submission, user=request.user).delete()
    return Response({"status": "vote removed"})


# ═══════════════════════════════════════════════════════════════
# BADGES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def badge_list(request, benchmark_id=None):
    qs = Badge.objects.all()
    if benchmark_id:
        qs = qs.filter(benchmark_id=benchmark_id)
    return Response(BadgeSerializer(qs, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_badges(request):
    badges = UserBadge.objects.filter(user=request.user).select_related(
        "badge", "submission__benchmark"
    )
    return Response(UserBadgeSerializer(badges, many=True).data)


# ═══════════════════════════════════════════════════════════════
# FOLLOW / UNFOLLOW
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def follow_benchmark(request, benchmark_id):
    benchmark = get_object_or_404(Benchmark, id=benchmark_id)
    follow, created = BenchmarkFollow.objects.get_or_create(
        user=request.user, benchmark=benchmark
    )
    return Response({"status": "following", "created": created})


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def unfollow_benchmark(request, benchmark_id):
    BenchmarkFollow.objects.filter(
        user=request.user, benchmark_id=benchmark_id
    ).delete()
    return Response({"status": "unfollowed"})


# ═══════════════════════════════════════════════════════════════
# TASKS / DATASETS / METRICS (Reference data)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def task_list(request):
    qs = BenchmarkTask.objects.all()
    category = request.query_params.get("category")
    if category:
        qs = qs.filter(category=category)
    return Response(BenchmarkTaskSerializer(qs, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dataset_list(request):
    qs = BenchmarkDataset.objects.all()
    task_slug = request.query_params.get("task")
    if task_slug:
        qs = qs.filter(task__slug=task_slug)
    return Response(BenchmarkDatasetSerializer(qs, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def metric_list(request):
    return Response(MetricSerializer(Metric.objects.all(), many=True).data)


# ═══════════════════════════════════════════════════════════════
# GLOBAL STATS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def global_stats(request):
    return Response(
        {
            "total_benchmarks": Benchmark.objects.filter(visibility="public").count(),
            "total_submissions": Submission.objects.count(),
            "total_users_ranked": Submission.objects.filter(status="completed")
            .values("user")
            .distinct()
            .count(),
            "top_benchmarks": list(
                Benchmark.objects.filter(visibility="public")
                .annotate(sc=Count("submissions"))
                .order_by("-sc")[:5]
                .values("id", "name", "slug", "sc")
            ),
        }
    )


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def _update_composite_score(submission):
    """Compute and save the composite score from all public scores."""
    scores = submission.scores.filter(is_public=True)
    if not scores.exists():
        submission.composite_score = None
        submission.save(update_fields=["composite_score"])
        return

    benchmark = submission.benchmark
    memberships = {
        m.metric_id: m.weight
        for m in BenchmarkMetricMembership.objects.filter(benchmark=benchmark)
    }
    total = 0.0
    total_weight = 0.0
    for sc in scores:
        w = memberships.get(sc.metric_id, 1.0)
        total += sc.value * w
        total_weight += w
    submission.composite_score = total / max(total_weight, 0.001)
    submission.public_score = submission.composite_score
    submission.save(update_fields=["composite_score", "public_score"])


def _check_and_award_badges(benchmark, user, submission):
    """Check badge conditions and award if met."""
    for badge in benchmark.badges.all():
        condition = badge.condition or {}
        rank_required = condition.get("rank")
        if rank_required:
            user_rank = submission.rank
            if user_rank and user_rank <= rank_required:
                UserBadge.objects.get_or_create(
                    user=user, badge=badge, submission=submission
                )
