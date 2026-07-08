import logging

from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import (Recommendation, ReputationEvent, ReputationScore,
                     ReputationSnapshot, ReputationWeight)
from .serializers import (RecommendationSerializer, ReputationEventSerializer,
                          ReputationScoreSerializer,
                          ReputationSnapshotSerializer,
                          ReputationWeightSerializer)

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Weights Configuration (Admin / Staff)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def get_weights(request):
    """Return the current reputation weight configuration."""
    weights = ReputationWeight.get_weights()
    return Response(weights)


@api_view(["PUT"])
@permission_classes([IsAuthenticated])
def update_weights(request):
    """Update the reputation weight configuration (staff only)."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)

    weights, _ = ReputationWeight.objects.get_or_create(pk=1)
    for field in ["OTHER_READS", "FULL_TEXT_READS", "RECOMMENDATIONS", "CITATIONS"]:
        if field in request.data:
            setattr(weights, field, request.data[field])
    weights.save()
    return Response(ReputationWeightSerializer(weights).data)


# ═══════════════════════════════════════════════════════════════
# Reputation Scores & Leaderboard
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def my_reputation(request):
    """Get the authenticated user's reputation score."""
    score, _ = ReputationScore.objects.get_or_create(user=request.user)
    return Response(ReputationScoreSerializer(score).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def user_reputation(request, username):
    """Get any user's reputation score by username."""
    user = get_object_or_404(User, username=username)
    score, _ = ReputationScore.objects.get_or_create(user=user)
    return Response(ReputationScoreSerializer(score).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def reputation_leaderboard(request):
    """Global reputation leaderboard with pagination."""
    qs = ReputationScore.objects.select_related("user").order_by("-total_score")

    # Optionally filter by minimum score
    min_score = request.query_params.get("min_score")
    if min_score:
        qs = qs.filter(total_score__gte=float(min_score))

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ReputationScoreSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Reputation Events (History)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_events(request):
    """List the authenticated user's reputation events."""
    qs = request.user.reputation_events.order_by("-created_at")

    event_type = request.query_params.get("event_type")
    if event_type:
        qs = qs.filter(event_type=event_type)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ReputationEventSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_recommendations(request):
    """Get context-aware recommendations for the authenticated user."""
    qs = (
        Recommendation.objects.filter(user=request.user, is_dismissed=False)
        .select_related("paper")
        .order_by("-score")
    )

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        RecommendationSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, recommendation_id):
    """Dismiss a recommendation."""
    rec = get_object_or_404(Recommendation, id=recommendation_id, user=request.user)
    rec.is_dismissed = True
    rec.save(update_fields=["is_dismissed"])
    return Response({"status": "dismissed"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_recommendations(request):
    """
    Generate context-aware recommendations for a user.
    Based on their saved papers, reputation events, and semantic similarity.
    This is a placeholder – in production, you'd use an LLM or embedding-based similarity.
    """
    from agents.models import ResearchPaper

    # Get user's saved papers
    saved_papers = ResearchPaper.objects.filter(saved_by=request.user).values_list(
        "id", flat=True
    )

    if not saved_papers:
        return Response(
            {"message": "Save some papers first to get recommendations.", "count": 0}
        )

    # Find papers that users with similar interests saved (co-occurrence)
    similar_users = (
        ResearchPaper.objects.filter(id__in=saved_papers)
        .values_list("saved_by", flat=True)
        .distinct()
    )

    recommended = (
        ResearchPaper.objects.filter(saved_by__in=similar_users)
        .exclude(id__in=saved_papers)
        .distinct()
        .order_by("-citation_count")[:20]
    )

    created = 0
    for paper in recommended:
        # Calculate a simple recommendation score based on citation count and shared users
        shared_count = ResearchPaper.objects.filter(
            id__in=saved_papers, saved_by__in=[request.user]
        ).count()
        score = min(1.0, (paper.citation_count or 0) / 1000.0 + shared_count * 0.1)
        rec, is_new = Recommendation.objects.get_or_create(
            user=request.user,
            paper=paper,
            defaults={
                "score": round(score, 2),
                "reason": "Based on your saved papers and interests",
            },
        )
        if is_new:
            created += 1

    return Response(
        {"message": f"Generated {created} new recommendations.", "count": created}
    )


# ═══════════════════════════════════════════════════════════════
# Reputation Events (Create)
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def record_event(request):
    """
    Record a reputation event.
    Body: { "event_type": "citation", "paper_id": "...", "description": "..." }
    """
    event_type = request.data.get("event_type")
    if event_type not in dict(ReputationEvent.EVENT_TYPES):
        return Response(
            {
                "error": f"Invalid event_type. Choose from: {list(dict(ReputationEvent.EVENT_TYPES).keys())}"
            },
            status=400,
        )

    paper_id = request.data.get("paper_id")
    paper = None
    if paper_id:
        from agents.models import ResearchPaper

        try:
            paper = ResearchPaper.objects.get(id=paper_id)
        except ResearchPaper.DoesNotExist:
            return Response({"error": "Paper not found"}, status=404)

    event = ReputationEvent.objects.create(
        user=request.user,
        event_type=event_type,
        paper=paper,
        source_user=request.data.get("source_user_id")
        and User.objects.get(id=request.data["source_user_id"])
        or None,
        description=request.data.get("description", ""),
    )

    return Response(ReputationEventSerializer(event).data, status=201)


# ═══════════════════════════════════════════════════════════════
# Reputation Snapshots (Historical)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_snapshots(request):
    """Get the authenticated user's daily reputation snapshots."""
    days = int(request.query_params.get("days", 30))
    from datetime import timedelta

    cutoff = timezone.now().date() - timedelta(days=days)

    snapshots = ReputationSnapshot.objects.filter(
        user=request.user, recorded_at__gte=cutoff
    ).order_by("recorded_at")

    return Response(ReputationSnapshotSerializer(snapshots, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def take_snapshot(request):
    """Take a daily snapshot of all reputation scores (staff only)."""
    if not request.user.is_staff:
        return Response({"error": "Staff only"}, status=403)

    scores = ReputationScore.objects.select_related("user").all()
    today = timezone.now().date()

    count = 0
    for score in scores:
        _, created = ReputationSnapshot.objects.get_or_create(
            user=score.user,
            recorded_at=today,
            defaults={
                "total_score": score.total_score,
                "other_reads_score": score.other_reads_score,
                "full_text_reads_score": score.full_text_reads_score,
                "recommendations_score": score.recommendations_score,
                "citations_score": score.citations_score,
                "percentile": score.percentile,
            },
        )
        if created:
            count += 1

    # Update percentiles
    total = ReputationScore.objects.count()
    for i, score in enumerate(ReputationScore.objects.order_by("-total_score"), 1):
        score.percentile = round((1 - i / total) * 100, 2)
        score.save(update_fields=["percentile"])

    return Response({"snapshots_created": count})
