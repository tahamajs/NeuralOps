import logging

from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import CollaboratorMatch, ResearcherInterestProfile
from .serializers import (
    CollaboratorMatchSerializer,
    ResearcherInterestProfileSerializer,
)
from .tasks import generate_matches_for_user

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Interest Profile
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_interest_profile(request):
    """Get the authenticated user's interest profile."""
    profile, _ = ResearcherInterestProfile.objects.get_or_create(user=request.user)
    serializer = ResearcherInterestProfileSerializer(profile)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def regenerate_interest_profile(request):
    """Regenerate the authenticated user's interest profile from their papers."""
    profile, _ = ResearcherInterestProfile.objects.get_or_create(user=request.user)
    # Trigger async task
    from .tasks import update_user_embedding

    update_user_embedding.delay(request.user.id)
    return Response({"status": "regeneration started"})


# ═══════════════════════════════════════════════════════════════
# Collaborator Matches
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
def suggested_matches(request):
    """Get suggested collaborator matches for the current user."""
    user = request.user
    if not getattr(user, "is_authenticated", False):
        user, _ = User.objects.get_or_create(
            username="researcher",
            defaults={"email": "researcher@neuralops.internal"},
        )
    # Ensure matches exist (generate on demand if needed)
    if not CollaboratorMatch.objects.filter(user=user).exists():
        try:
            generate_matches_for_user.delay(user.id)
        except Exception as e:
            logger.warning("Celery async task failed: %s", e)
            try:
                generate_matches_for_user(user.id)
            except Exception as inner_e:
                logger.warning("Synchronous generation failed: %s", inner_e)

    qs = CollaboratorMatch.objects.filter(
        user=user, status="suggested"
    ).order_by("-score")

    min_score = request.query_params.get("min_score")
    if min_score:
        try:
            qs = qs.filter(score__gte=float(min_score))
        except (ValueError, TypeError):
            pass

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        CollaboratorMatchSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def accept_match(request, match_id):
    """Accept a collaborator match."""
    match = get_object_or_404(CollaboratorMatch, id=match_id, user=request.user)
    match.status = "accepted"
    match.save(update_fields=["status", "updated_at"])
    return Response({"status": "accepted"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dismiss_match(request, match_id):
    """Dismiss a collaborator match."""
    match = get_object_or_404(CollaboratorMatch, id=match_id, user=request.user)
    match.status = "dismissed"
    match.save(update_fields=["status", "updated_at"])
    return Response({"status": "dismissed"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def match_history(request):
    """Get all matches (accepted, dismissed, suggested) for the current user."""
    qs = CollaboratorMatch.objects.filter(user=request.user).order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        CollaboratorMatchSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Cross-disciplinary suggestions
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def interdisciplinary_matches(request):
    """Get matches that are explicitly from different research domains."""
    # We'll flag matches where common_interests length is small, or where tags differ
    qs = CollaboratorMatch.objects.filter(
        user=request.user, status="suggested"
    ).order_by("-score")

    # Simple heuristic: interdisciplinary if common_interests list is short
    qs = qs.filter(common_interests__len__lte=2)  # few shared tags = different fields

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        CollaboratorMatchSerializer(page, many=True).data
    )
