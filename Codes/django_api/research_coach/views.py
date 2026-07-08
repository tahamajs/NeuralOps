import logging

from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import AnalysisResult, ChatSession, Recommendation
from .serializers import (AnalysisResultSerializer, ChatMessageSerializer,
                          ChatSessionSerializer, RecommendationSerializer)
from .tasks import run_periodic_analysis

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Analysis Results
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def analysis_list(request):
    """Get the authenticated user's analysis results."""
    qs = AnalysisResult.objects.filter(user=request.user).order_by("-created_at")

    analysis_type = request.query_params.get("type")
    if analysis_type:
        qs = qs.filter(analysis_type=analysis_type)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        AnalysisResultSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_analysis_read(request, analysis_id):
    """Mark an analysis result as read."""
    result = get_object_or_404(AnalysisResult, id=analysis_id, user=request.user)
    result.is_read = True
    result.save(update_fields=["is_read"])
    return Response({"status": "read"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def archive_analysis(request, analysis_id):
    """Archive an analysis result."""
    result = get_object_or_404(AnalysisResult, id=analysis_id, user=request.user)
    result.is_archived = True
    result.save(update_fields=["is_archived"])
    return Response({"status": "archived"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def trigger_analysis(request):
    """Manually trigger a background analysis for the current user."""
    run_periodic_analysis.delay(user_id=request.user.id)
    return Response({"status": "analysis triggered"}, status=202)


# ═══════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def recommendation_list(request):
    """Get the authenticated user's recommendations."""
    qs = Recommendation.objects.filter(user=request.user, is_dismissed=False).order_by(
        "-created_at"
    )

    rec_type = request.query_params.get("type")
    if rec_type:
        qs = qs.filter(recommendation_type=rec_type)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        RecommendationSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def apply_recommendation(request, recommendation_id):
    """Mark a recommendation as applied."""
    rec = get_object_or_404(Recommendation, id=recommendation_id, user=request.user)
    rec.is_applied = True
    rec.save(update_fields=["is_applied"])
    return Response({"status": "applied", "action_data": rec.action_data})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, recommendation_id):
    """Dismiss a recommendation."""
    rec = get_object_or_404(Recommendation, id=recommendation_id, user=request.user)
    rec.is_dismissed = True
    rec.save(update_fields=["is_dismissed"])
    return Response({"status": "dismissed"})


# ═══════════════════════════════════════════════════════════════
# Chat Sessions
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def chat_session_list(request):
    """List chat sessions or create a new one."""
    if request.method == "GET":
        qs = ChatSession.objects.filter(user=request.user).order_by("-updated_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ChatSessionSerializer(page, many=True).data
        )

    elif request.method == "POST":
        session = ChatSession.objects.create(
            user=request.user, title=request.data.get("title", "Research Coach Chat")
        )
        return Response(ChatSessionSerializer(session).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def chat_session_detail(request, session_id):
    """Get messages for a chat session."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by("created_at")
    data = ChatSessionSerializer(session).data
    data["messages"] = ChatMessageSerializer(messages, many=True).data
    return Response(data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def chat_session_delete(request, session_id):
    """Delete a chat session."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.delete()
    return Response(status=204)
