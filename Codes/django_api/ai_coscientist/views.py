import logging
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Hypothesis, ResearchSession, TournamentMatch
from .serializers import (
    HypothesisSerializer,
    ResearchSessionSerializer,
    TournamentMatchSerializer,
)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def session_list(request):
    """List or create research sessions."""
    if request.method == "POST":
        return create_session(request)
    qs = ResearchSession.objects.filter(user=request.user).order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ResearchSessionSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_session(request):
    """Create a new research session and start the workflow."""
    research_goal = (
        request.data.get("research_goal")
        or request.data.get("goal")
        or request.data.get("topic")
    )
    if not research_goal:
        return Response({"error": "research_goal required"}, status=400)

    config = request.data.get("config") or {}

    model_name = config.get("model_name") or request.data.get("model_name", "gpt-4o-mini")
    max_iterations = config.get("max_iterations") or request.data.get("max_iterations", 3)
    hypotheses_per_generation = config.get("hypotheses_per_generation") or request.data.get("hypotheses_per_generation", 10)
    tournament_size = config.get("tournament_size") or request.data.get("tournament_size", 8)
    evolution_top_k = config.get("evolution_top_k") or request.data.get("evolution_top_k", 3)

    session = ResearchSession.objects.create(
        user=request.user,
        research_goal=research_goal,
        model_name=model_name,
        max_iterations=max_iterations,
        hypotheses_per_generation=hypotheses_per_generation,
        tournament_size=tournament_size,
        evolution_top_k=evolution_top_k,
    )

    # Run asynchronously if celery is configured
    try:
        from .tasks import run_research_workflow
        run_research_workflow.delay(str(session.id))
    except Exception as e:
        logger.warning(f"Could not trigger background research task: {e}")

    data = ResearchSessionSerializer(session).data
    data["session_id"] = str(session.id)
    return Response(data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def session_detail(request, session_id):
    """Get details of a research session."""
    session = get_object_or_404(ResearchSession, id=session_id, user=request.user)
    data = ResearchSessionSerializer(session).data
    data["session_id"] = str(session.id)
    return Response(data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def session_hypotheses(request, session_id):
    """Get all hypotheses belonging to a research session."""
    session = get_object_or_404(ResearchSession, id=session_id, user=request.user)
    hypotheses = session.hypotheses.all().order_by("-elo_score")
    return Response(HypothesisSerializer(hypotheses, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def session_stop(request, session_id):
    """Stop a running research session."""
    session = get_object_or_404(ResearchSession, id=session_id, user=request.user)
    session.status = "completed"
    session.save(update_fields=["status"])
    return Response({"status": "stopped", "session_id": str(session.id)})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def session_evolve(request, session_id):
    """Trigger an evolution cycle on the top-ranking hypotheses."""
    session = get_object_or_404(ResearchSession, id=session_id, user=request.user)
    try:
        from .tasks import run_research_workflow
        run_research_workflow.delay(str(session.id))
    except Exception as e:
        logger.warning(f"Could not trigger background evolution: {e}")
    return Response({"status": "evolution cycle triggered", "session_id": str(session.id)})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hypothesis_detail(request, hypothesis_id):
    """Get details of a single hypothesis."""
    hypothesis = get_object_or_404(
        Hypothesis, id=hypothesis_id, session__user=request.user
    )
    return Response(HypothesisSerializer(hypothesis).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def tournament_history(request, session_id):
    """Get tournament matches for a session."""
    matches = TournamentMatch.objects.filter(session_id=session_id).order_by(
        "-created_at"
    )
    paginator = StandardPagination()
    page = paginator.paginate_queryset(matches, request)
    return paginator.get_paginated_response(
        TournamentMatchSerializer(page, many=True).data
    )
