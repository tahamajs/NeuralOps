from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination

from .models import ResearchSession, Hypothesis, TournamentMatch
from .serializers import ResearchSessionSerializer, HypothesisSerializer, TournamentMatchSerializer
from .tasks import run_research_workflow


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def session_list(request):
    """List the authenticated user's research sessions."""
    qs = ResearchSession.objects.filter(user=request.user).order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ResearchSessionSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_session(request):
    """Create a new research session and start the workflow."""
    research_goal = request.data.get('research_goal')
    if not research_goal:
        return Response({"error": "research_goal required"}, status=400)

    session = ResearchSession.objects.create(
        user=request.user,
        research_goal=research_goal,
        model_name=request.data.get('model_name', 'gpt-4o-mini'),
        max_iterations=request.data.get('max_iterations', 3),
        hypotheses_per_generation=request.data.get('hypotheses_per_generation', 10),
        tournament_size=request.data.get('tournament_size', 8),
        evolution_top_k=request.data.get('evolution_top_k', 3),
    )

    # Run asynchronously
    run_research_workflow.delay(str(session.id))

    return Response(ResearchSessionSerializer(session).data, status=201)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def session_detail(request, session_id):
    """Get details of a research session."""
    session = get_object_or_404(ResearchSession, id=session_id, user=request.user)
    return Response(ResearchSessionSerializer(session).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def hypothesis_detail(request, hypothesis_id):
    """Get details of a single hypothesis."""
    hypothesis = get_object_or_404(
        Hypothesis, id=hypothesis_id, session__user=request.user
    )
    return Response(HypothesisSerializer(hypothesis).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def tournament_history(request, session_id):
    """Get tournament matches for a session."""
    matches = TournamentMatch.objects.filter(session_id=session_id).order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(matches, request)
    return paginator.get_paginated_response(TournamentMatchSerializer(page, many=True).data)