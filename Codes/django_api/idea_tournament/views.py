import logging
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import IdeaTournament, Hypothesis, HypothesisReview, LiteratureCitation, TournamentLog
from .serializers import *
from .tasks import run_tournament

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Tournaments
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def tournament_list(request):
    """List or create idea tournaments."""
    if request.method == 'GET':
        qs = IdeaTournament.objects.filter(owner=request.user).order_by('-created_at')
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(IdeaTournamentSerializer(page, many=True).data)

    elif request.method == 'POST':
        serializer = IdeaTournamentDetailSerializer(data=request.data)
        if serializer.is_valid():
            tournament = serializer.save(owner=request.user)
            TournamentLog.objects.create(
                tournament=tournament,
                round_number=0,
                agent='system',
                action='Tournament created',
                description=f"Research domain: {tournament.research_domain}",
            )
            return Response(IdeaTournamentDetailSerializer(tournament).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def tournament_detail(request, tournament_id):
    """Get or delete a tournament."""
    tournament = get_object_or_404(IdeaTournament, id=tournament_id, owner=request.user)

    if request.method == 'GET':
        return Response(IdeaTournamentDetailSerializer(tournament).data)

    elif request.method == 'DELETE':
        tournament.delete()
        return Response(status=204)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_tournament(request, tournament_id):
    """Start the AI‑powered idea tournament."""
    tournament = get_object_or_404(IdeaTournament, id=tournament_id, owner=request.user)

    if tournament.status != 'pending':
        return Response({"error": "Tournament is already running or completed"}, status=400)

    tournament.status = 'running'
    tournament.save(update_fields=['status'])

    TournamentLog.objects.create(
        tournament=tournament,
        round_number=0,
        agent='system',
        action='Tournament started',
        description='Generation agent is creating initial hypotheses.',
    )

    # Trigger async task
    run_tournament.delay(str(tournament.id))

    return Response({'status': 'started', 'tournament_id': str(tournament.id)})


# ═══════════════════════════════════════════════════════════════
# Hypotheses
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def hypothesis_list(request, tournament_id):
    """List all hypotheses in a tournament."""
    tournament = get_object_or_404(IdeaTournament, id=tournament_id, owner=request.user)
    qs = tournament.hypotheses.order_by('-score')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(HypothesisSerializer(page, many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def hypothesis_detail(request, hypothesis_id):
    """Get a single hypothesis with all reviews and citations."""
    hypothesis = get_object_or_404(Hypothesis, id=hypothesis_id)
    return Response(HypothesisSerializer(hypothesis).data)


# ═══════════════════════════════════════════════════════════════
# Tournament Results
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def tournament_winners(request, tournament_id):
    """Get the winning hypotheses of a tournament."""
    tournament = get_object_or_404(IdeaTournament, id=tournament_id, owner=request.user)
    winners = tournament.hypotheses.filter(status='winner').order_by('rank')
    return Response(HypothesisSerializer(winners, many=True).data)