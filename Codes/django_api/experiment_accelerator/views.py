import logging
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination

from .models import ExperimentInsight, ExperimentRecommendation, ExperimentSummary
from .serializers import *
from .insight_engine import analyze_user_experiments
from .tasks import generate_insights_for_user

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 50


# ═══════════════════════════════════════════════════════════════
# Insights
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_insights(request):
    """Get AI insights for the authenticated user."""
    insight_type = request.query_params.get('type')
    qs = ExperimentInsight.objects.filter(user=request.user).order_by('-created_at')
    if insight_type:
        qs = qs.filter(insight_type=insight_type)

    unread = request.query_params.get('unread')
    if unread == 'true':
        qs = qs.filter(is_read=False)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ExperimentInsightSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_insights(request):
    """Manually trigger insight generation for the current user."""
    generate_insights_for_user.delay(request.user.id)
    return Response({'status': 'generation started'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_insight_read(request, insight_id):
    """Mark an insight as read."""
    insight = get_object_or_404(ExperimentInsight, id=insight_id, user=request.user)
    insight.is_read = True
    insight.save(update_fields=['is_read'])
    return Response({'status': 'read'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_all_insights_read(request):
    """Mark all insights as read."""
    ExperimentInsight.objects.filter(user=request.user, is_read=False).update(is_read=True)
    return Response({'status': 'all read'})


# ═══════════════════════════════════════════════════════════════
# Recommendations (Next experiments to try)
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommendations(request):
    """Get recommended next experiments."""
    qs = ExperimentRecommendation.objects.filter(
        user=request.user, status='suggested'
    ).order_by('-predicted_improvement')

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ExperimentRecommendationSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def launch_recommendation(request, recommendation_id):
    """Launch a recommended experiment."""
    rec = get_object_or_404(ExperimentRecommendation, id=recommendation_id, user=request.user)

    # Create a new training run from the recommendation's configuration
    from experiment.models import Experiment, TrainingRun

    experiment = Experiment.objects.create(name=rec.title)
    run = TrainingRun.objects.create(
        experiment=experiment,
        job_type='training',
        status='queued',
        config=rec.configuration,
        name=rec.title,
    )

    rec.status = 'launched'
    rec.save(update_fields=['status'])

    return Response({
        'status': 'launched',
        'run_id': str(run.id),
        'recommendation_id': str(rec.id),
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, recommendation_id):
    """Dismiss a recommendation."""
    rec = get_object_or_404(ExperimentRecommendation, id=recommendation_id, user=request.user)
    rec.status = 'dismissed'
    rec.save(update_fields=['status'])
    return Response({'status': 'dismissed'})


# ═══════════════════════════════════════════════════════════════
# Summaries
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def summaries(request):
    """Get periodic experiment summaries."""
    qs = ExperimentSummary.objects.filter(user=request.user).order_by('-period_end')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ExperimentSummarySerializer(page, many=True).data)