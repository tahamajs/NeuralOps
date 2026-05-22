import logging
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import *
from .serializers import *
from .engine import calculate_opportunity_index, generate_recommendations, generate_creative_variants

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Templates
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def template_list(request):
    if request.method == 'GET':
        qs = ExperimentTemplate.objects.filter(
            models.Q(is_public=True) | models.Q(created_by=request.user)
        ).order_by('-success_rate')

        category = request.query_params.get('category')
        if category:
            qs = qs.filter(category=category)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(ExperimentTemplateSerializer(page, many=True).data)

    elif request.method == 'POST':
        serializer = ExperimentTemplateSerializer(data=request.data)
        if serializer.is_valid():
            template = serializer.save(created_by=request.user)
            return Response(ExperimentTemplateSerializer(template).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def use_template(request, template_id):
    """Mark a template as used (increment usage count)."""
    template = get_object_or_404(ExperimentTemplate, id=template_id)
    template.usage_count += 1
    template.save(update_fields=['usage_count'])
    return Response({'status': 'used', 'usage_count': template.usage_count})


# ═══════════════════════════════════════════════════════════════
# Opportunity Index
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def opportunity_index(request):
    """Get the authenticated user's opportunity index."""
    gaps = OpportunityIndex.objects.filter(user=request.user).order_by('-opportunity_score')
    if not gaps.exists():
        calculate_opportunity_index(request.user)
        gaps = OpportunityIndex.objects.filter(user=request.user).order_by('-opportunity_score')
    serializer = OpportunityIndexSerializer(gaps, many=True)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def recalculate_opportunity(request):
    """Recalculate the opportunity index for the authenticated user."""
    calculate_opportunity_index(request.user)
    gaps = OpportunityIndex.objects.filter(user=request.user).order_by('-opportunity_score')
    return Response(OpportunityIndexSerializer(gaps, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommendation_list(request):
    """Get experiment recommendations for the current user."""
    qs = ExperimentRecommendation.objects.filter(
        user=request.user, status='suggested'
    ).order_by('-predicted_lift')

    if not qs.exists():
        generate_recommendations(request.user)
        qs = ExperimentRecommendation.objects.filter(
            user=request.user, status='suggested'
        ).order_by('-predicted_lift')

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ExperimentRecommendationSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def accept_recommendation(request, recommendation_id):
    """Accept a recommendation and auto-launch the experiment."""
    rec = get_object_or_404(ExperimentRecommendation, id=recommendation_id, user=request.user)
    rec.status = 'accepted'
    rec.save(update_fields=['status', 'updated_at'])

    # Auto‑launch the experiment
    from experiment.models import TrainingRun
    run = TrainingRun.objects.create(
        user=request.user,
        name=f"Recommended: {rec.title}",
        config=rec.config,
        status='queued',
    )

    rec.metadata = rec.metadata or {}
    rec.metadata['launched_run_id'] = str(run.id)
    rec.save(update_fields=['metadata'])

    return Response({
        'status': 'accepted',
        'run_id': str(run.id),
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, recommendation_id):
    """Dismiss a recommendation."""
    rec = get_object_or_404(ExperimentRecommendation, id=recommendation_id, user=request.user)
    rec.status = 'dismissed'
    rec.save(update_fields=['status', 'updated_at'])
    return Response({'status': 'dismissed'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def regenerate_recommendations(request):
    """Regenerate all recommendations for the authenticated user."""
    count = generate_recommendations(request.user)
    return Response({'recommendations_created': count})


# ═══════════════════════════════════════════════════════════════
# Creative Variants
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def creative_variants(request, run_id):
    """Get creative variants for an existing experiment."""
    variants = CreativeVariant.objects.filter(
        base_experiment_id=run_id, user=request.user
    ).order_by('-predicted_lift')

    if not variants.exists():
        generate_creative_variants(request.user, run_id)
        variants = CreativeVariant.objects.filter(
            base_experiment_id=run_id, user=request.user
        ).order_by('-predicted_lift')

    return Response(CreativeVariantSerializer(variants, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def accept_variant(request, variant_id):
    """Accept a creative variant and launch it."""
    variant = get_object_or_404(CreativeVariant, id=variant_id, user=request.user)
    variant.status = 'accepted'
    variant.save(update_fields=['status'])

    # Merge changes into base config
    base_config = variant.base_experiment.config or {}
    new_config = {**base_config}
    for key, change in (variant.changes or {}).items():
        new_config[key] = change.get('to', change)

    from experiment.models import TrainingRun
    run = TrainingRun.objects.create(
        user=request.user,
        name=f"Variant: {variant.title}",
        config=new_config,
        status='queued',
    )

    return Response({
        'status': 'accepted',
        'run_id': str(run.id),
    })