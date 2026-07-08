import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import ImpactPrediction, ResearchProject
from .scoring_engine import extract_features, predict
from .serializers import ImpactPredictionSerializer, ResearchProjectSerializer

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Project Management (CRUD)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def project_list(request):
    """List the user's projects, or create a new one."""
    if request.method == "GET":
        qs = ResearchProject.objects.filter(user=request.user).order_by("-created_at")
        serializer = ResearchProjectSerializer(qs, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        serializer = ResearchProjectSerializer(data=request.data)
        if serializer.is_valid():
            project = serializer.save(user=request.user)
            # Auto-score on creation
            features = extract_features(project)
            result = predict(features)
            ImpactPrediction.objects.create(project=project, **result)
            project.status = "scored"
            project.save(update_fields=["status"])
            response = ResearchProjectSerializer(project).data
            response["prediction"] = result
            return Response(response, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    """Get, update, or delete a single project."""
    project = get_object_or_404(ResearchProject, id=project_id, user=request.user)

    if request.method == "GET":
        data = ResearchProjectSerializer(project).data
        try:
            data["prediction"] = ImpactPredictionSerializer(
                project.impact_prediction
            ).data
        except ImpactPrediction.DoesNotExist:
            data["prediction"] = None
        return Response(data)

    elif request.method == "PATCH":
        serializer = ResearchProjectSerializer(project, data=request.data, partial=True)
        if serializer.is_valid():
            project = serializer.save()
            # Re-score if the project was already scored
            if project.status == "scored" or project.status == "submitted":
                features = extract_features(project)
                result = predict(features)
                ImpactPrediction.objects.update_or_create(
                    project=project, defaults=result
                )
            return Response(ResearchProjectSerializer(project).data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        project.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Impact Prediction
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_impact(request):
    """
    Predict impact score from provided project data (without saving).
    Body: all the fields of ResearchProject.
    """
    features = extract_features_from_dict(request.data)
    result = predict(features)
    result["features_used"] = features
    return Response(result)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def rescore_project(request, project_id):
    """Re-score a saved project."""
    project = get_object_or_404(ResearchProject, id=project_id, user=request.user)
    features = extract_features(project)
    result = predict(features)
    prediction, _ = ImpactPrediction.objects.update_or_create(
        project=project, defaults=result
    )
    project.status = "scored"
    project.save(update_fields=["status"])
    return Response(ImpactPredictionSerializer(prediction).data)


# ═══════════════════════════════════════════════════════════════
# Leaderboard / Rankings
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def project_leaderboard(request):
    """Get a leaderboard of projects ranked by predicted impact score."""
    qs = ImpactPrediction.objects.select_related("project").order_by("-impact_score")

    # Optionally filter by domain
    domain = request.query_params.get("domain")
    if domain:
        qs = qs.filter(project__domain__icontains=domain)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)

    results = []
    for pred in page:
        results.append(
            {
                "rank": None,  # computed by position in full queryset is expensive; we skip
                "project_id": str(pred.project.id),
                "title": pred.project.title,
                "username": pred.project.user.username,
                "impact_score": pred.impact_score,
                "percentile": pred.percentile,
                "publication_tier": pred.publication_tier,
                "funding_probability": pred.funding_probability,
            }
        )
    return paginator.get_paginated_response(results)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def extract_features_from_dict(data):
    """Extract features from a request data dict (no model instance)."""

    # Build a minimal object with attributes
    class ProjectMock:
        pass

    p = ProjectMock()
    for field in [
        "institution_age_years",
        "institution_location",
        "investigator_h_index",
        "investigator_publications",
        "investigator_previous_grants",
        "social_followers",
        "preprint_views",
        "online_mentions",
        "previous_funding_total",
        "previous_grants_count",
    ]:
        setattr(p, field, data.get(field))
    return extract_features(p)
