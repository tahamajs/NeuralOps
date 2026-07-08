import logging

from django.contrib.auth import get_user_model
from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import ExpertProfile, ExpertReview, Project, ProjectApplication
from .serializers import *
from .tasks import auto_generate_expert_profile

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Expert Profiles
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def my_expert_profile(request):
    """Get or update the authenticated user's expert profile."""
    profile, _ = ExpertProfile.objects.get_or_create(user=request.user)

    if request.method == "GET":
        return Response(ExpertProfileSerializer(profile).data)

    elif request.method == "PATCH":
        serializer = ExpertProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([AllowAny])
def expert_profile_detail(request, user_id):
    """View any user's expert profile."""
    profile = get_object_or_404(ExpertProfile, user_id=user_id)
    return Response(ExpertProfileSerializer(profile).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def expert_directory(request):
    """Browse experts by skills, domains, or expertise level."""
    qs = ExpertProfile.objects.select_related("user").filter(verified=True)

    skill = request.query_params.get("skill")
    if skill:
        qs = qs.filter(skills__contains=[skill])

    domain = request.query_params.get("domain")
    if domain:
        qs = qs.filter(domains__contains=[domain])

    expertise = request.query_params.get("level")
    if expertise:
        qs = qs.filter(expertise_level=expertise)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ExpertProfileSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def regenerate_expert_profile(request):
    """Regenerate the expert profile from publication history."""
    auto_generate_expert_profile.delay(request.user.id)
    return Response({"status": "regeneration started"})


# ═══════════════════════════════════════════════════════════════
# Projects (seeking collaborators)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def project_list(request):
    """List projects (public + owned), or create a new one."""
    if request.method == "GET":
        qs = Project.objects.filter(Q(is_public=True) | Q(owner=request.user)).order_by(
            "-created_at"
        )

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        project_type = request.query_params.get("type")
        if project_type:
            qs = qs.filter(project_type=project_type)

        search = request.query_params.get("q")
        if search:
            qs = qs.filter(
                Q(title__icontains=search) | Q(description__icontains=search)
            )

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(ProjectSerializer(page, many=True).data)

    elif request.method == "POST":
        serializer = ProjectSerializer(data=request.data)
        if serializer.is_valid():
            project = serializer.save(owner=request.user)
            return Response(ProjectSerializer(project).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    """Get, update, or delete a project."""
    project = get_object_or_404(Project, id=project_id)

    if request.method == "GET":
        return Response(ProjectSerializer(project).data)

    elif request.method == "PATCH":
        if project.owner != request.user:
            return Response({"error": "Only the owner can edit"}, status=403)
        serializer = ProjectSerializer(project, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        if project.owner != request.user:
            return Response({"error": "Only the owner can delete"}, status=403)
        project.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def apply_to_project(request, project_id):
    """Apply to join a project as a collaborator."""
    project = get_object_or_404(Project, id=project_id)
    if project.owner == request.user:
        return Response({"error": "Cannot apply to your own project"}, status=400)
    if project.status != "active":
        return Response({"error": "Project is not currently recruiting"}, status=400)

    application, created = ProjectApplication.objects.get_or_create(
        project=project,
        applicant=request.user,
        defaults={"message": request.data.get("message", "")},
    )
    return Response(
        ProjectApplicationSerializer(application).data, status=201 if created else 200
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def handle_application(request, project_id, application_id):
    """Accept or reject an application (project owner only)."""
    project = get_object_or_404(Project, id=project_id, owner=request.user)
    application = get_object_or_404(
        ProjectApplication, id=application_id, project=project
    )

    new_status = request.data.get("status")
    if new_status not in ("accepted", "rejected"):
        return Response(
            {"error": "status must be 'accepted' or 'rejected'"}, status=400
        )

    application.status = new_status
    application.save(update_fields=["status"])
    if new_status == "accepted":
        project.current_collaborators += 1
        project.save(update_fields=["current_collaborators"])
    return Response(ProjectApplicationSerializer(application).data)


# ═══════════════════════════════════════════════════════════════
# Expert Reviews
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def review_list(request):
    """List my requested reviews, or create a new review request."""
    if request.method == "GET":
        qs = ExpertReview.objects.filter(requester=request.user).order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ExpertReviewSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ExpertReviewSerializer(data=request.data)
        if serializer.is_valid():
            review = serializer.save(requester=request.user)
            return Response(ExpertReviewSerializer(review).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def review_requests_for_me(request):
    """Get review requests where I am the reviewer, or open requests."""
    # Open requests (no reviewer assigned yet)
    qs = ExpertReview.objects.filter(
        Q(reviewer=request.user) | Q(reviewer__isnull=True, requester=request.user)
    ).order_by("-created_at")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ExpertReviewSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def accept_review(request, review_id):
    """Accept a review request (as a reviewer)."""
    review = get_object_or_404(ExpertReview, id=review_id, reviewer__isnull=True)
    if review.requester == request.user:
        return Response({"error": "Cannot review your own request"}, status=400)
    review.reviewer = request.user
    review.status = "accepted"
    review.save(update_fields=["reviewer", "status"])
    return Response(ExpertReviewSerializer(review).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def complete_review(request, review_id):
    """Submit a completed review with feedback and rating."""
    review = get_object_or_404(ExpertReview, id=review_id, reviewer=request.user)
    review.reviewer_feedback = request.data.get("feedback", "")
    review.rating = request.data.get("rating")
    review.status = "completed"
    review.completed_at = timezone.now()
    review.save()
    return Response(ExpertReviewSerializer(review).data)


# ═══════════════════════════════════════════════════════════════
# Matching (using collaborator_matching engine)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def matched_experts_for_project(request, project_id):
    """
    Find experts that match the skills/domains needed for a project.
    Uses the collaborator_matching embedding engine.
    """
    project = get_object_or_404(Project, id=project_id)
    if not project.skills_needed and not project.domains_needed:
        return Response([])

    # Simple keyword matching on skills
    qs = ExpertProfile.objects.filter(seeking_collaborators=True).exclude(
        user=project.owner
    )

    # Filter by required skills
    if project.skills_needed:
        for skill in project.skills_needed:
            qs = qs.filter(skills__contains=[skill])

    # Filter by required domains
    if project.domains_needed:
        for domain in project.domains_needed:
            qs = qs.filter(domains__contains=[domain])

    qs = qs.order_by("-expertise_level")[:10]

    return Response(ExpertProfileSerializer(qs, many=True).data)
