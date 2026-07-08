# literature_review/views.py
import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *
from .tasks import generate_notebook_content, run_deep_review, run_topic_finder

logger = logging.getLogger(__name__)
StandardPagination = PageNumberPagination


# ═══════════════ Projects CRUD ═══════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def project_list(request):
    if request.method == "GET":
        qs = LiteratureProject.objects.filter(user=request.user).order_by("-updated_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            LiteratureProjectSerializer(page, many=True).data
        )
    elif request.method == "POST":
        serializer = LiteratureProjectSerializer(data=request.data)
        if serializer.is_valid():
            project = serializer.save(user=request.user)
            return Response(LiteratureProjectSerializer(project).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    project = get_object_or_404(LiteratureProject, id=project_id, user=request.user)
    if request.method == "GET":
        return Response(LiteratureProjectSerializer(project).data)
    elif request.method == "PATCH":
        serializer = LiteratureProjectSerializer(
            project, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        project.delete()
        return Response(status=204)


# ═══════════════ Workflow Steps ═══════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_topic_finder(request, project_id):
    project = get_object_or_404(LiteratureProject, id=project_id, user=request.user)
    if project.current_step != 1:
        return Response({"error": "Project is not in Topic Finder step"}, status=400)
    run_topic_finder.delay(str(project.id), request.data.get("query", ""))
    return Response({"status": "Topic finder started"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_notebook(request, project_id):
    project = get_object_or_404(LiteratureProject, id=project_id, user=request.user)
    if project.current_step != 2:
        return Response(
            {"error": "Project is not in Research Notebook step"}, status=400
        )
    generate_notebook_content.delay(str(project.id))
    return Response({"status": "Notebook generation started"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_deep_review(request, project_id):
    project = get_object_or_404(LiteratureProject, id=project_id, user=request.user)
    if project.current_step != 3:
        return Response({"error": "Project is not in Deep Review step"}, status=400)
    run_deep_review.delay(str(project.id), request.data.get("paper_ids", []))
    return Response({"status": "Deep review started"})


# ═══════════════ Notebook Sections ═══════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def notebook_section_list(request, project_id):
    project = get_object_or_404(LiteratureProject, id=project_id, user=request.user)
    if request.method == "GET":
        sections = project.notebook_sections.all()
        return Response(ResearchNotebookSectionSerializer(sections, many=True).data)
    elif request.method == "POST":
        serializer = ResearchNotebookSectionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(project=project)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def notebook_section_detail(request, section_id):
    section = get_object_or_404(ResearchNotebookSection, id=section_id)
    if request.method == "GET":
        return Response(ResearchNotebookSectionSerializer(section).data)
    elif request.method == "PUT":
        serializer = ResearchNotebookSectionSerializer(section, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        section.delete()
        return Response(status=204)


# ═══════════════ Deep Reviews ═══════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def deep_review_list(request, project_id):
    project = get_object_or_404(LiteratureProject, id=project_id, user=request.user)
    if request.method == "GET":
        reviews = project.deep_reviews.all()
        return Response(DeepReviewSerializer(reviews, many=True).data)
    elif request.method == "POST":
        serializer = DeepReviewSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(project=project)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def deep_review_detail(request, review_id):
    review = get_object_or_404(DeepReview, id=review_id)
    return Response(DeepReviewSerializer(review).data)
