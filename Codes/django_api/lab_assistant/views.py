import logging

from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .ai_assistant import AIAssistant
from .label_studio_client import LabelStudioClient
from .models import (
    AIModelFeedback,
    AnnotationGuideline,
    AnnotationProject,
    AnnotationReview,
    DataTask,
)
from .serializers import (
    AIModelFeedbackSerializer,
    AnnotationGuidelineSerializer,
    AnnotationProjectDetailSerializer,
    AnnotationProjectListSerializer,
    AnnotationReviewSerializer,
    DataTaskSerializer,
)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# PROJECTS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def project_list(request):
    if request.method == "GET":
        qs = (
            AnnotationProject.objects.filter(
                Q(owner=request.user) | Q(annotators=request.user)
            )
            .distinct()
            .annotate(task_count=Count("tasks"))
            .order_by("-created_at")
        )

        task_type = request.query_params.get("task_type")
        if task_type:
            qs = qs.filter(task_type=task_type)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            AnnotationProjectListSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = AnnotationProjectDetailSerializer(data=request.data)
        if serializer.is_valid():
            project = serializer.save(owner=request.user)
            return Response(AnnotationProjectDetailSerializer(project).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    project = get_object_or_404(AnnotationProject, id=project_id)
    if not (
        project.owner == request.user
        or project.annotators.filter(id=request.user.id).exists()
    ):
        return Response({"error": "Access denied"}, status=403)

    if request.method == "GET":
        return Response(AnnotationProjectDetailSerializer(project).data)
    elif request.method == "PATCH":
        if project.owner != request.user:
            return Response({"error": "Only the owner can edit"}, status=403)
        serializer = AnnotationProjectDetailSerializer(
            project, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if project.owner != request.user:
            return Response({"error": "Only the owner can delete"}, status=403)
        project.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# DATA TASKS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def task_list(request, project_id):
    project = get_object_or_404(AnnotationProject, id=project_id)

    if request.method == "GET":
        qs = project.tasks.all().order_by("task_index")

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            DataTaskSerializer(page, many=True).data
        )

    elif request.method == "POST":
        if project.owner != request.user:
            return Response({"error": "Only the owner can add tasks"}, status=403)

        tasks_data = request.data.get("tasks", [])
        created = []
        for i, task_data in enumerate(tasks_data):
            task = DataTask.objects.create(
                project=project, data=task_data, task_index=project.tasks.count() + i
            )
            created.append(DataTaskSerializer(task).data)

        project.total_tasks = project.tasks.count()
        project.save(update_fields=["total_tasks"])

        return Response(created, status=201)


@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def task_detail(request, project_id, task_id):
    project = get_object_or_404(AnnotationProject, id=project_id)
    task = get_object_or_404(DataTask, id=task_id, project=project)

    if request.method == "GET":
        return Response(DataTaskSerializer(task).data)

    elif request.method == "PUT":
        # Submit annotation
        annotations = request.data.get("annotations", [])
        task.annotations = annotations
        task.annotated_by = request.user
        task.status = "completed"
        task.annotated_at = timezone.now()
        task.save()

        # Update project stats
        project.completed_tasks = project.tasks.filter(status="completed").count()
        project.save(update_fields=["completed_tasks"])

        return Response(DataTaskSerializer(task).data)


# ═══════════════════════════════════════════════════════════════
# AI-ASSISTED PRE-LABELING
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_ai_pre_labeling(request, project_id):
    """Run AI-assisted pre-labeling on all pending tasks in a project."""
    project = get_object_or_404(AnnotationProject, id=project_id)
    if project.owner != request.user:
        return Response({"error": "Only the owner can run AI labeling"}, status=403)

    model_id = request.data.get("model_id", project.ai_model_id or "bert-base-uncased")
    assistant = AIAssistant(model_id=model_id)

    tasks = project.tasks.filter(status="pending")
    count = 0
    for task in tasks:
        if assistant.pre_label_task(task):
            count += 1

    return Response({"status": "completed", "tasks_pre_labeled": count})


# ═══════════════════════════════════════════════════════════════
# REVIEWS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_review(request, project_id, task_id):
    project = get_object_or_404(AnnotationProject, id=project_id)
    task = get_object_or_404(DataTask, id=task_id, project=project)

    review = AnnotationReview.objects.create(
        task=task,
        reviewer=request.user,
        status=request.data.get("status", "approved"),
        comment=request.data.get("comment", ""),
    )
    return Response(AnnotationReviewSerializer(review).data, status=201)


# ═══════════════════════════════════════════════════════════════
# GUIDELINES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def guidelines(request, project_id):
    project = get_object_or_404(AnnotationProject, id=project_id)

    if request.method == "GET":
        guidelines_obj, _ = AnnotationGuideline.objects.get_or_create(project=project)
        return Response(AnnotationGuidelineSerializer(guidelines_obj).data)

    elif request.method == "PUT":
        guidelines_obj, _ = AnnotationGuideline.objects.get_or_create(project=project)
        serializer = AnnotationGuidelineSerializer(
            guidelines_obj, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# LABEL STUDIO INTEGRATION
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def sync_to_label_studio(request, project_id):
    """Sync a project to an external Label Studio instance."""
    project = get_object_or_404(AnnotationProject, id=project_id)
    if project.owner != request.user:
        return Response({"error": "Only the owner can sync"}, status=403)

    client = LabelStudioClient()
    ls_project = client.create_project(
        name=project.name,
        description=project.description,
        label_config=project.labeling_config or "<View></View>",
    )

    if ls_project:
        project.label_studio_project_id = ls_project["id"]
        project.save(update_fields=["label_studio_project_id"])

        # Import tasks
        tasks_payload = [
            {"data": t.data, "id": t.task_index} for t in project.tasks.all()
        ]
        client.import_tasks(ls_project["id"], tasks_payload)
        return Response({"status": "synced", "label_studio_id": ls_project["id"]})

    return Response({"error": "Failed to sync with Label Studio"}, status=500)


# ═══════════════════════════════════════════════════════════════
# AI FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_ai_feedback(request, project_id, task_id):
    project = get_object_or_404(AnnotationProject, id=project_id)
    task = get_object_or_404(DataTask, id=task_id, project=project)

    feedback = AIModelFeedback.objects.create(
        task=task,
        ai_prediction=task.ai_predictions[0] if task.ai_predictions else {},
        human_correction=request.data.get("human_correction", {}),
        is_correct=request.data.get("is_correct", False),
    )
    return Response(AIModelFeedbackSerializer(feedback).data, status=201)
