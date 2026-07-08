"""
Developer Playground REST API — Prompts, Model Configs, Sessions, Comparisons, Tools.
"""

import logging
import time

from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 50


# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES (Langfuse-style)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def prompt_list(request):
    if request.method == "GET":
        qs = PromptTemplate.objects.filter(
            Q(is_public=True) | Q(created_by=request.user)
        ).order_by("-updated_at")

        search = request.query_params.get("q")
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))

        tag = request.query_params.get("tag")
        if tag:
            qs = qs.filter(tags__contains=[tag])

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            PromptTemplateSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = PromptTemplateSerializer(data=request.data)
        if serializer.is_valid():
            template = serializer.save(created_by=request.user)
            return Response(PromptTemplateSerializer(template).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def prompt_detail(request, template_id):
    template = get_object_or_404(PromptTemplate, id=template_id)
    if request.method == "GET":
        return Response(PromptTemplateSerializer(template).data)

    elif request.method == "PATCH":
        if template.created_by != request.user:
            return Response({"error": "Only the creator can edit"}, status=403)
        serializer = PromptTemplateSerializer(template, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        if template.created_by != request.user:
            return Response({"error": "Only the creator can delete"}, status=403)
        template.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def prompt_new_version(request, template_id):
    """Fork a prompt template to create a new version."""
    template = get_object_or_404(PromptTemplate, id=template_id)
    new_content = request.data.get("content", template.content)
    new_vars = request.data.get("variables_schema", template.variables_schema)
    new_version = template.create_new_version(new_content, new_vars)
    return Response(PromptTemplateSerializer(new_version).data, status=201)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def prompt_test_render(request, template_id):
    """Test render a prompt template with given variables."""
    template = get_object_or_404(PromptTemplate, id=template_id)
    variables = request.data.get("variables", {})
    rendered = template.render(variables)
    return Response({"rendered": rendered})


# ═══════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def model_config_list(request):
    if request.method == "GET":
        qs = ModelConfig.objects.filter(created_by=request.user).order_by(
            "-is_default", "-updated_at"
        )
        return Response(ModelConfigSerializer(qs, many=True).data)

    elif request.method == "POST":
        serializer = ModelConfigSerializer(data=request.data)
        if serializer.is_valid():
            config = serializer.save(created_by=request.user)
            return Response(ModelConfigSerializer(config).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def model_config_detail(request, config_id):
    config = get_object_or_404(ModelConfig, id=config_id, created_by=request.user)
    if request.method == "PATCH":
        serializer = ModelConfigSerializer(config, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        config.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# PLAYGROUND SESSIONS (Chat)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def session_list(request):
    if request.method == "GET":
        qs = PlaygroundSession.objects.filter(user=request.user).order_by("-updated_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            PlaygroundSessionSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = PlaygroundSessionSerializer(data=request.data)
        if serializer.is_valid():
            session = serializer.save(user=request.user)
            return Response(PlaygroundSessionSerializer(session).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def session_send_message(request, session_id):
    """
    Send a user message and get an AI response.
    This is a simplified simulation – in production, you'd call the actual LLM API.
    """
    session = get_object_or_404(PlaygroundSession, id=session_id, user=request.user)
    user_content = request.data.get("content", "")
    if not user_content:
        return Response({"error": "content required"}, status=400)

    start_time = time.time()

    # Save user message
    user_msg = PlaygroundMessage.objects.create(
        session=session, role="user", content=user_content
    )

    # Simulate LLM call (replace with actual API call in production)
    model_name = session.model_config.model_name if session.model_config else "gpt-4o"
    response_text = f"[Simulated {model_name} response to: {user_content[:100]}...]"

    latency = int((time.time() - start_time) * 1000)

    # Save assistant message
    assistant_msg = PlaygroundMessage.objects.create(
        session=session,
        role="assistant",
        content=response_text,
        token_count=len(response_text.split()),
        latency_ms=latency,
    )

    session.save(update_fields=["updated_at"])

    return Response(
        {
            "user_message": PlaygroundMessageSerializer(user_msg).data,
            "assistant_message": PlaygroundMessageSerializer(assistant_msg).data,
        }
    )


# ═══════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISONS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def comparison_list(request):
    if request.method == "GET":
        qs = ComparisonSet.objects.filter(user=request.user).order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ComparisonSetSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ComparisonSetSerializer(data=request.data)
        if serializer.is_valid():
            comp = serializer.save(user=request.user)
            # For each model config provided, run the comparison
            model_config_ids = request.data.get("model_config_ids", [])
            prompt_template_ids = request.data.get("prompt_template_ids", [])
            input_text = request.data.get("input_text", "")

            for i, mcid in enumerate(model_config_ids):
                config = ModelConfig.objects.get(id=mcid)
                prompt = (
                    PromptTemplate.objects.get(id=prompt_template_ids[i])
                    if i < len(prompt_template_ids)
                    else None
                )
                # Simulated output
                output = f"[Simulated output from {config.model_name}]"
                ComparisonResult.objects.create(
                    comparison_set=comp,
                    model_config=config,
                    prompt_template=prompt,
                    output_text=output,
                    latency_ms=500,
                    token_count=len(output.split()),
                )
            return Response(ComparisonSetSerializer(comp).data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def tool_list(request):
    if request.method == "GET":
        qs = ToolDefinition.objects.filter(
            Q(is_active=True) & (Q(created_by=request.user) | Q(is_active=True))
        ).order_by("name")
        return Response(ToolDefinitionSerializer(qs, many=True).data)

    elif request.method == "POST":
        serializer = ToolDefinitionSerializer(data=request.data)
        if serializer.is_valid():
            tool = serializer.save(created_by=request.user)
            return Response(ToolDefinitionSerializer(tool).data, status=201)
        return Response(serializer.errors, status=400)
