from django.db.models import Q, Sum
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .gateway import LLMGateway
from .memory import SmartMemory
from .models import *
from .serializers import *


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# PROVIDERS & MODELS (read‑only)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def provider_list(request):
    providers = LLMProvider.objects.filter(is_active=True)
    return Response(LLMProviderSerializer(providers, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_list(request):
    models = LLMModel.objects.filter(is_active=True).select_related("provider")
    provider = request.query_params.get("provider")
    if provider:
        models = models.filter(provider__name=provider)
    return Response(LLMModelSerializer(models, many=True).data)


# ═══════════════════════════════════════════════════════════════
# PLAYGROUND
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def playground_chat(request):
    model_id = request.data.get("model_id")
    messages = request.data.get("messages", [])
    temperature = request.data.get("temperature", 0.7)
    max_tokens = request.data.get("max_tokens", 1024)
    tools = request.data.get("tools")

    result = LLMGateway.complete(
        model_id, messages, temperature=temperature, max_tokens=max_tokens, tools=tools
    )
    if "error" in result:
        return Response(result, status=400)

    # Log usage
    LLMGateway.log_usage(
        request.user,
        model_id,
        result["tokens_input"],
        result["tokens_output"],
        result["cost_usd"],
    )
    return Response(result)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def playground_compare(request):
    prompt = request.data.get("prompt", "")
    model_ids = request.data.get("model_ids", [])
    temperature = request.data.get("temperature", 0.7)
    results = {}
    for mid in model_ids:
        r = LLMGateway.complete(
            mid, [{"role": "user", "content": prompt}], temperature=temperature
        )
        results[mid] = r
    return Response(results)


# ═══════════════════════════════════════════════════════════════
# PROMPT MANAGEMENT
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def prompt_list(request):
    if request.method == "GET":
        prompts = PromptTemplate.objects.filter(
            Q(user=request.user) | Q(is_public=True)
        ).order_by("-updated_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(prompts, request)
        return paginator.get_paginated_response(
            PromptTemplateSerializer(page, many=True).data
        )
    elif request.method == "POST":
        serializer = PromptTemplateSerializer(data=request.data)
        if serializer.is_valid():
            prompt = serializer.save(user=request.user)
            # Create initial version
            PromptVersion.objects.create(
                prompt=prompt,
                version_number=1,
                content=request.data.get("content", ""),
                model_config=request.data.get("model_config", {}),
                created_by=request.user,
            )
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def prompt_detail(request, prompt_id):
    prompt = get_object_or_404(PromptTemplate, id=prompt_id)
    if request.method == "GET":
        return Response(PromptTemplateSerializer(prompt).data)
    elif request.method == "PATCH":
        if prompt.user != request.user:
            return Response({"error": "Not allowed"}, status=403)
        serializer = PromptTemplateSerializer(prompt, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if prompt.user != request.user:
            return Response({"error": "Not allowed"}, status=403)
        prompt.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def prompt_new_version(request, prompt_id):
    prompt = get_object_or_404(PromptTemplate, id=prompt_id, user=request.user)
    prompt.current_version += 1
    prompt.save()
    version = PromptVersion.objects.create(
        prompt=prompt,
        version_number=prompt.current_version,
        content=request.data.get("content", ""),
        model_config=request.data.get("model_config", {}),
        change_notes=request.data.get("change_notes", ""),
        created_by=request.user,
    )
    return Response(PromptVersionSerializer(version).data, status=201)


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def eval_suite_list(request):
    if request.method == "GET":
        suites = EvalSuite.objects.filter(user=request.user)
        return Response(EvalSuiteSerializer(suites, many=True).data)
    elif request.method == "POST":
        serializer = EvalSuiteSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_evaluation(request):
    suite_id = request.data.get("suite_id")
    model_id = request.data.get("model_id")
    suite = get_object_or_404(EvalSuite, id=suite_id, user=request.user)
    model = get_object_or_404(LLMModel, model_id=model_id)
    run = EvalRun.objects.create(suite=suite, model=model, status="queued")
    from .tasks import run_evaluation

    run_evaluation.delay(str(run.id))
    return Response(EvalRunSerializer(run).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def eval_run_detail(request, run_id):
    run = get_object_or_404(EvalRun, id=run_id)
    return Response(EvalRunSerializer(run).data)


# ═══════════════════════════════════════════════════════════════
# COST DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def cost_dashboard(request):
    usage = UsageRecord.objects.filter(user=request.user)
    total_cost = usage.aggregate(s=Sum("cost_usd"))["s"] or 0
    total_tokens = (
        usage.aggregate(s=Sum("tokens_input") + Sum("tokens_output"))["s"] or 0
    )
    by_model = (
        usage.values("model__display_name")
        .annotate(cost=Sum("cost_usd"))
        .order_by("-cost")
    )
    by_source = usage.values("source").annotate(cost=Sum("cost_usd")).order_by("-cost")
    return Response(
        {
            "total_cost_usd": float(total_cost),
            "total_tokens": total_tokens,
            "by_model": list(by_model),
            "by_source": list(by_source),
            "recent_usage": UsageRecordSerializer(
                usage.order_by("-created_at")[:20], many=True
            ).data,
        }
    )


# ═══════════════════════════════════════════════════════════════
# SMART MEMORY
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def memory_list(request):
    if request.method == "GET":
        memories = MemoryEntry.objects.filter(user=request.user)
        project_id = request.query_params.get("project_id")
        if project_id:
            memories = memories.filter(project_id=project_id)
        return Response(MemoryEntrySerializer(memories, many=True).data)
    elif request.method == "POST":
        serializer = MemoryEntrySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def memory_context(request):
    project_id = request.query_params.get("project_id")
    context = SmartMemory.get_context(request.user, project_id)
    return Response({"context": context})


# ═══════════════════════════════════════════════════════════════
# TOOLS & SCHEMAS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def tool_list(request):
    if request.method == "GET":
        tools = ToolDefinition.objects.filter(Q(user=request.user) | Q(is_public=True))
        return Response(ToolDefinitionSerializer(tools, many=True).data)
    elif request.method == "POST":
        serializer = ToolDefinitionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def schema_list(request):
    if request.method == "GET":
        schemas = StructuredOutputSchema.objects.filter(user=request.user)
        return Response(StructuredOutputSchemaSerializer(schemas, many=True).data)
    elif request.method == "POST":
        serializer = StructuredOutputSchemaSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


import random


@api_view(["GET"])
@permission_classes([AllowAny])
def get_battle(request):
    """Get a random battle between two models. If not enough, create new ones."""
    # Try to find an existing battle with no votes yet
    battles = ArenaBattle.objects.annotate(vote_count=models.Count("votes")).filter(
        vote_count=0
    )
    if battles.exists():
        battle = random.choice(battles)
        serializer = ArenaBattleSerializer(battle)
        return Response(serializer.data)
    # If none, we can create synthetic ones? Or require admin to seed.
    # For now, we'll create a new battle on the fly if we have at least two active models.
    models = LLMModel.objects.filter(is_active=True)
    if models.count() < 2:
        return Response({"error": "Not enough models for battle"}, status=400)
    model_a, model_b = random.sample(list(models), 2)
    prompt = (
        "Explain the concept of blockchain in simple terms."  # could be from a pool
    )
    # Call the gateway to get responses
    from .gateway import LLMGateway

    resp_a = LLMGateway.complete(
        model_a.model_id, [{"role": "user", "content": prompt}]
    )
    resp_b = LLMGateway.complete(
        model_b.model_id, [{"role": "user", "content": prompt}]
    )
    battle = ArenaBattle.objects.create(
        model_a=model_a,
        model_b=model_b,
        prompt=prompt,
        response_a=resp_a.get("content", ""),
        response_b=resp_b.get("content", ""),
    )
    serializer = ArenaBattleSerializer(battle)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes([AllowAny])
def submit_vote(request, battle_id):
    """Submit a vote for a battle."""
    battle = get_object_or_404(ArenaBattle, id=battle_id)
    winner = request.data.get("winner")
    if winner not in ("A", "B", "tie"):
        return Response({"error": "Invalid winner"}, status=400)
    ArenaVote.objects.create(
        battle=battle,
        user=request.user if request.user.is_authenticated else None,
        winner=winner,
    )
    # Update ELO ratings
    _update_elo(battle)
    return Response({"status": "vote recorded"})


@api_view(["GET"])
@permission_classes([AllowAny])
def arena_leaderboard(request):
    """Get the arena leaderboard sorted by ELO rating."""
    elos = ModelElo.objects.select_related("model").order_by("-rating")
    data = [
        {
            "model_id": e.model.model_id,
            "display_name": e.model.display_name,
            "rating": e.rating,
            "matches": e.matches_played,
        }
        for e in elos
    ]
    return Response(data)


def _update_elo(battle):
    """Update ELO ratings based on battle votes."""
    votes = battle.votes.all()
    if not votes:
        return
    # Simple: for each vote, adjust ELO (K=32)
    K = 32
    for vote in votes:
        elo_a, _ = ModelElo.objects.get_or_create(model=battle.model_a)
        elo_b, _ = ModelElo.objects.get_or_create(model=battle.model_b)
        if vote.winner == "A":
            expected_a = 1 / (1 + 10 ** ((elo_b.rating - elo_a.rating) / 400))
            elo_a.rating += K * (1 - expected_a)
            elo_b.rating += K * (0 - (1 - expected_a))
            elo_a.matches_played += 1
            elo_b.matches_played += 1
        elif vote.winner == "B":
            expected_b = 1 / (1 + 10 ** ((elo_a.rating - elo_b.rating) / 400))
            elo_a.rating += K * (0 - expected_b)
            elo_b.rating += K * (1 - expected_b)
            elo_a.matches_played += 1
            elo_b.matches_played += 1
        # tie: small adjustment towards each other
        else:
            elo_a.rating = elo_a.rating + K * (
                0.5 - 1 / (1 + 10 ** ((elo_b.rating - elo_a.rating) / 400))
            )
            elo_b.rating = elo_b.rating + K * (
                0.5 - 1 / (1 + 10 ** ((elo_a.rating - elo_b.rating) / 400))
            )
            elo_a.matches_played += 1
            elo_b.matches_played += 1
        elo_a.save()
        elo_b.save()


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_guardrail(request):
    serializer = GuardrailSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(created_by=request.user)
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def attach_guardrail(request, session_id):
    session = get_object_or_404(PlaygroundSession, id=session_id, user=request.user)
    guardrail_id = request.data.get("guardrail_id")
    guardrail = get_object_or_404(Guardrail, id=guardrail_id)
    GuardrailAttachment.objects.get_or_create(guardrail=guardrail, session=session)
    return Response({"status": "attached"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def test_guardrail(request, guardrail_id):
    guardrail = get_object_or_404(Guardrail, id=guardrail_id)
    text = request.data.get("text", "")
    result = GuardrailEngine.apply(text, guardrail)
    return Response({"original": text, "filtered": result})
