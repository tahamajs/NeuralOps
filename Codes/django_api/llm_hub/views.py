from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import *
from .gateway import LLMGateway
from .serializers import *

# ── Playground ───────────────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def playground_chat(request):
    """Send a chat message to a model and get a response."""
    model_id = request.data.get('model_id')
    messages = request.data.get('messages', [])
    temperature = request.data.get('temperature', 0.7)
    max_tokens = request.data.get('max_tokens', 1024)
    tools = request.data.get('tools')

    result = LLMGateway.complete(model_id, messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
    if 'error' in result:
        return Response(result, status=400)

    # Log usage
    LLMGateway.log_usage(request.user, model_id, result['tokens_input'], result['tokens_output'], result['cost_usd'])
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def playground_compare(request):
    """Compare the same prompt across multiple models side-by-side."""
    prompt = request.data.get('prompt', '')
    model_ids = request.data.get('model_ids', [])
    temperature = request.data.get('temperature', 0.7)
    results = {}
    for mid in model_ids:
        result = LLMGateway.complete(mid, [{'role': 'user', 'content': prompt}], temperature=temperature)
        results[mid] = result
    return Response(results)

# ── Prompt Management ────────────────────────────────────────────
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def prompt_list(request):
    if request.method == 'GET':
        prompts = PromptTemplate.objects.filter(Q(user=request.user) | Q(is_public=True))
        serializer = PromptTemplateSerializer(prompts, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = PromptTemplateSerializer(data=request.data)
        if serializer.is_valid():
            prompt = serializer.save(user=request.user)
            # Create initial version
            PromptVersion.objects.create(
                prompt=prompt, version_number=1,
                content=request.data.get('content', ''),
                model_config=request.data.get('model_config', {}),
                created_by=request.user,
            )
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def prompt_new_version(request, prompt_id):
    """Create a new version of a prompt."""
    prompt = get_object_or_404(PromptTemplate, id=prompt_id, user=request.user)
    prompt.current_version += 1
    prompt.save()
    version = PromptVersion.objects.create(
        prompt=prompt,
        version_number=prompt.current_version,
        content=request.data.get('content', ''),
        model_config=request.data.get('model_config', {}),
        change_notes=request.data.get('change_notes', ''),
        created_by=request.user,
    )
    return Response(PromptVersionSerializer(version).data, status=201)

# ── Evaluation ───────────────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_evaluation(request):
    """Start an evaluation run."""
    suite_id = request.data.get('suite_id')
    model_id = request.data.get('model_id')
    suite = get_object_or_404(EvalSuite, id=suite_id, user=request.user)
    model = get_object_or_404(LLMModel, model_id=model_id)
    run = EvalRun.objects.create(suite=suite, model=model, status='queued')
    from .tasks import run_evaluation
    run_evaluation.delay(str(run.id))
    return Response(EvalRunSerializer(run).data, status=201)

# ── Cost Dashboard ───────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def cost_dashboard(request):
    """Get cost summary for the authenticated user."""
    usage = UsageRecord.objects.filter(user=request.user)
    total_cost = usage.aggregate(s=models.Sum('cost_usd'))['s'] or 0
    total_tokens = usage.aggregate(s=models.Sum('tokens_input') + models.Sum('tokens_output'))['s'] or 0
    by_model = usage.values('model__display_name').annotate(cost=models.Sum('cost_usd')).order_by('-cost')
    by_source = usage.values('source').annotate(cost=models.Sum('cost_usd')).order_by('-cost')
    return Response({
        'total_cost_usd': float(total_cost),
        'total_tokens': total_tokens,
        'by_model': list(by_model),
        'by_source': list(by_source),
    })

# ── Smart Memory ─────────────────────────────────────────────────
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def memory_list(request):
    if request.method == 'GET':
        memories = MemoryEntry.objects.filter(user=request.user)
        project_id = request.query_params.get('project_id')
        if project_id:
            memories = memories.filter(project_id=project_id)
        return Response(MemoryEntrySerializer(memories, many=True).data)
    elif request.method == 'POST':
        serializer = MemoryEntrySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def memory_context(request):
    """Retrieve relevant memories for the current LLM context window."""
    project_id = request.query_params.get('project_id')
    memories = MemoryEntry.objects.filter(user=request.user, project_id=project_id).order_by('-importance')[:10]
    context = '\n'.join(f"- {m.key}: {m.value}" for m in memories)
    return Response({'context': context})