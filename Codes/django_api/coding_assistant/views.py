import logging
from django.shortcuts import get_object_or_404
from django.http import StreamingHttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import CodeGenerationSession, CodeMessage, GeneratedCode, CodeReview, TestGeneration
from .serializers import *
from .llm_client import generate_code, review_code, generate_tests, get_llm_response

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Sessions
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def session_list(request):
    if request.method == 'GET':
        qs = CodeGenerationSession.objects.filter(user=request.user).order_by('-updated_at')
        serializer = CodeGenerationSessionSerializer(qs, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = CodeGenerationSessionSerializer(data=request.data)
        if serializer.is_valid():
            session = serializer.save(user=request.user)
            return Response(CodeGenerationSessionSerializer(session).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def session_detail(request, session_id):
    session = get_object_or_404(CodeGenerationSession, id=session_id, user=request.user)
    if request.method == 'GET':
        return Response(CodeGenerationSessionDetailSerializer(session).data)
    elif request.method == 'DELETE':
        session.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Chat (Send Message → Get Code)
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def send_message(request, session_id):
    """Send a prompt and get AI‑generated code back."""
    session = get_object_or_404(CodeGenerationSession, id=session_id, user=request.user)
    user_content = request.data.get('content', '')
    language = request.data.get('language', 'python')

    if not user_content.strip():
        return Response({"error": "content required"}, status=400)

    # Save user message
    CodeMessage.objects.create(session=session, role='user', content=user_content, language=language)

    # Build message history
    history = session.messages.order_by('created_at')
    messages = [{'role': 'system', 'content': f'You are an expert {language} developer. Write clean, well-documented code. Context: {session.context or {}}'}]
    for msg in history:
        messages.append({'role': msg.role, 'content': msg.content})

    # Generate code
    response = get_llm_response(messages)

    # Save assistant message
    assistant_msg = CodeMessage.objects.create(
        session=session, role='assistant', content=response, language=language
    )
    session.save(update_fields=['updated_at'])

    return Response(CodeMessageSerializer(assistant_msg).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def stream_message(request, session_id):
    """Stream the AI response as Server‑Sent Events."""
    session = get_object_or_404(CodeGenerationSession, id=session_id, user=request.user)
    user_content = request.data.get('content', '')
    language = request.data.get('language', 'python')

    if not user_content.strip():
        return Response({"error": "content required"}, status=400)

    CodeMessage.objects.create(session=session, role='user', content=user_content, language=language)

    history = session.messages.order_by('created_at')
    messages = [{'role': 'system', 'content': f'You are an expert {language} developer.'}]
    for msg in history:
        messages.append({'role': msg.role, 'content': msg.content})

    def generate():
        full_response = ''
        for chunk in get_llm_response(messages, stream=True):
            full_response += chunk
            yield f"data: {chunk}\n\n"
        # Save final response
        CodeMessage.objects.create(
            session=session, role='assistant', content=full_response, language=language
        )
        yield "data: [DONE]\n\n"

    return StreamingHttpResponse(generate(), content_type='text/event-stream')


# ═══════════════════════════════════════════════════════════════
# Generated Code Management
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_generated_code(request, session_id):
    """Save a generated code snippet to a file."""
    session = get_object_or_404(CodeGenerationSession, id=session_id, user=request.user)
    serializer = GeneratedCodeSerializer(data=request.data)
    if serializer.is_valid():
        code_file = serializer.save(session=session)
        return Response(GeneratedCodeSerializer(code_file).data, status=201)
    return Response(serializer.errors, status=400)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def generated_code_detail(request, file_id):
    code_file = get_object_or_404(GeneratedCode, id=file_id, session__user=request.user)
    if request.method == 'GET':
        return Response(GeneratedCodeSerializer(code_file).data)
    elif request.method == 'PATCH':
        serializer = GeneratedCodeSerializer(code_file, data=request.data, partial=True)
        if serializer.is_valid():
            code_file = serializer.save()
            code_file.version += 1
            code_file.save(update_fields=['version'])
            return Response(GeneratedCodeSerializer(code_file).data)
        return Response(serializer.errors, status=400)
    elif request.method == 'DELETE':
        code_file.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Code Review
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def request_review(request, file_id):
    """Request an AI code review for a generated file."""
    code_file = get_object_or_404(GeneratedCode, id=file_id, session__user=request.user)
    review_data = review_code(code_file.content, code_file.language)

    review = CodeReview.objects.create(
        generated_code=code_file,
        issues=review_data.get('issues', []),
        suggestions=review_data.get('suggestions', []),
        overall_assessment=review_data.get('overall_assessment', ''),
    )
    code_file.status = 'reviewing'
    code_file.save(update_fields=['status'])
    return Response(CodeReviewSerializer(review).data, status=201)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_review(request, file_id):
    code_file = get_object_or_404(GeneratedCode, id=file_id, session__user=request.user)
    reviews = code_file.reviews.order_by('-created_at')
    return Response(CodeReviewSerializer(reviews, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Test Generation
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_test(request, file_id):
    """Generate unit tests for a code file."""
    code_file = get_object_or_404(GeneratedCode, id=file_id, session__user=request.user)
    framework = request.data.get('framework', 'pytest')

    test_content = generate_tests(code_file.content, code_file.language, framework)
    test, created = TestGeneration.objects.update_or_create(
        generated_code=code_file,
        defaults={
            'test_content': test_content,
            'test_framework': framework,
            'coverage_estimate': 75.0,  # placeholder
        }
    )
    return Response(TestGenerationSerializer(test).data, status=201 if created else 200)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_test(request, file_id):
    code_file = get_object_or_404(GeneratedCode, id=file_id, session__user=request.user)
    if not hasattr(code_file, 'generated_tests'):
        return Response({"error": "No tests generated yet"}, status=404)
    return Response(TestGenerationSerializer(code_file.generated_tests).data)


# ═══════════════════════════════════════════════════════════════
# Experiment Config Generation
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_experiment_config(request):
    """Generate a training/experiment config based on user inputs."""
    model_name = request.data.get('model_name', 'distilbert-base-uncased')
    task_name = request.data.get('task_name', 'mrpc')
    attention_type = request.data.get('attention_type', 'full')
    method = request.data.get('method', 'baseline')

    prompt = f"""Generate a Python training configuration file for an NLP experiment with:
- Model: {model_name}
- Task: {task_name}
- Attention type: {attention_type}
- Method: {method}

The config should be a Python dictionary containing all necessary hyperparameters (learning_rate, batch_size, epochs, max_seq_length, etc.).
Return ONLY the Python code, no explanation."""

    code = generate_code(prompt, 'python')
    return Response({'code': code, 'language': 'python'})


# ═══════════════════════════════════════════════════════════════
# Documentation Generation
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_documentation(request, file_id):
    """Generate documentation for a code file."""
    code_file = get_object_or_404(GeneratedCode, id=file_id, session__user=request.user)

    prompt = f"Generate comprehensive documentation (docstrings and comments) for the following {code_file.language} code. Return ONLY the code with documentation added:\n\n{code_file.content}"

    documented_code = generate_code(prompt, code_file.language)
    return Response({'documented_code': documented_code})