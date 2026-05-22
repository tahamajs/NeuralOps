import logging
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import AgentSession, AgentMessage, GeneratedIdea
from .serializers import AgentSessionSerializer, AgentMessageSerializer, GeneratedIdeaSerializer
from .tasks import generate_ideas_for_user

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def session_list(request):
    if request.method == 'GET':
        sessions = AgentSession.objects.filter(user=request.user).order_by('-updated_at')
        paginator = StandardPagination()
        page = paginator.paginate_queryset(sessions, request)
        return paginator.get_paginated_response(AgentSessionSerializer(page, many=True).data)
    elif request.method == 'POST':
        session = AgentSession.objects.create(user=request.user, title=request.data.get('title', ''))
        return Response(AgentSessionSerializer(session).data, status=201)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def session_detail(request, session_id):
    session = get_object_or_404(AgentSession, id=session_id, user=request.user)
    if request.method == 'GET':
        messages = session.messages.order_by('created_at')
        return Response({
            'session': AgentSessionSerializer(session).data,
            'messages': AgentMessageSerializer(messages, many=True).data,
        })
    elif request.method == 'DELETE':
        session.delete()
        return Response(status=204)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_ideas(request):
    """Manually trigger idea generation for the user."""
    task = generate_ideas_for_user.delay(request.user.id)
    return Response({'status': 'idea generation started', 'task_id': task.id})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_ideas(request):
    """Get generated ideas for the user."""
    qs = GeneratedIdea.objects.filter(user=request.user).order_by('-created_at')
    idea_type = request.query_params.get('type')
    if idea_type:
        qs = qs.filter(idea_type=idea_type)
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(GeneratedIdeaSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_idea(request, idea_id):
    """Toggle saving an idea."""
    idea = get_object_or_404(GeneratedIdea, id=idea_id, user=request.user)
    idea.is_saved = not idea.is_saved
    idea.save(update_fields=['is_saved'])
    return Response({'is_saved': idea.is_saved})