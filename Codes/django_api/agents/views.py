from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from .models import ChatSession, ChatMessage
from .serializers import ChatSessionSerializer, ChatMessageSerializer
from .orchestrator import AgentOrchestrator

class ChatSessionListCreateView(generics.ListCreateAPIView):
    """List all chat sessions for the user, or create a new one."""
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by('-updated_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ChatSessionDetailView(generics.RetrieveAPIView):
    """Get a single chat session with all messages."""
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)


class MessageListView(generics.ListAPIView):
    """List messages for a chat session."""
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        session_id = self.kwargs['session_id']
        return ChatMessage.objects.filter(session_id=session_id, session__user=self.request.user).order_by('created_at')


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def send_message(request, session_id):
    """REST fallback for sending a message (if WebSocket is unavailable)."""
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
    except ChatSession.DoesNotExist:
        return Response({"error": "Session not found"}, status=404)

    text = request.data.get('text', '')
    if not text:
        return Response({"error": "Message text required"}, status=400)

    orchestrator = AgentOrchestrator(session)
    result = orchestrator.process_user_message(text)

    messages = ChatMessage.objects.filter(session=session).order_by('created_at')
    serializer = ChatMessageSerializer(messages, many=True)

    return Response({
        'messages': serializer.data,
        'agent_response': result,
    })

from .tools import search_papers, save_paper, search_library, get_citation_graph
from .models import ResearchPaper

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_search_papers(request):
    result = search_papers(request.data)
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_save_paper(request):
    data = {**request.data, "user_id": request.user.id}
    result = save_paper(data)
    return Response(result)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_search_library(request):
    result = search_library({
        "user_id": request.user.id,
        "query": request.query_params.get("q", ""),
        "tag": request.query_params.get("tag", ""),
    })
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_citation_graph(request):
    result = get_citation_graph({
        "paper_ids": request.data.get("paper_ids", []),
        "user_id": request.user.id,
    })
    return Response(result)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_saved_papers_list(request):
    papers = ResearchPaper.objects.filter(saved_by=request.user).order_by('-created_at')
    return Response([{
        "id": str(p.id),
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "tags": p.tags,
        "citation_count": p.citation_count,
    } for p in papers])