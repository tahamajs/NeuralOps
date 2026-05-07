import uuid
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from django.shortcuts import get_object_or_404

from .models import (
    ChatSession, ChatMessage, ResearchPaper, CitationEdge, TrainingAnalysis
)
from .serializers import (
    ChatSessionSerializer, ChatMessageSerializer,
    ResearchPaperSerializer, TrainingAnalysisSerializer
)
from .orchestrator import LangGraphAgent  # updated import
from .tools import (
    search_papers, save_paper, search_library,
    get_citation_graph, annotate_paper, analyze_training, suggest_next_experiment
)

# ----------------------------------------------------------------------
# Chat Sessions
# ----------------------------------------------------------------------
class ChatSessionListCreateView(generics.ListCreateAPIView):
    """List all chat sessions for the user, or create a new one."""
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by('-updated_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ChatSessionDetailView(generics.RetrieveDestroyAPIView):
    """Get or delete a single chat session."""
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
        return ChatMessage.objects.filter(
            session_id=session_id,
            session__user=self.request.user
        ).order_by('created_at')


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def send_message(request, session_id):
    """REST fallback for sending a message when WebSocket is unavailable."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)

    text = request.data.get('text', '')
    if not text:
        return Response({"error": "Message text required"}, status=status.HTTP_400_BAD_REQUEST)

    agent = LangGraphAgent(str(session.id))
    result = agent.run(text)

    messages = ChatMessage.objects.filter(session=session).order_by('created_at')
    serializer = ChatMessageSerializer(messages, many=True)

    return Response({
        'messages': serializer.data,
        'agent_response': result,
    })


# ----------------------------------------------------------------------
# Paper Research & Library
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_search_papers(request):
    """Search Semantic Scholar / arXiv for papers."""
    result = search_papers(request.data)
    return Response(result)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_save_paper(request):
    """Save a paper to the user's library."""
    data = {**request.data, "user_id": request.user.id}
    result = save_paper(data)
    return Response(result)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_search_library(request):
    """Search the user's saved paper library."""
    result = search_library({
        "user_id": request.user.id,
        "query": request.query_params.get("q", ""),
        "tag": request.query_params.get("tag", ""),
        "limit": int(request.query_params.get("limit", 20)),
    })
    return Response(result)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_saved_papers_list(request):
    """List all saved papers for the user."""
    papers = ResearchPaper.objects.filter(
        saved_by=request.user
    ).order_by('-created_at')
    serializer = ResearchPaperSerializer(papers, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_paper_detail(request, paper_id):
    """Get detail of a single saved paper."""
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    serializer = ResearchPaperSerializer(paper)
    return Response(serializer.data)


@api_view(['PATCH'])
@permission_classes([permissions.IsAuthenticated])
def api_annotate_paper(request, paper_id):
    """Add notes or tags to a saved paper."""
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    data = request.data
    annotate_paper({"paper_id": str(paper.id), **data})
    paper.refresh_from_db()
    serializer = ResearchPaperSerializer(paper)
    return Response(serializer.data)


@api_view(['DELETE'])
@permission_classes([permissions.IsAuthenticated])
def api_delete_paper(request, paper_id):
    """Remove a paper from the user's library."""
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    paper.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_citation_graph(request):
    """Get the citation graph for a set of saved papers."""
    result = get_citation_graph({
        "paper_ids": request.data.get("paper_ids", []),
        "user_id": request.user.id,
    })
    return Response(result)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_add_citation_edge(request):
    """Manually add a citation edge between two saved papers."""
    source_id = request.data.get("source_id")
    target_id = request.data.get("target_id")
    if not source_id or not target_id:
        return Response({"error": "source_id and target_id required"}, status=400)

    source = get_object_or_404(ResearchPaper, id=source_id, saved_by=request.user)
    target = get_object_or_404(ResearchPaper, id=target_id, saved_by=request.user)

    _, created = CitationEdge.objects.get_or_create(
        source_paper=source,
        target_paper=target,
        defaults={"relationship": request.data.get("relationship", "cites")}
    )
    return Response({"status": "created" if created else "already_exists"})


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_bulk_import_papers(request):
    """Bulk import multiple papers from search results."""
    papers_data = request.data.get("papers", [])
    saved = []
    for paper in papers_data:
        data = {**paper, "user_id": request.user.id}
        result = save_paper(data)
        saved.append(result)
    return Response({"saved": saved})


# ----------------------------------------------------------------------
# Training Analysis & Suggestions
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_analyze_run(request):
    """Analyze a training run for issues (overfitting, plateau, etc.)."""
    run_id = request.data.get("run_id")
    if not run_id:
        return Response({"error": "run_id required"}, status=400)
    result = analyze_training({"run_id": run_id})
    return Response(result)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_run_analysis(request, run_id):
    """Get the latest analysis for a specific run."""
    analysis = get_object_or_404(TrainingAnalysis, run_id=run_id)
    serializer = TrainingAnalysisSerializer(analysis)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_suggest_next(request):
    """Suggest the next experiment to try based on previous runs."""
    project_id = request.data.get("project_id")
    if not project_id:
        return Response({"error": "project_id required"}, status=400)
    result = suggest_next_experiment({
        "project_id": project_id,
        "metric_name": request.data.get("metric_name", "eval_accuracy"),
        "direction": request.data.get("direction", "maximize"),
    })
    return Response(result)


# ----------------------------------------------------------------------
# Conversation Analysis (summary, export)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_conversation_summary(request, session_id):
    """Get a summary of a conversation (key decisions, experiments launched)."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by('created_at')

    experiments_launched = messages.filter(role='experiment_card', metadata__type='experiment_launched').count()
    drafts_proposed = messages.filter(role='experiment_card', metadata__type='experiment_draft').count()
    papers_saved = messages.filter(role='tool', metadata__tool_name='save_paper').count()
    analyses_run = messages.filter(role='tool', metadata__tool_name='analyze_training').count()

    summary = {
        "session_id": str(session.id),
        "title": session.title,
        "total_messages": messages.count(),
        "experiments_launched": experiments_launched,
        "drafts_proposed": drafts_proposed,
        "papers_saved": papers_saved,
        "analyses_run": analyses_run,
        "key_findings": list(
            messages.filter(
                role='assistant',
                content__icontains='i recommend'
            ).values_list('content', flat=True)[:5]
        ),
    }
    return Response(summary)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_export_conversation(request, session_id):
    """Export a conversation as JSON for external analysis."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by('created_at')
    serializer = ChatMessageSerializer(messages, many=True)
    return Response({
        "session": ChatSessionSerializer(session).data,
        "messages": serializer.data,
    })