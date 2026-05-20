import uuid
from datetime import datetime
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from django.shortcuts import get_object_or_404

from .models import (
    ChatSession, ChatMessage, ResearchPaper, CitationEdge, TrainingAnalysis,
    AgentSession, AgentMemory, AgentAction
)
from .serializers import (
    ChatSessionSerializer, ChatMessageSerializer,
    ResearchPaperSerializer, TrainingAnalysisSerializer
)
from .tools import (
    search_papers, save_paper, search_library,
    get_citation_graph, annotate_paper, analyze_training, suggest_next_experiment,
    test_prompt_template,
)
from .monitor import monitor_training_run
from .paper_recommender import (
    get_paper_recommendations_for_run,
    get_recommendations_from_saved_papers,
    search_papers_for_topic,
)
from experiment.models import TrainingRun   # for paper recommendations


# ----------------------------------------------------------------------
# Chat Sessions (unchanged)
# ----------------------------------------------------------------------
class ChatSessionListCreateView(generics.ListCreateAPIView):
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by('-updated_at')
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ChatSessionDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_url_kwarg = 'session_id'
    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)

class MessageListView(generics.ListAPIView):
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        session_id = self.kwargs['session_id']
        return ChatMessage.objects.filter(
            session_id=session_id, session__user=self.request.user
        ).order_by('created_at')

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def send_message(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    text = request.data.get('text', '')
    if not text:
        return Response({"error": "Message text required"}, status=400)
    ChatMessage.objects.create(session=session, role='user', content=text)
    try:
        from .orchestrator import LangGraphAgent
    except ImportError as exc:
        return Response({"error": "Agent runtime dependencies not installed", "detail": str(exc)}, status=503)
    agent = LangGraphAgent(str(session.id))
    result = agent.run(text)
    messages = ChatMessage.objects.filter(session=session).order_by('created_at')
    serializer = ChatMessageSerializer(messages, many=True)
    return Response({'messages': serializer.data, 'agent_response': result})


# ----------------------------------------------------------------------
# Paper Research & Library (unchanged)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_search_papers(request):
    return Response(search_papers(request.data))

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_save_paper(request):
    data = {**request.data, "user_id": request.user.id}
    return Response(save_paper(**data))

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_search_library(request):
    return Response(search_library(
        user_id=request.user.id,
        query=request.query_params.get("q", ""),
        tag=request.query_params.get("tag", ""),
        limit=int(request.query_params.get("limit", 20)),
    ))

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_saved_papers_list(request):
    papers = ResearchPaper.objects.filter(saved_by=request.user).order_by('-created_at')
    serializer = ResearchPaperSerializer(papers, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_paper_detail(request, paper_id):
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    return Response(ResearchPaperSerializer(paper).data)

@api_view(['PATCH'])
@permission_classes([permissions.IsAuthenticated])
def api_annotate_paper(request, paper_id):
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    annotate_paper(paper_id=str(paper.id), **request.data)
    paper.refresh_from_db()
    return Response(ResearchPaperSerializer(paper).data)

@api_view(['DELETE'])
@permission_classes([permissions.IsAuthenticated])
def api_delete_paper(request, paper_id):
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    paper.delete()
    return Response(status=204)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_citation_graph(request):
    return Response(get_citation_graph(paper_id=request.data.get("paper_id", ""),
                                       depth=request.data.get("depth", 1)))

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_add_citation_edge(request):
    source_id = request.data.get("source_id")
    target_id = request.data.get("target_id")
    if not source_id or not target_id:
        return Response({"error": "source_id and target_id required"}, status=400)
    source = get_object_or_404(ResearchPaper, id=source_id, saved_by=request.user)
    target = get_object_or_404(ResearchPaper, id=target_id, saved_by=request.user)
    _, created = CitationEdge.objects.get_or_create(
        source_paper=source, target_paper=target,
        defaults={"relationship": request.data.get("relationship", "cites")}
    )
    return Response({"status": "created" if created else "already_exists"})

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_bulk_import_papers(request):
    papers_data = request.data.get("papers", [])
    saved = []
    for paper in papers_data:
        paper["user_id"] = request.user.id
        saved.append(save_paper(**paper))
    return Response({"saved": saved})


# ----------------------------------------------------------------------
# Training Analysis & Suggestions (FIXED)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_analyze_run(request):
    run_id = request.data.get("run_id")
    if not run_id:
        return Response({"error": "run_id required"}, status=400)
    result = analyze_training(run_id=run_id)      # ← fixed
    return Response(result)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_run_analysis(request, run_id):
    analysis = get_object_or_404(TrainingAnalysis, run_id=run_id)
    return Response(TrainingAnalysisSerializer(analysis).data)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_suggest_next(request):
    project_id = request.data.get("project_id")
    if not project_id:
        return Response({"error": "project_id required"}, status=400)
    result = suggest_next_experiment(project_id=project_id,
                                     metric_name=request.data.get("metric_name", "eval_accuracy"),
                                     direction=request.data.get("direction", "maximize"))
    return Response(result)


# ----------------------------------------------------------------------
# Conversation Analysis (unchanged)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_conversation_summary(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by('created_at')
    summary = {
        "session_id": str(session.id), "title": session.title,
        "total_messages": messages.count(),
        "experiments_launched": messages.filter(role='experiment_card', metadata__type='experiment_launched').count(),
        "drafts_proposed": messages.filter(role='experiment_card', metadata__type='experiment_draft').count(),
        "papers_saved": messages.filter(role='tool', metadata__tool_name='save_paper').count(),
        "analyses_run": messages.filter(role='tool', metadata__tool_name='analyze_training').count(),
    }
    return Response(summary)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def api_export_conversation(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by('created_at')
    serializer = ChatMessageSerializer(messages, many=True)
    return Response({"session": ChatSessionSerializer(session).data, "messages": serializer.data})


# ----------------------------------------------------------------------
# Monitor & Paper Recommendations (unchanged)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def start_training_monitor(request):
    run_id = request.data.get("run_id")
    if not run_id:
        return Response({"error": "run_id required"}, status=400)
    task = monitor_training_run.delay(
        run_id=run_id, user_id=str(request.user.id),
        check_interval_ratio=float(request.data.get("check_interval_ratio", 0.25)),
        auto_stop_enabled=request.data.get("auto_stop_enabled", True),
        note_generation_enabled=request.data.get("note_generation_enabled", True),
    )
    return Response({"status": "monitor_started", "task_id": task.id, "run_id": run_id})

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def paper_recommendations_for_run(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    return Response({"recommendations": get_paper_recommendations_for_run(run, limit=10)})

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def paper_recommendations_from_library(request):
    return Response({"recommendations": get_recommendations_from_saved_papers(request.user.id, limit=10)})

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def search_papers_for_run_topic(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    config = run.config or {}
    topic_str = " ".join(filter(None, [
        config.get("attention_type"), config.get("method"),
        config.get("model_name"), config.get("task_name")
    ]))
    return Response({"topic": topic_str, "results": search_papers_for_topic(topic_str, limit=10)})

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def api_test_prompt(request):
    if not request.data.get("template") or not isinstance(request.data.get("test_cases"), list):
        return Response({"error": "template and test_cases[] required"}, status=400)
    return Response(test_prompt_template(**request.data))


# ----------------------------------------------------------------------
# NEW – Autonomous Agent Session endpoints
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def start_agent_session(request):
    objective = request.data.get('objective', '')
    if not objective:
        return Response({"error": "objective required"}, status=400)
    session = AgentSession.objects.create(
        user=request.user,
        name=request.data.get('name', f'Agent {datetime.now().strftime("%Y-%m-%d %H:%M")}'),
        objective=objective,
        agent_type=request.data.get('agent_type', 'deep_researcher'),
        max_cycles=request.data.get('max_cycles', 100),
        max_llm_cost_cents=request.data.get('max_llm_cost_cents', 500),
        project_id=request.data.get('project_id'),
        metric_name=request.data.get('metric_name', 'eval_accuracy'),
        metric_direction=request.data.get('metric_direction', 'maximize'),
    )
    from agents.tasks import run_autonomous_agent
    task = run_autonomous_agent.delay(str(session.id), request.user.id)
    return Response({"session_id": str(session.id), "task_id": task.id}, status=201)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def stop_agent_session(request, session_id):
    session = get_object_or_404(AgentSession, id=session_id, user=request.user)
    if session.status == 'active':
        session.status = 'paused'
        session.save()
    return Response({"status": session.status})