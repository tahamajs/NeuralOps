from datetime import datetime

from django.shortcuts import get_object_or_404
from experiment.models import TrainingRun  # for paper recommendations
from rest_framework import generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from .models import (
    AgentSession,
    ChatMessage,
    ChatSession,
    CitationEdge,
    ResearchPaper,
    TrainingAnalysis,
)
from .monitor import monitor_training_run
from .paper_recommender import (
    get_paper_recommendations_for_run,
    get_recommendations_from_saved_papers,
    search_papers_for_topic,
)
from .serializers import (
    ChatMessageSerializer,
    ChatSessionSerializer,
    ResearchPaperSerializer,
    TrainingAnalysisSerializer,
)
from .tools import (
    analyze_training,
    annotate_paper,
    get_citation_graph,
    save_paper,
    search_library,
    search_papers,
    suggest_next_experiment,
    test_prompt_template,
)


# ----------------------------------------------------------------------
# Chat Sessions (unchanged)
# ----------------------------------------------------------------------
class ChatSessionListCreateView(generics.ListCreateAPIView):
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by(
            "-updated_at"
        )

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ChatSessionDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_url_kwarg = "session_id"

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)


class MessageListView(generics.ListAPIView):
    serializer_class = ChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        session_id = self.kwargs["session_id"]
        return ChatMessage.objects.filter(
            session_id=session_id, session__user=self.request.user
        ).order_by("created_at")


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def send_message(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    text = request.data.get("text", "")
    if not text:
        return Response({"error": "Message text required"}, status=400)

    # Store user message
    ChatMessage.objects.create(session=session, role="user", content=text)

    # Update session title from first message
    if not session.title and len(text) > 0:
        session.title = text[:80]
        session.save(update_fields=["title"])

    assistant_content = None

    # Try LangGraph orchestrator first
    try:
        from .orchestrator import LangGraphAgent

        agent = LangGraphAgent(str(session.id))
        result = agent.run(text)
        assistant_content = (
            result.get("response", str(result))
            if isinstance(result, dict)
            else str(result)
        )
    except ImportError:
        pass
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("LangGraphAgent failed: %s", exc)

    # Fallback: rule-based intelligent responses
    if not assistant_content:
        tl = text.lower()
        if any(w in tl for w in ["paper", "search", "arxiv", "literature"]):
            assistant_content = (
                "I've received your research request. Here's what I can do:\n\n"
                "📚 **Paper Search**: I can search Semantic Scholar and ArXiv for papers matching your query.\n"
                "🔗 **Citation Graph**: I can build a citation network to show related work.\n"
                "💾 **Save to Library**: Found papers can be saved to your personal library.\n\n"
                "⚠️ *Note: The AI backend is currently starting up. Please try again in a moment or check the system status.*"
            )
        elif any(
            w in tl
            for w in ["experiment", "train", "model", "bert", "gpt", "lora", "fine-tun"]
        ):
            assistant_content = (
                "I understand you want to configure an experiment. Here's what I need:\n\n"
                "🔬 **Model**: Which base model? (e.g. bert-base-uncased, distilbert)\n"
                "📊 **Task**: Which benchmark? (e.g. SST-2, MRPC, CoLA)\n"
                "⚡ **Method**: Compression method? (LoRA, SVD, full fine-tuning)\n\n"
                "Please specify these details and I'll prepare the full experiment configuration for review before launching.\n\n"
                "⚠️ *The AI orchestrator is warming up. Full LLM responses will be available shortly.*"
            )
        elif any(w in tl for w in ["analyz", "metric", "loss", "accuracy", "overfit"]):
            assistant_content = (
                "Training analysis mode activated. I'll inspect your runs for:\n\n"
                "📈 **Overfitting**: Diverging train/val loss\n"
                "📉 **Underfitting**: Both losses still decreasing at end of training\n"
                "⚡ **Plateau**: Learning stagnation — may need LR adjustment\n"
                "🔥 **Diverging**: NaN/Inf losses or exploding gradients\n\n"
                "Please provide the experiment run ID or navigate to `/experiments` to select a run.\n\n"
                "⚠️ *Full AI analysis requires the LLM backend to be running.*"
            )
        elif any(w in tl for w in ["hello", "hi", "hey", "help", "what"]):
            assistant_content = (
                "👋 Hello! I'm your **ML Research Assistant**.\n\n"
                "I can help you with:\n"
                "- 📚 **Search papers** on ArXiv & Semantic Scholar\n"
                "- 🔬 **Configure & launch** ML experiments\n"
                "- 📊 **Analyze** training metrics and suggest fixes\n"
                "- 🔗 **Build citation graphs** for any paper\n"
                "- ✍️ **Edit your LaTeX paper** draft\n"
                "- 💡 **Suggest next experiments** based on results\n\n"
                "What would you like to work on today?\n\n"
                "⚠️ *Note: Full AI capabilities require the LLM service. Ensure `OPENAI_API_KEY` is set in your environment.*"
            )
        else:
            assistant_content = (
                f"I received your message: *\"{text[:100]}{'...' if len(text) > 100 else ''}\"*\n\n"
                "I'm your ML Research Assistant. I can help with paper search, experiment configuration, "
                "training analysis, and more.\n\n"
                "⚠️ **AI Backend Status**: The LLM orchestrator is currently unavailable. "
                "To enable full AI responses, ensure the following are installed and configured:\n"
                "```\npip install langgraph langchain-openai\nexport OPENAI_API_KEY=your_key\n```"
            )

    # Save assistant response
    ChatMessage.objects.create(
        session=session, role="assistant", content=assistant_content
    )

    messages = ChatMessage.objects.filter(session=session).order_by("created_at")
    serializer = ChatMessageSerializer(messages, many=True)
    return Response({"messages": serializer.data})


# ----------------------------------------------------------------------
# Paper Research & Library (unchanged)
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_search_papers(request):
    return Response(search_papers(request.data))


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_save_paper(request):
    data = {**request.data, "user_id": request.user.id}
    return Response(save_paper(**data))


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def api_search_library(request):
    return Response(
        search_library(
            user_id=request.user.id,
            query=request.query_params.get("q", ""),
            tag=request.query_params.get("tag", ""),
            limit=int(request.query_params.get("limit", 20)),
        )
    )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def api_saved_papers_list(request):
    papers = ResearchPaper.objects.filter(saved_by=request.user).order_by("-created_at")
    serializer = ResearchPaperSerializer(papers, many=True)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def api_paper_detail(request, paper_id):
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    return Response(ResearchPaperSerializer(paper).data)


@api_view(["PATCH"])
@permission_classes([permissions.IsAuthenticated])
def api_annotate_paper(request, paper_id):
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    annotate_paper(paper_id=str(paper.id), **request.data)
    paper.refresh_from_db()
    return Response(ResearchPaperSerializer(paper).data)


@api_view(["DELETE"])
@permission_classes([permissions.IsAuthenticated])
def api_delete_paper(request, paper_id):
    paper = get_object_or_404(ResearchPaper, id=paper_id, saved_by=request.user)
    paper.delete()
    return Response(status=204)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_citation_graph(request):
    return Response(
        get_citation_graph(
            paper_id=request.data.get("paper_id", ""),
            depth=request.data.get("depth", 1),
        )
    )


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_add_citation_edge(request):
    source_id = request.data.get("source_id")
    target_id = request.data.get("target_id")
    if not source_id or not target_id:
        return Response({"error": "source_id and target_id required"}, status=400)
    source = get_object_or_404(ResearchPaper, id=source_id, saved_by=request.user)
    target = get_object_or_404(ResearchPaper, id=target_id, saved_by=request.user)
    _, created = CitationEdge.objects.get_or_create(
        source_paper=source,
        target_paper=target,
        defaults={"relationship": request.data.get("relationship", "cites")},
    )
    return Response({"status": "created" if created else "already_exists"})


@api_view(["POST"])
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
@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_analyze_run(request):
    run_id = request.data.get("run_id")
    if not run_id:
        return Response({"error": "run_id required"}, status=400)
    result = analyze_training(run_id=run_id)  # ← fixed
    return Response(result)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def api_run_analysis(request, run_id):
    analysis = get_object_or_404(TrainingAnalysis, run_id=run_id)
    return Response(TrainingAnalysisSerializer(analysis).data)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_suggest_next(request):
    project_id = request.data.get("project_id")
    if not project_id:
        return Response({"error": "project_id required"}, status=400)
    result = suggest_next_experiment(
        project_id=project_id,
        metric_name=request.data.get("metric_name", "eval_accuracy"),
        direction=request.data.get("direction", "maximize"),
    )
    return Response(result)


# ----------------------------------------------------------------------
# Conversation Analysis (unchanged)
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def api_conversation_summary(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by("created_at")
    summary = {
        "session_id": str(session.id),
        "title": session.title,
        "total_messages": messages.count(),
        "experiments_launched": messages.filter(
            role="experiment_card", metadata__type="experiment_launched"
        ).count(),
        "drafts_proposed": messages.filter(
            role="experiment_card", metadata__type="experiment_draft"
        ).count(),
        "papers_saved": messages.filter(
            role="tool", metadata__tool_name="save_paper"
        ).count(),
        "analyses_run": messages.filter(
            role="tool", metadata__tool_name="analyze_training"
        ).count(),
    }
    return Response(summary)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def api_export_conversation(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.order_by("created_at")
    serializer = ChatMessageSerializer(messages, many=True)
    return Response(
        {"session": ChatSessionSerializer(session).data, "messages": serializer.data}
    )


# ----------------------------------------------------------------------
# Monitor & Paper Recommendations (unchanged)
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def start_training_monitor(request):
    run_id = request.data.get("run_id")
    if not run_id:
        return Response({"error": "run_id required"}, status=400)
    task = monitor_training_run.delay(
        run_id=run_id,
        user_id=str(request.user.id),
        check_interval_ratio=float(request.data.get("check_interval_ratio", 0.25)),
        auto_stop_enabled=request.data.get("auto_stop_enabled", True),
        note_generation_enabled=request.data.get("note_generation_enabled", True),
    )
    return Response({"status": "monitor_started", "task_id": task.id, "run_id": run_id})


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def paper_recommendations_for_run(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    return Response(
        {"recommendations": get_paper_recommendations_for_run(run, limit=10)}
    )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def paper_recommendations_from_library(request):
    return Response(
        {
            "recommendations": get_recommendations_from_saved_papers(
                request.user.id, limit=10
            )
        }
    )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def search_papers_for_run_topic(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    config = run.config or {}
    topic_str = " ".join(
        filter(
            None,
            [
                config.get("attention_type"),
                config.get("method"),
                config.get("model_name"),
                config.get("task_name"),
            ],
        )
    )
    return Response(
        {"topic": topic_str, "results": search_papers_for_topic(topic_str, limit=10)}
    )


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def api_test_prompt(request):
    if not request.data.get("template") or not isinstance(
        request.data.get("test_cases"), list
    ):
        return Response({"error": "template and test_cases[] required"}, status=400)
    return Response(test_prompt_template(**request.data))


# ----------------------------------------------------------------------
# NEW – Autonomous Agent Session endpoints
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def start_agent_session(request):
    objective = request.data.get("objective", "")
    if not objective:
        return Response({"error": "objective required"}, status=400)
    session = AgentSession.objects.create(
        user=request.user,
        name=request.data.get(
            "name", f'Agent {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        ),
        objective=objective,
        agent_type=request.data.get("agent_type", "deep_researcher"),
        max_cycles=request.data.get("max_cycles", 100),
        max_llm_cost_cents=request.data.get("max_llm_cost_cents", 500),
        project_id=request.data.get("project_id"),
        metric_name=request.data.get("metric_name", "eval_accuracy"),
        metric_direction=request.data.get("metric_direction", "maximize"),
    )
    from agents.tasks import run_autonomous_agent

    task = run_autonomous_agent.delay(str(session.id), request.user.id)
    return Response({"session_id": str(session.id), "task_id": task.id}, status=201)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def stop_agent_session(request, session_id):
    session = get_object_or_404(AgentSession, id=session_id, user=request.user)
    if session.status == "active":
        session.status = "paused"
        session.save()
    return Response({"status": session.status})


# agents/views.py — add these


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def create_research_plan(request):
    """Create a research plan and start async execution."""
    plan = ResearchPlan.objects.create(
        user=request.user,
        title=request.data.get("title", ""),
        objective=request.data.get("objective", ""),
        steps=request.data.get("steps", []),
        status="in_progress",
    )
    # Trigger Celery task
    from agents.tasks import execute_research_plan

    execute_research_plan.delay(str(plan.id))
    return Response({"plan_id": str(plan.id)}, status=201)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def research_plan_status(request, plan_id):
    """Get the current status of a research plan."""
    plan = get_object_or_404(ResearchPlan, id=plan_id, user=request.user)
    findings = plan.findings.order_by("-relevance_score")
    return Response(
        {
            "plan_id": str(plan.id),
            "status": plan.status,
            "steps": plan.steps,
            "findings": [
                {
                    "title": f.source_title,
                    "url": f.source_url,
                    "credibility": f.credibility_score,
                    "summary": f.summary,
                }
                for f in findings
            ],
            "report": (
                plan.researchreport.content_markdown
                if hasattr(plan, "researchreport")
                else None
            ),
        }
    )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def list_tools(request):
    """List all available tools in ToolRegistry."""
    from agents.tools import ToolRegistry

    tools_list = []
    for tool_name, tool_info in ToolRegistry._tools.items():
        tools_list.append(
            {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "category": tool_info["category"],
                "requires_confirmation": tool_info["requires_confirmation"],
                "parameters": {},
            }
        )
    return Response(tools_list)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def agent_status(request):
    """Get the current operational status of the agent subsystem."""
    from agents.tools import ToolRegistry

    active_sessions = AgentSession.objects.filter(
        user=request.user, status="active"
    ).count()
    return Response(
        {
            "status": "operational",
            "model": "claude-3-5-sonnet",
            "tools_count": len(ToolRegistry._tools),
            "active_sessions": active_sessions,
            "ws_enabled": True,
        }
    )
