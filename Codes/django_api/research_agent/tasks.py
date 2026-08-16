import json
import logging

from celery import shared_task
from django.conf import settings

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from .models import GeneratedIdea

logger = logging.getLogger(__name__)


@shared_task
def generate_ideas_for_user(user_id):
    """Generate research ideas based on a user's history."""
    from agents.models import ResearchPaper
    from django.contrib.auth import get_user_model
    from experiment.models import TrainingRun
    from notes.models import ResearcherNote
    from reputation.models import ReputationScore

    User = get_user_model()
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return 0

    api_key = getattr(settings, "OPENAI_API_KEY", "")
    if ChatOpenAI is None or not api_key:
        logger.warning("ChatOpenAI or OPENAI_API_KEY not configured. Skipping automated idea generation.")
        return 0

    # Collect data
    runs = TrainingRun.objects.filter(user=user).order_by("-created_at")[:10]
    notes = ResearcherNote.objects.filter(
        author=user, deleted_at__isnull=True
    ).order_by("-created_at")[:5]
    papers = ResearchPaper.objects.filter(saved_by=user).order_by("-created_at")[:5]
    score = ReputationScore.objects.filter(user=user).first()

    context = f"""
User: {user.username}
Experiments: {json.dumps([{'name': r.name, 'status': r.status, 'metrics': r.final_metrics} for r in runs])}
Notes: {json.dumps([n.title for n in notes])}
Papers: {json.dumps([p.title for p in papers])}
Reputation: {score.total_score if score else 0}
"""

    try:
        llm = ChatOpenAI(
            model=getattr(settings, "OPENAI_MODEL", "gpt-4o"),
            temperature=0.7,
            api_key=api_key,
        )

        prompt = f"""Based on this researcher's activity, generate 3 personalised ideas for their next steps. Each idea should include a title, a short description, and a type (experiment, collaboration, paper_topic, improvement, or compression_strategy).

{context}

Return the ideas as a JSON list:
[{{"title": "...", "description": "...", "type": "..."}}, ...]
"""
        response = llm.invoke(prompt)
        ideas = json.loads(response.content)
    except Exception as e:
        logger.warning("Failed to generate ideas via ChatOpenAI: %s", e)
        ideas = []

    for idea in ideas:
        GeneratedIdea.objects.create(
            user=user,
            title=idea.get("title", "Untitled"),
            description=idea.get("description", ""),
            idea_type=idea.get("type", "experiment"),
        )

    return len(ideas)


@shared_task
def store_experiment_in_memory(user_id, run_id):
    """Store an experiment summary in ChromaDB for long‑term memory."""
    # Implementation depends on your ChromaDB setup
    pass
