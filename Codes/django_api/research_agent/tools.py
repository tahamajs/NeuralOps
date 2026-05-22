"""
LangChain tools for the research agent – access to experiments, notes, papers, and reputation.
"""
import json
from langchain_core.tools import tool
from django.contrib.auth import get_user_model
from experiment.models import TrainingRun
from notes.models import ResearcherNote
from agents.models import ResearchPaper
from reputation.models import ReputationScore
from collaborator_matching.models import CollaboratorMatch

User = get_user_model()


def _get_user_from_run_id(run_id: str) -> int:
    try:
        run = TrainingRun.objects.get(id=run_id)
        return run.user_id
    except TrainingRun.DoesNotExist:
        return None


@tool
def get_user_experiments(user_id: int, status: str = None, limit: int = 5):
    """Get recent experiments for a user, optionally filtered by status."""
    qs = TrainingRun.objects.filter(user_id=user_id)
    if status:
        qs = qs.filter(status=status)
    runs = qs.order_by('-created_at')[:limit]
    return [
        {
            'id': str(r.id),
            'name': r.name,
            'status': r.status,
            'model': r.config.get('model_name', ''),
            'task': r.config.get('task_name', ''),
            'metrics': r.final_metrics or {},
            'created_at': r.created_at.isoformat(),
        }
        for r in runs
    ]


@tool
def get_user_notes(user_id: int, limit: int = 5):
    """Get recent research notes for a user."""
    notes = ResearcherNote.objects.filter(
        author_id=user_id, deleted_at__isnull=True
    ).order_by('-created_at')[:limit]
    return [
        {
            'id': str(n.id),
            'title': n.title,
            'content_preview': (n.content or '')[:300],
            'tags': n.tags,
            'created_at': n.created_at.isoformat(),
        }
        for n in notes
    ]


@tool
def get_user_papers(user_id: int, limit: int = 5):
    """Get recently saved papers for a user."""
    papers = ResearchPaper.objects.filter(
        saved_by_id=user_id
    ).order_by('-created_at')[:limit]
    return [
        {
            'id': str(p.id),
            'title': p.title,
            'abstract': (p.abstract or '')[:300],
            'tags': p.tags,
            'year': p.year,
            'citation_count': p.citation_count,
        }
        for p in papers
    ]


@tool
def get_user_reputation(user_id: int):
    """Get the reputation score for a user."""
    score = ReputationScore.objects.filter(user_id=user_id).first()
    if not score:
        return {'total_score': 0, 'percentile': 0}
    return {
        'total_score': score.total_score,
        'percentile': score.percentile,
        'citations_score': score.citations_score,
        'recommendations_score': score.recommendations_score,
    }


@tool
def get_user_collaborator_matches(user_id: int, limit: int = 3):
    """Get top collaborator matches for a user."""
    matches = CollaboratorMatch.objects.filter(
        user_id=user_id, status='suggested'
    ).order_by('-score')[:limit]
    return [
        {
            'id': str(m.id),
            'matched_user': m.matched_user.username,
            'score': m.score,
            'common_interests': m.common_interests,
        }
        for m in matches
    ]


ALL_TOOLS = [
    get_user_experiments,
    get_user_notes,
    get_user_papers,
    get_user_reputation,
    get_user_collaborator_matches,
]