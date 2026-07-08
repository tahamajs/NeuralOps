"""Celery task to run the AI Co‑Scientist workflow asynchronously."""

from celery import shared_task

from .engine import AIScientistFramework
from .models import ResearchSession


@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def run_research_workflow(self, session_id: str):
    """Run the research workflow for a session."""
    try:
        session = ResearchSession.objects.get(id=session_id)
    except ResearchSession.DoesNotExist:
        return {"error": "Session not found"}

    framework = AIScientistFramework(
        session=session,
        model_name=session.model_name,
        max_iterations=session.max_iterations,
        hypotheses_per_generation=session.hypotheses_per_generation,
        tournament_size=session.tournament_size,
        evolution_top_k=session.evolution_top_k,
    )

    return framework.run_research_workflow()
