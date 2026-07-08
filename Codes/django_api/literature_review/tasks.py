# literature_review/tasks.py
import logging

from agents.models import ResearchPaper
from celery import shared_task
from django.contrib.auth import get_user_model

from .models import (DeepReview, LiteratureProject, PaperReviewNote,
                     ResearchNotebookSection, Topic)

logger = logging.getLogger(__name__)
User = get_user_model()


@shared_task
def run_topic_finder(project_id, query):
    """Simulate AI topic finder: generate subtopics, themes, gaps from papers."""
    project = LiteratureProject.objects.get(id=project_id)
    # In production, call an LLM or use SciSpace API. Here we create a placeholder.
    papers = ResearchPaper.objects.filter(title__icontains=query or "deep learning")[
        :10
    ]
    subtopics = [p.title for p in papers]
    Topic.objects.create(
        project=project,
        name=f"Analysis of '{query or 'Deep Learning'}'",
        description="Auto‑generated topic analysis",
        subtopics=[{"title": t, "citations": 5} for t in subtopics],
        key_themes=["Novelty", "Performance", "Scalability"],
        research_gaps=[
            {"description": "Lack of cross‑domain evaluation", "importance": 0.8}
        ],
    )
    project.current_step = 2
    project.progress = 33.0
    project.save()


@shared_task
def generate_notebook_content(project_id):
    """Generate thematic notebook sections based on saved papers."""
    project = LiteratureProject.objects.get(id=project_id)
    # Create sections for each theme
    themes = ["Introduction", "Related Work", "Methodology", "Results", "Discussion"]
    for i, theme in enumerate(themes):
        ResearchNotebookSection.objects.get_or_create(
            project=project,
            title=theme,
            defaults={
                "content": f"### {theme}\n\n(Content to be generated)",
                "order": i,
            },
        )
    project.current_step = 3
    project.progress = 66.0
    project.save()


@shared_task
def run_deep_review(project_id, paper_ids):
    """Analyze 20+ papers and generate paragraphs and notes."""
    project = LiteratureProject.objects.get(id=project_id)
    if not paper_ids:
        paper_ids = list(
            ResearchPaper.objects.filter(saved_by=project.user).values_list(
                "id", flat=True
            )[:20]
        )
    review = DeepReview.objects.create(
        project=project,
        title=f"Deep Review - {project.title}",
        papers_analyzed=len(paper_ids),
    )
    for pid in paper_ids[:20]:
        try:
            paper = ResearchPaper.objects.get(id=pid)
            PaperReviewNote.objects.create(
                deep_review=review,
                paper=paper,
                notes=f"Key findings from {paper.title}",
                key_points=["Point 1", "Point 2"],
            )
        except ResearchPaper.DoesNotExist:
            pass
    review.generated_paragraphs = [
        "## Literature Review Summary\n\nThis review synthesizes findings from the analyzed papers...",
        "## Research Gap\n\nSeveral gaps were identified...",
    ]
    review.ethical_ai_guidelines = True
    review.save()
    project.progress = 100.0
    project.save()
