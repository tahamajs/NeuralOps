"""
Auto-update reputation when papers are saved, read, recommended, or cited.
"""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import ReputationEvent

logger = logging.getLogger(__name__)


@receiver(post_save, sender="agents.ResearchPaper")
def on_paper_saved(sender, instance, created, **kwargs):
    """When a paper is saved by a user, this is a 'full_text_read' event."""
    if created:
        # For each user who saved this paper, record a full_text_read event
        # (This is a simplification – in reality you'd have a separate 'read' action)
        pass  # The actual event is created when the paper is explicitly read


def record_read(user, paper, read_type="full_text_read"):
    """Utility to record a read event."""
    event = ReputationEvent.objects.create(
        user=paper.saved_by if hasattr(paper, "saved_by") else user,
        event_type=read_type,
        paper=paper,
        source_user=user,
        description=f"Paper read by {user.username}",
    )
    return event


def record_recommendation(user, paper, source_user):
    """Utility to record a recommendation event."""
    event = ReputationEvent.objects.create(
        user=user,
        event_type="recommendation",
        paper=paper,
        source_user=source_user,
        description=f"Recommended by {source_user.username}",
    )
    return event


def record_citation(user, paper):
    """Utility to record a citation event."""
    event = ReputationEvent.objects.create(
        user=user,
        event_type="citation",
        paper=paper,
        description="Paper cited",
    )
    return event
