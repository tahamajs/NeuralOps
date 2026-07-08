"""
Auto-update interest profiles when papers are saved.
"""

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .tasks import update_user_embedding


@receiver(post_save, sender="agents.ResearchPaper")
def on_paper_saved(sender, instance, created, **kwargs):
    """When a user saves a paper, update their interest profile."""
    if instance.saved_by:
        # Use .delay to run asynchronously
        update_user_embedding.delay(instance.saved_by_id)


@receiver(post_delete, sender="agents.ResearchPaper")
def on_paper_deleted(sender, instance, **kwargs):
    """When a paper is removed, update the user's interest profile."""
    if instance.saved_by:
        update_user_embedding.delay(instance.saved_by_id)
