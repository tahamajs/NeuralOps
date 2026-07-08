"""Auto-generate expert profile when papers are saved."""

from django.db.models.signals import post_save
from django.dispatch import receiver

from .tasks import auto_generate_expert_profile


@receiver(post_save, sender="agents.ResearchPaper")
def on_paper_saved(sender, instance, created, **kwargs):
    if instance.saved_by:
        auto_generate_expert_profile.delay(instance.saved_by_id)
