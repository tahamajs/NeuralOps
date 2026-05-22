"""
Auto-create activity feed items and alerts when researchers publish, save papers, etc.
"""
import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import ActivityFeed, PublicationAlert, Follow

logger = logging.getLogger(__name__)


@receiver(post_save, sender='agents.ResearchPaper')
def on_paper_saved(sender, instance, created, **kwargs):
    """When a paper is saved, create an activity feed item and alert followers."""
    if created and instance.saved_by:
        ActivityFeed.objects.create(
            user=instance.saved_by,
            activity_type='paper_saved',
            title=f"Saved paper: {instance.title}",
            description=instance.abstract[:200] if instance.abstract else '',
            link=instance.url or '',
            metadata={'paper_id': str(instance.id)},
        )

        # Alert followers
        followers = Follow.objects.filter(following=instance.saved_by).select_related('follower')
        for follow in followers:
            PublicationAlert.objects.create(
                user=instance.saved_by,
                follower=follow.follower,
                alert_type='paper_saved',
                content=f"{instance.saved_by.username} saved a new paper: {instance.title}",
                link=instance.url or '',
            )


@receiver(post_save, sender='experiment.TrainingRun')
def on_experiment_completed(sender, instance, **kwargs):
    """When a training run completes, create an activity feed item."""
    if instance.status == 'completed' and hasattr(instance, 'user'):
        ActivityFeed.objects.create(
            user=instance.user,
            activity_type='experiment_completed',
            title=f"Experiment completed: {instance.name or instance.id}",
            metadata={'run_id': str(instance.id)},
        )

        # Alert followers
        followers = Follow.objects.filter(following=instance.user).select_related('follower')
        for follow in followers:
            PublicationAlert.objects.create(
                user=instance.user,
                follower=follow.follower,
                alert_type='experiment_completed',
                content=f"{instance.user.username} completed an experiment: {instance.name or instance.id}",
            )


@receiver(post_save, sender='notes.ResearcherNote')
def on_note_created(sender, instance, created, **kwargs):
    """When a note is created, create an activity feed item."""
    if created and instance.is_public:
        ActivityFeed.objects.create(
            user=instance.author,
            activity_type='note_created',
            title=f"Created note: {instance.title or 'Untitled'}",
            metadata={'note_id': str(instance.id)},
        )