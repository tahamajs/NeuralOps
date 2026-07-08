# leaderboards/signals.py
import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Submission

logger = logging.getLogger(__name__)

# Try to import Notification – it may not exist in experiment.models
try:
    from experiment.models import Notification
except ImportError:
    Notification = None


@receiver(post_save, sender=Submission)
def on_submission_completed(sender, instance, **kwargs):
    """Notify followers and award badges when a submission completes."""
    if instance.status == "completed" and Notification:
        # Notify followers of the benchmark
        for follow in instance.benchmark.followers.select_related("user").all():
            try:
                Notification.objects.create(
                    recipient=follow.user,
                    sender=instance.user,
                    type="system",
                    content=f"New submission on '{instance.benchmark.name}': {instance.model_name}",
                )
            except Exception as e:
                logger.warning("Failed to create notification: %s", e)

    # Award badges (this part doesn't need Notification)
    from leaderboards.views import _check_and_award_badges

    _check_and_award_badges(instance.benchmark, instance.user, instance)
