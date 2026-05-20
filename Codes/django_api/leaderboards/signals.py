# leaderboards/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Submission, Badge, UserBadge, BenchmarkFollow
from experiment.models import Notification
from django.contrib.auth import get_user_model

User = get_user_model()

@receiver(post_save, sender=Submission)
def on_submission_completed(sender, instance, **kwargs):
    """Notify followers and award badges when a submission completes."""
    if instance.status == 'completed':
        # Notify followers of the benchmark
        for follow in instance.benchmark.followers.select_related('user').all():
            Notification.objects.create(
                recipient=follow.user,
                sender=instance.user,
                type='system',
                content=f"New submission on '{instance.benchmark.name}': {instance.model_name}",
            )
        # Award badges
        from leaderboards.views import _check_and_award_badges
        _check_and_award_badges(instance.benchmark, instance.user, instance)