from agents.models import ResearchPaper
from django.contrib.auth import get_user_model
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
from jobscheduler.models import Job
from leaderboards.models import Submission
from reviews.models import Review

from .models import ActivityEvent, Portfolio, ShowcaseItem

User = get_user_model()


@receiver(post_save, sender=Job)
def on_training_complete(sender, instance, **kwargs):
    if instance.status == "completed":
        # Activity
        ActivityEvent.objects.get_or_create(
            user=instance.user,
            date=timezone.now().date(),
            event_type="training",
            defaults={"count": 1},
        )
        # Auto-add to portfolio if public
        portfolio, _ = Portfolio.objects.get_or_create(user=instance.user)
        if portfolio.is_public:
            ShowcaseItem.objects.get_or_create(
                portfolio=portfolio,
                item_type="model",
                title=instance.name or f"Training run {instance.id}",
                defaults={
                    "description": f"Model: {instance.model_name}, Task: {instance.task_name}",
                    "metadata": {"run_id": str(instance.id)},
                },
            )


@receiver(post_save, sender=ResearchPaper)
def on_paper_saved(sender, instance, created, **kwargs):
    if created:
        ActivityEvent.objects.get_or_create(
            user=instance.saved_by,
            date=timezone.now().date(),
            event_type="paper_saved",
            defaults={"count": 1},
        )
        portfolio, _ = Portfolio.objects.get_or_create(user=instance.saved_by)
        if portfolio.is_public:
            ShowcaseItem.objects.get_or_create(
                portfolio=portfolio,
                item_type="paper",
                title=instance.title,
                defaults={
                    "description": instance.abstract or "",
                    "url": instance.url,
                    "metadata": {"paper_id": str(instance.id)},
                },
            )


@receiver(post_save, sender=Submission)
def on_leaderboard_submission(sender, instance, created, **kwargs):
    if created:
        ActivityEvent.objects.get_or_create(
            user=instance.user,
            date=timezone.now().date(),
            event_type="submission",
            defaults={"count": 1},
        )


@receiver(post_save, sender=Review)
def on_review_completed(sender, instance, **kwargs):
    if instance.status == "completed":
        ActivityEvent.objects.get_or_create(
            user=instance.reviewer,
            date=timezone.now().date(),
            event_type="review",
            defaults={"count": 1},
        )
