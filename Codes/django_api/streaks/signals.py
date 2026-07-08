"""
Auto-record daily activity when relevant events happen across the platform.
"""

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import DailyActivity


@receiver(post_save, sender="agents.ResearchPaper")
def on_paper_saved(sender, instance, created, **kwargs):
    if created and instance.saved_by:
        DailyActivity.record(instance.saved_by, "paper_read")


@receiver(post_save, sender="experiment.TrainingRun")
def on_experiment_completed(sender, instance, **kwargs):
    if instance.status == "completed" and hasattr(instance, "user"):
        DailyActivity.record(instance.user, "experiment_run")


@receiver(post_save, sender="notes.ResearcherNote")
def on_note_created(sender, instance, created, **kwargs):
    if created:
        DailyActivity.record(instance.author, "note_created")


@receiver(post_save, sender="paper.PaperFile")
def on_paper_file_saved(sender, instance, created, **kwargs):
    """When a paper file is saved/edited."""
    if instance.project and instance.project.owner:
        DailyActivity.record(instance.project.owner, "paper_written")


@receiver(post_save, sender="leaderboards.Submission")
def on_benchmark_submission(sender, instance, created, **kwargs):
    if created:
        DailyActivity.record(instance.user, "benchmark_submission")


@receiver(post_save, sender="social_network.DiscussionReply")
def on_discussion_reply(sender, instance, created, **kwargs):
    if created:
        DailyActivity.record(instance.author, "discussion_post")
