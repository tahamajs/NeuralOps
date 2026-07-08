from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from .models import *


def update_activity_log(user, field):
    """Increment a specific activity field for today."""
    today = timezone.now().date()
    log, _ = ActivityLog.objects.get_or_create(user=user, date=today)
    setattr(log, field, getattr(log, field) + 1)
    log.save()


@receiver(post_save, sender=Answer)
def on_answer_saved(sender, instance, created, **kwargs):
    if created:
        _check_badges(instance.author)
        _update_trust_level(instance.author)


@receiver(post_save, sender=Question)
def on_question_saved(sender, instance, created, **kwargs):
    if created:
        _check_badges(instance.author)
        _update_trust_level(instance.author)


@receiver(post_save, sender=Vote)
def on_vote_saved(sender, instance, created, **kwargs):
    if created and instance.answer:
        _check_badges(instance.answer.author)


def _check_badges(user):
    total_rep = user.reputation_events.aggregate(s=models.Sum("amount"))["s"] or 0
    answer_count = user.community_answers.count()
    question_count = user.community_questions.count()

    badge_conditions = {
        ("First Question", "bronze"): lambda: question_count >= 1,
        ("First Answer", "bronze"): lambda: answer_count >= 1,
        ("Scholar", "bronze"): lambda: user.community_answers.filter(
            is_accepted=True
        ).exists(),
        ("Popular Question", "bronze"): lambda: user.community_questions.filter(
            score__gte=5
        ).exists(),
        ("Good Answer", "bronze"): lambda: user.community_answers.filter(
            score__gte=10
        ).exists(),
        ("Great Answer", "silver"): lambda: user.community_answers.filter(
            score__gte=25
        ).exists(),
        ("Centurion", "silver"): lambda: total_rep >= 1000,
        ("Millennium", "gold"): lambda: total_rep >= 10000,
    }
    for badge_name, tier in badge_conditions:
        try:
            badge = Badge.objects.get(name=badge_name, tier=tier)
            if badge_conditions[(badge_name, tier)]():
                UserBadge.objects.get_or_create(user=user, badge=badge)
        except Badge.DoesNotExist:
            pass


def _update_trust_level(user):
    tl, _ = TrustLevel.objects.get_or_create(user=user)
    days_active = ActivityLog.objects.filter(user=user).count()
    posts_read = user.community_questions.count() + user.community_answers.count()
    if tl.level < 1 and days_active >= 3 and posts_read >= 5:
        tl.level = 1
        tl.save()
    elif tl.level < 2 and days_active >= 15 and posts_read >= 20:
        tl.level = 2
        tl.save()
    elif tl.level < 3 and days_active >= 30:
        total_rep = user.reputation_events.aggregate(s=models.Sum("amount"))["s"] or 0
        if total_rep >= 200:
            tl.level = 3
            tl.save()
