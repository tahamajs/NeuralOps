"""Auto-award points and check badge criteria on platform events."""
import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils import timezone
from .models import (
    ContributionEvent, UserStreak, UserBadge, Badge,
    AchievementUnlock, PremiumFeature
)

logger = logging.getLogger(__name__)


def _award_points(user, event_type, description='', object_id=''):
    """Utility to award points and update streak."""
    event = ContributionEvent.objects.create(
        user=user,
        event_type=event_type,
        description=description,
        related_object_id=str(object_id) if object_id else ''
    )

    # Update streak
    today = timezone.now().date()
    streak, _ = UserStreak.objects.get_or_create(user=user)
    if streak.last_active_date == today:
        pass  # already recorded today
    elif streak.last_active_date and (today - streak.last_active_date).days == 1:
        streak.current_streak += 1
    else:
        streak.current_streak = 1
    streak.last_active_date = today
    if streak.current_streak > streak.longest_streak:
        streak.longest_streak = streak.current_streak
    streak.total_points += event.points
    streak.weekly_points += event.points
    streak.save()

    # Check badge criteria
    _check_badges(user)

    # Check achievement unlocks
    _check_achievements(user)

    return event


def _check_badges(user):
    """Check if user qualifies for any new badges."""
    total_points = ContributionEvent.objects.filter(user=user).aggregate(
        total=models.Sum('points')
    )['total'] or 0
    papers_saved = user.saved_papers.count() if hasattr(user, 'saved_papers') else 0
    notes_created = user.notes.count() if hasattr(user, 'notes') else 0
    streak = UserStreak.objects.get(user=user) if hasattr(user, 'streak') else None
    reviews_count = ContributionEvent.objects.filter(
        user=user, event_type='review'
    ).count()
    answers_accepted = ContributionEvent.objects.filter(
        user=user, event_type='answer_accepted'
    ).count()

    badges = Badge.objects.all()
    for badge in badges:
        criteria = badge.criteria or {}
        if (
            ('total_points' in criteria and total_points >= criteria['total_points']) or
            ('papers_saved' in criteria and papers_saved >= criteria['papers_saved']) or
            ('notes_created' in criteria and notes_created >= criteria['notes_created']) or
            ('days_streak' in criteria and (streak and streak.longest_streak >= criteria['days_streak'])) or
            ('reviews' in criteria and reviews_count >= criteria['reviews']) or
            ('answers_accepted' in criteria and answers_accepted >= criteria['answers_accepted'])
        ):
            UserBadge.objects.get_or_create(user=user, badge=badge)


def _check_achievements(user):
    """Check if user qualifies for any premium feature unlocks."""
    total_points = ContributionEvent.objects.filter(user=user).aggregate(
        total=models.Sum('points')
    )['total'] or 0
    features = PremiumFeature.objects.filter(is_active=True)
    for feature in features:
        if total_points >= feature.required_points:
            # Check required badges
            if feature.required_badges.exists():
                user_badge_ids = UserBadge.objects.filter(user=user).values_list('badge_id', flat=True)
                if not all(b.id in user_badge_ids for b in feature.required_badges.all()):
                    continue
            AchievementUnlock.objects.get_or_create(
                user=user,
                achievement=feature.name,
                defaults={'required_points': feature.required_points}
            )


# ── Signal receivers for various platform events ──────────────────

@receiver(post_save, sender='agents.ResearchPaper')
def on_paper_saved(sender, instance, created, **kwargs):
    if created and instance.saved_by:
        _award_points(instance.saved_by, 'save_paper', f"Saved paper: {instance.title}", instance.id)


@receiver(post_save, sender='notes.ResearcherNote')
def on_note_created(sender, instance, created, **kwargs):
    if created and instance.author:
        _award_points(instance.author, 'note_create', f"Created note: {instance.title or 'Untitled'}", instance.id)


@receiver(post_save, sender='leaderboards.Submission')
def on_benchmark_submitted(sender, instance, created, **kwargs):
    if created and instance.user:
        _award_points(instance.user, 'benchmark_submit', f"Benchmark submission", instance.id)


@receiver(post_save, sender='community.Answer')
def on_answer_accepted(sender, instance, **kwargs):
    if instance.is_accepted and instance.author:
        _award_points(instance.author, 'answer_accepted', f"Answer accepted", instance.id)


@receiver(post_save, sender='community.Vote')
def on_upvote_received(sender, instance, created, **kwargs):
    if created and instance.direction == 'up':
        if instance.question:
            user = instance.question.author
        elif instance.answer:
            user = instance.answer.author
        else:
            return
        _award_points(user, 'upvote_received', f"Upvote received", instance.id)


@receiver(post_save, sender='social_network.Follow')
def on_follow_gained(sender, instance, created, **kwargs):
    if created:
        _award_points(instance.following, 'follow_gained', f"New follower: {instance.follower.username}", instance.id)