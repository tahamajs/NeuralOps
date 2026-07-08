"""Celery tasks for streaks, leaderboard snapshots, and weekly resets."""

import logging
from datetime import timedelta

from celery import shared_task
from django.utils import timezone

from .models import ContributionEvent, LeaderboardSnapshot, UserStreak

logger = logging.getLogger(__name__)


@shared_task
def reset_weekly_points():
    """Reset weekly points for all streaks (run every Monday)."""
    UserStreak.objects.all().update(weekly_points=0)


@shared_task
def check_streak_decay():
    """Decay streaks for users who haven't been active today."""
    today = timezone.now().date()
    yesterday = today - timedelta(days=1)
    streaks = UserStreak.objects.filter(last_active_date__lt=yesterday).exclude(
        current_streak=0
    )
    for streak in streaks:
        streak.current_streak = 0
        streak.save(update_fields=["current_streak"])


@shared_task
def take_leaderboard_snapshot():
    """Take a snapshot of the current leaderboard."""
    top_users = UserStreak.objects.select_related("user").order_by("-total_points")[
        :100
    ]
    rankings = []
    for i, streak in enumerate(top_users, 1):
        rankings.append(
            {
                "user_id": streak.user.id,
                "username": streak.user.username,
                "points": streak.total_points,
                "rank": i,
            }
        )
    LeaderboardSnapshot.objects.create(rankings=rankings)


@shared_task
def award_streak_bonus():
    """Award bonus points for current streaks ≥ 7 days."""
    streaks = UserStreak.objects.filter(current_streak__gte=7)
    for streak in streaks:
        ContributionEvent.objects.create(
            user=streak.user,
            event_type="streak_bonus",
            points=5,
            description=f"Streak bonus for {streak.current_streak} days",
        )
