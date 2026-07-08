"""
Celery tasks for streaks and weekly summaries.
"""

from datetime import date, timedelta

from celery import shared_task
from django.contrib.auth import get_user_model
from django.db.models import Sum

User = get_user_model()


@shared_task
def generate_weekly_summaries():
    """Generate weekly summaries for all active users for the previous week."""
    from .models import DailyActivity, WeeklySummary

    today = date.today()
    # Last week (Monday to Sunday)
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)

    users = User.objects.filter(is_active=True)
    count = 0

    for user in users:
        activities = DailyActivity.objects.filter(
            user=user, date__gte=last_monday, date__lte=last_sunday
        )
        if not activities.exists():
            continue

        # Aggregate totals
        totals = activities.aggregate(
            paper_reads=Sum("paper_reads"),
            experiments=Sum("experiments_run"),
            notes=Sum("notes_created"),
            papers=Sum("papers_written"),
        )

        total = sum(v or 0 for v in totals.values())
        most_active = activities.order_by("-total_count").first()

        summary_text = (
            f"You had {total} research activities this week. "
            f"Top activity: {_top_activity(activities)}. Keep up the great work!"
        )

        _, created = WeeklySummary.objects.update_or_create(
            user=user,
            week_start=last_monday,
            defaults={
                "total_activities": total,
                "paper_reads": totals["paper_reads"] or 0,
                "experiments_run": totals["experiments"] or 0,
                "notes_created": totals["notes"] or 0,
                "papers_written": totals["papers"] or 0,
                "most_active_day": most_active.date if most_active else None,
                "summary_text": summary_text,
            },
        )
        if created:
            count += 1

    return f"Generated {count} weekly summaries"


def _top_activity(activities):
    """Find the most frequent activity type in the queryset."""
    types = {
        "Paper reading": "paper_reads",
        "Experiment running": "experiments_run",
        "Note taking": "notes_created",
        "Paper writing": "papers_written",
    }
    best = None
    best_val = 0
    for label, field in types.items():
        total = activities.aggregate(s=Sum(field))["s"] or 0
        if total > best_val:
            best_val = total
            best = label
    return best or "Research"


@shared_task
def check_and_reset_broken_streaks():
    """
    Check all users and reset streaks if they haven't been active for more than 1 day.
    This is a safety net – streaks are normally updated by DailyActivity.record().
    """
    from .models import Streak

    today = date.today()
    for streak in Streak.objects.filter(current_streak__gt=0):
        if streak.last_active_date:
            delta = (today - streak.last_active_date).days
            if delta > 1:
                # Streak is broken
                streak.current_streak = 0
                streak.save(update_fields=["current_streak"])
