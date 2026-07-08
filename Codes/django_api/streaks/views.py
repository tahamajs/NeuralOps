import logging
from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import DailyActivity, Streak, WeeklySummary
from .serializers import (DailyActivitySerializer, StreakSerializer,
                          WeeklySummarySerializer)

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 30
    page_size_query_param = "page_size"
    max_page_size = 200


# ═══════════════════════════════════════════════════════════════
# Heatmap Data (GitHub‑style)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def heatmap_data(request, username):
    """
    Return daily activity counts for the last 365 days for a user.
    Returns a list of {date: 'YYYY-MM-DD', count: int}.
    """
    user = get_object_or_404(User, username=username)
    today = date.today()
    year_ago = today - timedelta(days=365)

    activities = (
        DailyActivity.objects.filter(user=user, date__gte=year_ago)
        .values("date", "total_count")
        .order_by("date")
    )

    # Build a dict for all 365 days (some may be zero)
    result = {}
    for d in range(366):
        day = year_ago + timedelta(days=d)
        result[day.isoformat()] = 0

    for activity in activities:
        result[activity["date"].isoformat()] = activity["total_count"]

    # Return as list of {date, count}
    data = [{"date": k, "count": v} for k, v in sorted(result.items())]
    return Response(data)


# ═══════════════════════════════════════════════════════════════
# Streak Info
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def streak_info(request, username):
    """
    Get streak information for a user.
    """
    user = get_object_or_404(User, username=username)
    streak, _ = Streak.objects.get_or_create(user=user)
    return Response(StreakSerializer(streak).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_streak(request):
    """Get the authenticated user's streak info."""
    streak, _ = Streak.objects.get_or_create(user=request.user)
    return Response(StreakSerializer(streak).data)


# ═══════════════════════════════════════════════════════════════
# Weekly Summaries
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_weekly_summaries(request):
    """Get the authenticated user's weekly summaries."""
    qs = WeeklySummary.objects.filter(user=request.user).order_by("-week_start")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        WeeklySummarySerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def weekly_summaries(request, username):
    """Get weekly summaries for any user (public profile)."""
    user = get_object_or_404(User, username=username)
    qs = WeeklySummary.objects.filter(user=user).order_by("-week_start")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        WeeklySummarySerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Leaderboard (Top Streaks)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def streak_leaderboard(request):
    """Return top users by current streak."""
    qs = Streak.objects.select_related("user").order_by("-current_streak")[:50]
    return Response(StreakSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Daily Activity Detail
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_daily_activities(request):
    """Get the authenticated user's daily activities for the last 30 days."""
    today = date.today()
    month_ago = today - timedelta(days=30)
    qs = DailyActivity.objects.filter(user=request.user, date__gte=month_ago).order_by(
        "-date"
    )
    return Response(DailyActivitySerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Generate Weekly Summary (staff / automated)
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_weekly_summary(request):
    """Manually trigger weekly summary generation (staff only)."""
    if not request.user.is_staff:
        return Response({"error": "Staff only"}, status=403)
    from .tasks import generate_weekly_summaries

    count = generate_weekly_summaries.delay()
    return Response({"status": "weekly summary generation queued"})
