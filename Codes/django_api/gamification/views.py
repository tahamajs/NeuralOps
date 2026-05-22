import logging
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination

from .models import (
    Badge, UserBadge, ContributionEvent, UserStreak,
    LeaderboardSnapshot, AchievementUnlock, PremiumFeature
)
from .serializers import (
    BadgeSerializer, UserBadgeSerializer, ContributionEventSerializer,
    UserStreakSerializer, LeaderboardSnapshotSerializer,
    AchievementUnlockSerializer, PremiumFeatureSerializer
)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Badges
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def badge_list(request):
    """List all available badges."""
    qs = Badge.objects.all().order_by('tier', 'name')
    return Response(BadgeSerializer(qs, many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_badges(request):
    """Get badges awarded to the current user."""
    badges = UserBadge.objects.filter(user=request.user).select_related('badge')
    return Response(UserBadgeSerializer(badges, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Contribution Events
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_contributions(request):
    """Get the current user's contribution history."""
    qs = ContributionEvent.objects.filter(user=request.user).order_by('-created_at')

    event_type = request.query_params.get('event_type')
    if event_type:
        qs = qs.filter(event_type=event_type)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ContributionEventSerializer(page, many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_points_summary(request):
    """Get a summary of the current user's points."""
    events = ContributionEvent.objects.filter(user=request.user)
    total = events.aggregate(s=models.Sum('points'))['s'] or 0
    streak = UserStreak.objects.get_or_create(user=request.user)[0]
    return Response({
        'total_points': total,
        'current_streak': streak.current_streak,
        'longest_streak': streak.longest_streak,
        'weekly_points': streak.weekly_points,
    })


# ═══════════════════════════════════════════════════════════════
# Streaks
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_streak(request):
    """Get the current user's streak details."""
    streak, _ = UserStreak.objects.get_or_create(user=request.user)
    return Response(UserStreakSerializer(streak).data)


# ═══════════════════════════════════════════════════════════════
# Leaderboard
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def leaderboard(request):
    """Get the global leaderboard (live)."""
    qs = UserStreak.objects.select_related('user').order_by('-total_points')
    field = request.query_params.get('field')
    institution = request.query_params.get('institution')
    # In production, filter by user profile fields (affiliation, research interests)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    data = []
    for i, streak in enumerate(page, 1):
        data.append({
            'rank': i,
            'username': streak.user.username,
            'total_points': streak.total_points,
            'current_streak': streak.current_streak,
            'badges_count': streak.user.gamification_badges.count(),
        })
    return paginator.get_paginated_response(data)


# ═══════════════════════════════════════════════════════════════
# Achievement Unlocks / Premium Features
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_unlocks(request):
    """Get features unlocked by the current user."""
    unlocks = AchievementUnlock.objects.filter(user=request.user)
    return Response(AchievementUnlockSerializer(unlocks, many=True).data)


@api_view(['GET'])
@permission_classes([AllowAny])
def premium_features(request):
    """List all premium features and their requirements."""
    features = PremiumFeature.objects.filter(is_active=True)
    return Response(PremiumFeatureSerializer(features, many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_feature_access(request, feature_name):
    """Check if the current user has access to a premium feature."""
    feature = get_object_or_404(PremiumFeature, name=feature_name, is_active=True)
    unlocked = AchievementUnlock.objects.filter(
        user=request.user, achievement=feature.name
    ).exists()
    return Response({
        'feature': feature.name,
        'unlocked': unlocked,
        'required_points': feature.required_points,
    })