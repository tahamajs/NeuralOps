import logging

from django.contrib.auth import get_user_model
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (
    ActivityFeed,
    Discussion,
    Follow,
    GroupMembership,
    PublicationAlert,
    ResearchGroup,
)
from .serializers import *

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Follow / Unfollow
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def follow_user(request, user_id):
    """Follow another researcher."""
    target = get_object_or_404(User, id=user_id)
    if target == request.user:
        return Response({"error": "Cannot follow yourself"}, status=400)
    follow, created = Follow.objects.get_or_create(
        follower=request.user, following=target
    )
    if created:
        ActivityFeed.objects.create(
            user=request.user,
            activity_type="followed",
            title=f"Started following {target.username}",
            metadata={"target_user_id": target.id},
        )
    return Response(
        {"status": "following", "username": target.username},
        status=201 if created else 200,
    )


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def unfollow_user(request, user_id):
    """Unfollow a researcher."""
    target = get_object_or_404(User, id=user_id)
    Follow.objects.filter(follower=request.user, following=target).delete()
    return Response({"status": "unfollowed"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_following(request):
    """List users I follow."""
    follows = Follow.objects.filter(follower=request.user).select_related("following")
    return Response(FollowSerializer(follows, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_followers(request):
    """List users who follow me."""
    followers = Follow.objects.filter(following=request.user).select_related("follower")
    return Response(FollowSerializer(followers, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Activity Feed
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def activity_feed(request):
    """Get the activity feed from followed users."""
    following_ids = request.user.following.values_list("following_id", flat=True)
    # Include own activities
    user_ids = list(following_ids) + [request.user.id]

    qs = ActivityFeed.objects.filter(user_id__in=user_ids).order_by("-created_at")

    activity_type = request.query_params.get("type")
    if activity_type:
        qs = qs.filter(activity_type=activity_type)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ActivityFeedSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Research Groups
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def group_list(request):
    """List all public groups, or create a new one."""
    if request.method == "GET":
        qs = (
            ResearchGroup.objects.filter(visibility="public")
            .annotate(
                member_count=Count("members"), discussion_count=Count("discussions")
            )
            .order_by("-member_count")
        )

        search = request.query_params.get("q")
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ResearchGroupSerializer(page, many=True, context={"request": request}).data
        )

    elif request.method == "POST":
        serializer = ResearchGroupDetailSerializer(data=request.data)
        if serializer.is_valid():
            group = serializer.save(created_by=request.user)
            GroupMembership.objects.create(user=request.user, group=group, role="admin")
            group.member_count = 1
            group.save()
            ActivityFeed.objects.create(
                user=request.user,
                activity_type="group_joined",
                title=f"Created group '{group.name}'",
                metadata={"group_id": str(group.id)},
            )
            return Response(
                ResearchGroupDetailSerializer(group, context={"request": request}).data,
                status=201,
            )
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def group_detail(request, group_id):
    """Get, update, or delete a research group."""
    group = get_object_or_404(ResearchGroup, id=group_id)

    if request.method == "GET":
        return Response(
            ResearchGroupDetailSerializer(group, context={"request": request}).data
        )

    elif request.method == "PATCH":
        membership = GroupMembership.objects.filter(
            user=request.user, group=group, role__in=["admin", "moderator"]
        ).first()
        if not membership:
            return Response(
                {"error": "Only admins and moderators can edit"}, status=403
            )
        serializer = ResearchGroupDetailSerializer(
            group, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        membership = GroupMembership.objects.filter(
            user=request.user, group=group, role="admin"
        ).first()
        if not membership:
            return Response({"error": "Only admins can delete"}, status=403)
        group.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def join_group(request, group_id):
    """Join a research group."""
    group = get_object_or_404(ResearchGroup, id=group_id)
    if group.visibility != "public" and group.created_by != request.user:
        return Response({"error": "Group is not open for joining"}, status=403)

    membership, created = GroupMembership.objects.get_or_create(
        user=request.user, group=group, defaults={"role": "member"}
    )
    if created:
        group.member_count = group.groupmembership_set.count()
        group.save()
        ActivityFeed.objects.create(
            user=request.user,
            activity_type="group_joined",
            title=f"Joined group '{group.name}'",
            metadata={"group_id": str(group.id)},
        )
    return Response({"status": "joined"}, status=201 if created else 200)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def leave_group(request, group_id):
    """Leave a research group."""
    group = get_object_or_404(ResearchGroup, id=group_id)
    membership = GroupMembership.objects.filter(user=request.user, group=group).first()
    if not membership:
        return Response({"error": "Not a member"}, status=400)
    if (
        membership.role == "admin"
        and group.groupmembership_set.filter(role="admin").count() <= 1
    ):
        return Response(
            {"error": "Cannot leave as the last admin. Transfer ownership first."},
            status=400,
        )
    membership.delete()
    group.member_count = group.groupmembership_set.count()
    group.save()
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Discussions
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def discussion_list(request, group_id):
    """List discussions in a group, or create a new one."""
    group = get_object_or_404(ResearchGroup, id=group_id)

    if request.method == "GET":
        qs = group.discussions.annotate(reply_count=Count("replies")).order_by(
            "-is_pinned", "-updated_at"
        )

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            DiscussionSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = DiscussionDetailSerializer(data=request.data)
        if serializer.is_valid():
            discussion = serializer.save(group=group, author=request.user)
            group.discussion_count = group.discussions.count()
            group.save()
            ActivityFeed.objects.create(
                user=request.user,
                activity_type="discussion_started",
                title=f"Started discussion '{discussion.title}' in {group.name}",
                metadata={
                    "group_id": str(group.id),
                    "discussion_id": str(discussion.id),
                },
            )
            return Response(DiscussionDetailSerializer(discussion).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def discussion_detail(request, discussion_id):
    """Get, update, or delete a discussion."""
    discussion = get_object_or_404(Discussion, id=discussion_id)
    group = discussion.group

    if request.method == "GET":
        discussion.view_count += 1
        discussion.save(update_fields=["view_count"])
        return Response(DiscussionDetailSerializer(discussion).data)

    elif request.method == "PATCH":
        if (
            discussion.author != request.user
            and not group.groupmembership_set.filter(
                user=request.user, role__in=["admin", "moderator"]
            ).exists()
        ):
            return Response({"error": "Not allowed"}, status=403)
        serializer = DiscussionDetailSerializer(
            discussion, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        if (
            discussion.author != request.user
            and not group.groupmembership_set.filter(
                user=request.user, role="admin"
            ).exists()
        ):
            return Response({"error": "Not allowed"}, status=403)
        discussion.delete()
        group.discussion_count = group.discussions.count()
        group.save()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def reply_to_discussion(request, discussion_id):
    """Reply to a discussion."""
    discussion = get_object_or_404(Discussion, id=discussion_id)
    if discussion.is_closed:
        return Response({"error": "Discussion is closed"}, status=400)

    serializer = DiscussionReplySerializer(data=request.data)
    if serializer.is_valid():
        reply = serializer.save(discussion=discussion, author=request.user)
        discussion.reply_count = discussion.replies.count()
        discussion.save(update_fields=["reply_count", "updated_at"])
        return Response(DiscussionReplySerializer(reply).data, status=201)
    return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# Publication Alerts
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_alerts(request):
    """Get my publication alerts."""
    qs = PublicationAlert.objects.filter(follower=request.user).order_by("-created_at")

    unread_only = request.query_params.get("unread") == "true"
    if unread_only:
        qs = qs.filter(is_read=False)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        PublicationAlertSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_alert_read(request, alert_id):
    """Mark an alert as read."""
    alert = get_object_or_404(PublicationAlert, id=alert_id, follower=request.user)
    alert.is_read = True
    alert.save(update_fields=["is_read"])
    return Response({"status": "read"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_all_alerts_read(request):
    """Mark all alerts as read."""
    PublicationAlert.objects.filter(follower=request.user, is_read=False).update(
        is_read=True
    )
    return Response({"status": "all read"})


# ═══════════════════════════════════════════════════════════════
# Suggested Researchers (based on shared interests)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def suggested_researchers(request):
    """Suggest researchers to follow based on shared interests and papers."""
    from agents.models import ResearchPaper

    # Get the user's saved paper tags and categories
    user_papers = ResearchPaper.objects.filter(saved_by=request.user)
    user_tags = set()
    for paper in user_papers:
        if paper.tags:
            user_tags.update(paper.tags)

    # Find users who saved papers with the same tags
    suggested = (
        User.objects.filter(saved_papers__tags__overlap=list(user_tags))
        .exclude(id=request.user.id)
        .exclude(followers__follower=request.user)  # Exclude already followed
        .annotate(
            shared_papers=Count(
                "saved_papers", filter=Q(saved_papers__tags__overlap=list(user_tags))
            )
        )
        .order_by("-shared_papers")
        .distinct()[:10]
    )

    return Response(
        [
            {
                "id": u.id,
                "username": u.username,
                "shared_interests": getattr(u, "shared_papers", 0),
            }
            for u in suggested
        ]
    )
