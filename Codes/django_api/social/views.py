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
    Connection,
    Discussion,
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
    target = get_object_or_404(User, id=user_id)
    if target == request.user:
        return Response({"error": "Cannot follow yourself"}, status=400)
    conn, created = Connection.objects.get_or_create(
        from_user=request.user,
        to_user=target,
        defaults={"connection_type": "FOLLOW", "status": "ACCEPTED"},
    )
    if created:
        ActivityFeed.objects.create(
            user=request.user,
            activity_type="CONNECTION",
            title=f"Started following {target.username}",
        )
    return Response({"status": "following"}, status=201 if created else 200)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def unfollow_user(request, user_id):
    target = get_object_or_404(User, id=user_id)
    Connection.objects.filter(
        from_user=request.user, to_user=target, connection_type="FOLLOW"
    ).delete()
    return Response({"status": "unfollowed"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def followers_list(request, user_id):
    user = get_object_or_404(User, id=user_id)
    conns = Connection.objects.filter(
        to_user=user, connection_type="FOLLOW", status="ACCEPTED"
    ).select_related("from_user")
    return Response(ConnectionSerializer(conns, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def following_list(request, user_id):
    user = get_object_or_404(User, id=user_id)
    conns = Connection.objects.filter(
        from_user=user, connection_type="FOLLOW", status="ACCEPTED"
    ).select_related("to_user")
    return Response(ConnectionSerializer(conns, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Activity Feed
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def activity_feed(request):
    following_ids = request.user.following_connections.filter(
        connection_type="FOLLOW", status="ACCEPTED"
    ).values_list("to_user_id", flat=True)
    user_ids = list(following_ids) + [request.user.id]
    qs = ActivityFeed.objects.filter(user_id__in=user_ids).order_by("-created_at")
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
    if request.method == "GET":
        qs = ResearchGroup.objects.annotate(mc=Count("members")).order_by("-mc")
        if not request.user.is_staff:
            qs = qs.filter(Q(is_private=False) | Q(members=request.user)).distinct()
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ResearchGroupSerializer(page, many=True, context={"request": request}).data
        )
    elif request.method == "POST":
        serializer = ResearchGroupDetailSerializer(data=request.data)
        if serializer.is_valid():
            group = serializer.save(created_by=request.user)
            GroupMembership.objects.create(user=request.user, group=group, role="OWNER")
            group.member_count = 1
            group.save()
            ActivityFeed.objects.create(
                user=request.user,
                activity_type="GROUP_JOINED",
                title=f"Created group '{group.name}'",
            )
            return Response(
                ResearchGroupDetailSerializer(group, context={"request": request}).data,
                status=201,
            )
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def group_detail(request, group_id):
    group = get_object_or_404(ResearchGroup, id=group_id)
    if request.method == "GET":
        return Response(
            ResearchGroupDetailSerializer(group, context={"request": request}).data
        )
    elif request.method == "PATCH":
        membership = GroupMembership.objects.filter(
            user=request.user, group=group, role__in=["OWNER", "MODERATOR"]
        ).first()
        if not membership:
            return Response({"error": "Not authorized"}, status=403)
        serializer = ResearchGroupDetailSerializer(
            group, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        membership = GroupMembership.objects.filter(
            user=request.user, group=group, role="OWNER"
        ).first()
        if not membership:
            return Response({"error": "Only owner can delete"}, status=403)
        group.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def join_group(request, group_id):
    group = get_object_or_404(ResearchGroup, id=group_id)
    if group.is_private and request.user != group.created_by:
        return Response({"error": "Group is private"}, status=403)
    membership, created = GroupMembership.objects.get_or_create(
        user=request.user, group=group
    )
    if created:
        group.member_count = group.groupmembership_set.count()
        group.save()
        ActivityFeed.objects.create(
            user=request.user,
            activity_type="GROUP_JOINED",
            title=f"Joined group '{group.name}'",
        )
    return Response({"status": "joined"}, status=201 if created else 200)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def leave_group(request, group_id):
    group = get_object_or_404(ResearchGroup, id=group_id)
    membership = GroupMembership.objects.filter(user=request.user, group=group).first()
    if not membership:
        return Response({"error": "Not a member"}, status=400)
    if membership.role == "OWNER":
        return Response({"error": "Owner must transfer ownership first"}, status=400)
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
    group = get_object_or_404(ResearchGroup, id=group_id)
    if request.method == "GET":
        qs = group.discussions.annotate(rc=Count("replies")).order_by(
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
            disc = serializer.save(group=group, author=request.user)
            group.discussion_count = group.discussions.count()
            group.save()
            ActivityFeed.objects.create(
                user=request.user,
                activity_type="DISCUSSION_STARTED",
                title=f"Started discussion '{disc.title}' in {group.name}",
            )
            return Response(DiscussionDetailSerializer(disc).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def discussion_detail(request, discussion_id):
    disc = get_object_or_404(Discussion, id=discussion_id)
    if request.method == "GET":
        disc.view_count += 1
        disc.save(update_fields=["view_count"])
        return Response(DiscussionDetailSerializer(disc).data)
    elif request.method == "PATCH":
        if disc.author != request.user:
            return Response({"error": "Not allowed"}, status=403)
        serializer = DiscussionDetailSerializer(disc, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if disc.author != request.user:
            return Response({"error": "Not allowed"}, status=403)
        disc.delete()
        disc.group.discussion_count = disc.group.discussions.count()
        disc.group.save()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def reply_to_discussion(request, discussion_id):
    disc = get_object_or_404(Discussion, id=discussion_id)
    if disc.is_closed:
        return Response({"error": "Discussion is closed"}, status=400)
    serializer = DiscussionReplySerializer(data=request.data)
    if serializer.is_valid():
        reply = serializer.save(discussion=disc, author=request.user)
        disc.reply_count = disc.replies.count()
        disc.save(update_fields=["reply_count", "updated_at"])
        return Response(DiscussionReplySerializer(reply).data, status=201)
    return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# Alerts
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_alerts(request):
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
    alert = get_object_or_404(PublicationAlert, id=alert_id, follower=request.user)
    alert.is_read = True
    alert.save(update_fields=["is_read"])
    return Response({"status": "read"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_all_alerts_read(request):
    PublicationAlert.objects.filter(follower=request.user, is_read=False).update(
        is_read=True
    )
    return Response({"status": "all read"})


# ═══════════════════════════════════════════════════════════════
# Suggested Researchers
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def suggested_researchers(request):
    from agents.models import ResearchPaper

    user_tags = set()
    for paper in ResearchPaper.objects.filter(saved_by=request.user):
        if paper.tags:
            user_tags.update(paper.tags)
    if not user_tags:
        return Response([])
    suggested = (
        User.objects.filter(saved_papers__tags__overlap=list(user_tags))
        .exclude(id=request.user.id)
        .exclude(
            follower_connections__from_user=request.user,
            follower_connections__connection_type="FOLLOW",
        )
        .annotate(
            shared=Count(
                "saved_papers", filter=Q(saved_papers__tags__overlap=list(user_tags))
            )
        )
        .order_by("-shared")
        .distinct()[:10]
    )
    return Response(
        [
            {"id": u.id, "username": u.username, "shared_interests": u.shared}
            for u in suggested
        ]
    )
