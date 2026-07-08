"""
Community Lab REST API – Research Labs, Agents, Chatrooms, Workflows, Discoveries.
"""

import logging
import secrets
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)
User = get_user_model()


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# RESEARCH LABS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def lab_list(request):
    if request.method == "GET":
        qs = (
            ResearchLab.objects.filter(
                Q(creator=request.user)
                | Q(human_collaborators=request.user)
                | Q(visibility="public")
            )
            .distinct()
            .order_by("-created_at")
        )

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ResearchLabListSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ResearchLabDetailSerializer(data=request.data)
        if serializer.is_valid():
            lab = serializer.save(creator=request.user)

            # Auto-create default agent roles from the prompt
            default_roles = [
                (
                    "Literature Reviewer",
                    "researcher",
                    "You are a literature reviewer. Search for relevant papers and summarize findings.",
                ),
                (
                    "Experiment Designer",
                    "experimenter",
                    "You are an experiment designer. Configure and launch ML experiments.",
                ),
                (
                    "Critic",
                    "critic",
                    "You are a critic. Review hypotheses and results, pointing out flaws and suggesting improvements.",
                ),
                (
                    "Coordinator",
                    "coordinator",
                    "You are a lab coordinator. Manage the research cycle and delegate tasks to other agents.",
                ),
            ]
            for name, role_type, sys_prompt in default_roles:
                AgentRole.objects.create(
                    lab=lab,
                    name=name,
                    role_type=role_type,
                    description=sys_prompt,
                    system_prompt=sys_prompt,
                )

            return Response(ResearchLabDetailSerializer(lab).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def lab_detail(request, lab_id):
    lab = get_object_or_404(ResearchLab, id=lab_id)
    if request.method == "GET":
        return Response(ResearchLabDetailSerializer(lab).data)

    elif request.method == "PATCH":
        if lab.creator != request.user:
            return Response({"error": "Only the creator can edit"}, status=403)
        serializer = ResearchLabDetailSerializer(lab, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        if lab.creator != request.user:
            return Response({"error": "Only the creator can delete"}, status=403)
        lab.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# AGENT ROLES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def agent_role_list(request, lab_id):
    lab = get_object_or_404(ResearchLab, id=lab_id)
    if request.method == "GET":
        roles = lab.agent_roles.all()
        return Response(AgentRoleSerializer(roles, many=True).data)

    elif request.method == "POST":
        serializer = AgentRoleSerializer(data=request.data)
        if serializer.is_valid():
            role = serializer.save(lab=lab)
            return Response(AgentRoleSerializer(role).data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# CHATROOMS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def chatroom_list(request, lab_id):
    lab = get_object_or_404(ResearchLab, id=lab_id)
    if request.method == "GET":
        rooms = lab.chatrooms.filter(
            Q(is_private=False) | Q(members=request.user)
        ).distinct()
        return Response(
            ChatRoomSerializer(rooms, many=True, context={"request": request}).data
        )

    elif request.method == "POST":
        room = ChatRoom.objects.create(
            lab=lab,
            name=request.data.get("name", "General"),
            description=request.data.get("description", ""),
            created_by=request.user,
            is_private=request.data.get("is_private", False),
        )
        room.members.add(request.user)
        return Response(
            ChatRoomSerializer(room, context={"request": request}).data, status=201
        )


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def chatroom_messages(request, lab_id, room_id):
    room = get_object_or_404(ChatRoom, id=room_id, lab_id=lab_id)
    if request.method == "GET":
        msgs = room.messages.order_by("created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(msgs, request)
        return paginator.get_paginated_response(
            ChatMessageSerializer(page, many=True).data
        )

    elif request.method == "POST":
        msg = ChatMessage.objects.create(
            room=room,
            sender_type="human",
            sender_user=request.user,
            content=request.data.get("content", ""),
        )
        # Broadcast via Channels
        try:
            from asgiref.sync import async_to_sync
            from channels.layers import get_channel_layer

            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f"chat_{room_id}",
                {"type": "chat_message", "message": ChatMessageSerializer(msg).data},
            )
        except Exception:
            pass
        return Response(ChatMessageSerializer(msg).data, status=201)


# ═══════════════════════════════════════════════════════════════
# WORKFLOWS (Visual Designer)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def workflow_list(request, lab_id):
    lab = get_object_or_404(ResearchLab, id=lab_id)
    if request.method == "GET":
        workflows = lab.workflows.all()
        if request.query_params.get("templates"):
            workflows = WorkflowGraph.objects.filter(is_template=True)
        return Response(WorkflowGraphSerializer(workflows, many=True).data)

    elif request.method == "POST":
        serializer = WorkflowGraphSerializer(data=request.data)
        if serializer.is_valid():
            wf = serializer.save(lab=lab, creator=request.user)
            return Response(WorkflowGraphSerializer(wf).data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# DISCOVERIES & PEER REVIEW
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def discovery_list(request, lab_id=None):
    if request.method == "GET":
        qs = DiscoverySubmission.objects.all()
        if lab_id:
            qs = qs.filter(lab_id=lab_id)
        # Public listing only shows non-draft
        if lab_id is None:
            qs = qs.exclude(status="draft")

        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        qs = qs.order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            DiscoverySubmissionSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = DiscoverySubmissionSerializer(data=request.data)
        if serializer.is_valid():
            sub = serializer.save(submitted_by=request.user)
            return Response(DiscoverySubmissionSerializer(sub).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_review(request, submission_id):
    submission = get_object_or_404(DiscoverySubmission, id=submission_id)
    serializer = DiscoveryReviewSerializer(data=request.data)
    if serializer.is_valid():
        review = serializer.save(submission=submission, reviewer=request.user)
        # Update the submission's review count
        submission.comment_count = submission.reviews.count()
        submission.save(update_fields=["comment_count"])
        return Response(DiscoveryReviewSerializer(review).data, status=201)
    return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# INVITATIONS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def invite_to_lab(request, lab_id):
    lab = get_object_or_404(ResearchLab, id=lab_id)
    if lab.creator != request.user:
        return Response({"error": "Only the creator can invite"}, status=403)

    email = request.data.get("email")
    if not email:
        return Response({"error": "email required"}, status=400)

    token = secrets.token_urlsafe(32)
    invitation = LabInvitation.objects.create(
        lab=lab,
        email=email,
        invited_by=request.user,
        token=token,
        expires_at=timezone.now() + timedelta(days=7),
    )
    # TODO: send email with invitation link
    return Response(LabInvitationSerializer(invitation).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_invitations(request):
    invs = LabInvitation.objects.filter(
        email=request.user.email, status="pending", expires_at__gt=timezone.now()
    )
    return Response(LabInvitationSerializer(invs, many=True).data)
