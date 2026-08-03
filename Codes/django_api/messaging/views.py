# messaging/views.py
import logging

from django.contrib.auth.models import User
from django.db.models import Max
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import ChatRoom, Message
from .serializers import (
    ChatRoomDetailSerializer,
    ChatRoomListSerializer,
    MessageSerializer,
)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Chat Room List
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def chat_list(request):
    """List all chat rooms for the current user, ordered by latest message."""
    rooms = (
        ChatRoom.objects.filter(participants=request.user)
        .annotate(last_msg_time=Max("messages__created_at"))
        .order_by("-last_msg_time")
    )

    serializer = ChatRoomListSerializer(rooms, many=True, context={"request": request})
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# Start or Get Chat Room
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_chat(request):
    """Create or get an existing chat room with another user."""
    other_user_id = request.data.get("user_id")
    if not other_user_id:
        return Response({"error": "user_id required"}, status=400)

    other_user = get_object_or_404(User, id=other_user_id)

    if other_user == request.user:
        return Response({"error": "Cannot chat with yourself"}, status=400)

    # Find existing room
    room = (
        ChatRoom.objects.filter(participants=request.user)
        .filter(participants=other_user)
        .distinct()
        .first()
    )

    if room:
        serializer = ChatRoomDetailSerializer(room, context={"request": request})
        return Response(serializer.data)

    # Create new room
    room = ChatRoom.objects.create()
    room.participants.add(request.user, other_user)

    serializer = ChatRoomDetailSerializer(room, context={"request": request})
    return Response(serializer.data, status=status.HTTP_201_CREATED)


# ═══════════════════════════════════════════════════════════════
# Chat Room Detail & Messages
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def chat_detail(request, room_id):
    """Get all messages in a chat room."""
    room = get_object_or_404(ChatRoom, id=room_id, participants=request.user)

    # Mark messages as read
    room.messages.filter(read=False).exclude(sender=request.user).update(read=True)

    serializer = ChatRoomDetailSerializer(room, context={"request": request})
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def send_message(request, room_id):
    """Send a message to a chat room via REST (fallback for WebSocket)."""
    room = get_object_or_404(ChatRoom, id=room_id, participants=request.user)

    content = request.data.get("content", "")
    uploaded_file = request.FILES.get("file")

    if not content and not uploaded_file:
        return Response({"error": "content or file required"}, status=400)

    msg = Message.objects.create(
        room=room,
        sender=request.user,
        content=content,
        file=uploaded_file if uploaded_file else None,
        file_name=uploaded_file.name if uploaded_file else "",
    )
    room.save(update_fields=["updated_at"])

    # Notify via Channels (if available)
    try:
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"chat_{room_id}",
            {
                "type": "chat_message",
                "message": MessageSerializer(msg).data,
            },
        )
    except Exception:
        pass

    return Response(MessageSerializer(msg).data, status=status.HTTP_201_CREATED)


# ═══════════════════════════════════════════════════════════════
# Mark Messages as Read
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_read(request, room_id):
    """Mark all messages in a room as read."""
    room = get_object_or_404(ChatRoom, id=room_id, participants=request.user)
    room.messages.filter(read=False).exclude(sender=request.user).update(read=True)
    return Response({"status": "marked read"})


# ═══════════════════════════════════════════════════════════════
# Delete Chat Room
# ═══════════════════════════════════════════════════════════════


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_chat(request, room_id):
    """Delete a chat room (any participant can delete)."""
    room = get_object_or_404(ChatRoom, id=room_id, participants=request.user)
    room.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)
