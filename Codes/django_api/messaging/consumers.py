# messaging/consumers.py
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token

from .models import ChatRoom, Message


class ChatConsumer(AsyncJsonWebsocketConsumer):
    """Real-time private chat via WebSocket."""

    async def connect(self):
        self.room_id = self.scope["url_route"]["kwargs"]["room_id"]
        self.room_group_name = f"chat_{self.room_id}"

        # Authenticate via token in query string
        query_string = self.scope.get("query_string", b"").decode()
        token_key = query_string.replace("token=", "")
        user = await self._get_user_from_token(token_key)

        if user is None or isinstance(user, AnonymousUser):
            await self.close(code=4001)
            return

        self.user = user
        self.username = user.username

        # Verify user is a participant of this chat room
        is_member = await self._is_room_member()
        if not is_member:
            await self.close(code=4003)
            return

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

        # Notify others that user is online
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "user_status",
                "username": self.username,
                "status": "online",
            },
        )

    async def disconnect(self, close_code):
        if hasattr(self, "room_group_name"):
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "user_status",
                    "username": getattr(self, "username", "unknown"),
                    "status": "offline",
                },
            )
            await self.channel_layer.group_discard(
                self.room_group_name, self.channel_name
            )

    async def receive_json(self, content):
        msg_type = content.get("type", "message")

        if msg_type == "message":
            text = content.get("content", "")
            if not text.strip():
                return

            # Save to database
            message = await self._save_message(text)

            # Broadcast to group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "chat_message",
                    "message": {
                        "id": str(message.id),
                        "room_id": self.room_id,
                        "sender_id": self.user.id,
                        "sender_username": self.username,
                        "content": text,
                        "created_at": message.created_at.isoformat(),
                    },
                },
            )

        elif msg_type == "typing":
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "user_typing",
                    "username": self.username,
                    "is_typing": content.get("is_typing", False),
                },
            )

    # ── Handlers for group messages ──────────────────────────

    async def chat_message(self, event):
        """Send a chat message to the WebSocket."""
        await self.send_json(
            {
                "type": "message",
                "message": event["message"],
            }
        )

    async def user_status(self, event):
        await self.send_json(
            {
                "type": "presence",
                "username": event["username"],
                "status": event["status"],
            }
        )

    async def user_typing(self, event):
        await self.send_json(
            {
                "type": "typing",
                "username": event["username"],
                "is_typing": event["is_typing"],
            }
        )

    # ── Database helpers ─────────────────────────────────────

    @database_sync_to_async
    def _get_user_from_token(self, token_key):
        try:
            token = Token.objects.select_related("user").get(key=token_key)
            return token.user
        except Token.DoesNotExist:
            return AnonymousUser()

    @database_sync_to_async
    def _is_room_member(self):
        try:
            return ChatRoom.objects.filter(id=self.room_id, participants=self.user).exists()
        except Exception:
            return False

    @database_sync_to_async
    def _save_message(self, content):
        try:
            msg = Message.objects.create(
                room_id=self.room_id,
                sender=self.user,
                content=content,
            )
            # Update room timestamp
            ChatRoom.objects.filter(id=self.room_id).update(updated_at=msg.created_at)
            return msg
        except Exception:
            return None
