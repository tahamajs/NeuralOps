"""
WebSocket consumer for real-time collaborative note editing.
Users editing the same note see each other's changes instantly.
"""
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token
from .models import ResearcherNote


class NoteCollaborationConsumer(AsyncWebsocketConsumer):
    """
    Handles real-time collaborative editing of a single note.
    Group name: note_{note_id}
    """

    async def connect(self):
        self.note_id = self.scope['url_route']['kwargs']['note_id']
        self.group_name = f'note_collab_{self.note_id}'

        # Authenticate via token in query string
        token_key = self.scope.get('query_string', b'').decode()
        token_key = token_key.replace('token=', '') if 'token=' in token_key else ''
        user = await self._get_user_from_token(token_key)

        if user is None or isinstance(user, AnonymousUser):
            await self.close(code=4001)
            return

        self.user = user
        self.username = user.username

        # Check permission: user must be author or note must be public
        has_access = await self._check_note_access()
        if not has_access:
            await self.close(code=4003)
            return

        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

        # Send current note content to the new joiner
        content = await self._get_note_content()
        await self.send(text_data=json.dumps({
            "type": "init",
            "content": content,
            "version": await self._get_note_version(),
        }))

        # Notify others that a user joined
        await self.channel_layer.group_send(
            self.group_name,
            {
                "type": "user_joined",
                "username": self.username,
            }
        )

    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "user_left",
                    "username": getattr(self, 'username', 'unknown'),
                }
            )
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    # ---- Receive from WebSocket ----
    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        msg_type = data.get("type", "edit")

        if msg_type == "edit":
            content = data.get("content", "")
            cursor_pos = data.get("cursor", {})
            # Save to DB
            await self._save_note_content(content)
            # Broadcast to all other users
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "note_edit",
                    "content": content,
                    "cursor": cursor_pos,
                    "username": self.username,
                    "version": await self._get_note_version(),
                }
            )
        elif msg_type == "cursor":
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "cursor_update",
                    "cursor": data.get("cursor", {}),
                    "username": self.username,
                }
            )

    # ---- Handlers for group messages ----
    async def note_edit(self, event):
        """Broadcast note edit to client."""
        await self.send(text_data=json.dumps({
            "type": "edit",
            "content": event["content"],
            "cursor": event.get("cursor", {}),
            "username": event["username"],
            "version": event.get("version"),
        }))

    async def cursor_update(self, event):
        """Broadcast cursor position."""
        await self.send(text_data=json.dumps({
            "type": "cursor",
            "cursor": event["cursor"],
            "username": event["username"],
        }))

    async def user_joined(self, event):
        await self.send(text_data=json.dumps({
            "type": "presence",
            "action": "joined",
            "username": event["username"],
        }))

    async def user_left(self, event):
        await self.send(text_data=json.dumps({
            "type": "presence",
            "action": "left",
            "username": event["username"],
        }))

    # ---- Database helpers ----
    @database_sync_to_async
    def _get_user_from_token(self, token_key):
        try:
            token = Token.objects.select_related('user').get(key=token_key)
            return token.user
        except Token.DoesNotExist:
            return AnonymousUser()

    @database_sync_to_async
    def _check_note_access(self):
        try:
            note = ResearcherNote.objects.get(id=self.note_id, deleted_at__isnull=True)
            return note.is_public or note.author == self.user or self.user.is_staff
        except ResearcherNote.DoesNotExist:
            return False

    @database_sync_to_async
    def _get_note_content(self):
        try:
            note = ResearcherNote.objects.get(id=self.note_id)
            return note.content
        except ResearcherNote.DoesNotExist:
            return ""

    @database_sync_to_async
    def _get_note_version(self):
        try:
            note = ResearcherNote.objects.get(id=self.note_id)
            return note.version
        except ResearcherNote.DoesNotExist:
            return 1

    @database_sync_to_async
    def _save_note_content(self, content):
        try:
            note = ResearcherNote.objects.get(id=self.note_id)
            if note.content != content:
                note.content = content
                note.save()
        except ResearcherNote.DoesNotExist:
            pass