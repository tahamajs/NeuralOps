import logging
import urllib.parse

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token

from .models import PaperFile, PaperProject

logger = logging.getLogger(__name__)


class PaperCollabConsumer(AsyncJsonWebsocketConsumer):
    """Real-time collaborative editing for paper files."""

    async def connect(self):
        kwargs = self.scope.get("url_route", {}).get("kwargs", {})
        self.project_id = kwargs.get("project_id") or kwargs.get("doc_id", "default")
        self.group_name = f"paper_collab_{self.project_id}"

        # Parse token from query string
        raw_query = self.scope.get("query_string", b"").decode("utf-8")
        query_params = urllib.parse.parse_qs(raw_query) if raw_query else {}
        token_key = query_params.get("token", [None])[0]

        user = await self._get_user(token_key)
        if user is None or isinstance(user, AnonymousUser) or getattr(user, "is_anonymous", True):
            # Check if user is in scope (e.g. from session auth or TokenAuthMiddleware)
            scope_user = self.scope.get("user")
            if scope_user and getattr(scope_user, "is_authenticated", False):
                user = scope_user
            else:
                await self.close(code=4001)
                return

        self.user = user
        has_access = await self._check_access()
        if not has_access:
            await self.close(code=4003)
            return

        if self.channel_layer:
            await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "group_name") and self.channel_layer:
            try:
                await self.channel_layer.group_discard(
                    self.group_name, self.channel_name
                )
            except Exception:
                pass

    async def receive_json(self, content):
        if not isinstance(content, dict):
            await self.send_json(
                {"type": "error", "error": "Invalid payload; expected a JSON object."}
            )
            return

        msg_type = content.get("type")
        if msg_type == "ping":
            await self.send_json({"type": "pong"})
            return

        if msg_type == "file_edit":
            filename = content.get("filename")
            new_content = content.get("content", "")
            if filename:
                await self._save_file(filename, new_content)
                if self.channel_layer:
                    await self.channel_layer.group_send(
                        self.group_name,
                        {
                            "type": "broadcast_edit",
                            "filename": filename,
                            "content": new_content,
                            "username": getattr(self.user, "username", "collaborator"),
                        },
                    )
        elif msg_type in ("cursor", "cursor_move"):
            if self.channel_layer:
                await self.channel_layer.group_send(
                    self.group_name,
                    {
                        "type": "broadcast_cursor",
                        "filename": content.get("filename"),
                        "cursor": content.get("cursor", {}),
                        "line": content.get("line"),
                        "col": content.get("col"),
                        "username": getattr(self.user, "username", "collaborator"),
                    },
                )

    async def broadcast_edit(self, event):
        await self.send_json(
            {
                "type": "edit",
                "filename": event.get("filename"),
                "content": event.get("content"),
                "username": event.get("username"),
            }
        )

    async def broadcast_cursor(self, event):
        await self.send_json(
            {
                "type": "cursor",
                "filename": event.get("filename"),
                "cursor": event.get("cursor"),
                "line": event.get("line"),
                "col": event.get("col"),
                "username": event.get("username"),
            }
        )

    @database_sync_to_async
    def _get_user(self, token_key):
        if not token_key:
            return AnonymousUser()
        try:
            return Token.objects.select_related("user").get(key=token_key).user
        except (Token.DoesNotExist, Exception):
            return AnonymousUser()

    @database_sync_to_async
    def _check_access(self):
        try:
            project = PaperProject.objects.get(id=self.project_id)
            return (
                project.owner == self.user
                or project.collaborators.filter(id=self.user.id).exists()
            )
        except (PaperProject.DoesNotExist, Exception):
            return False

    @database_sync_to_async
    def _save_file(self, filename, content):
        try:
            f = PaperFile.objects.get(project_id=self.project_id, filename=filename)
            f.content = content
            f.version += 1
            f.save(update_fields=["content", "version", "updated_at"])
        except (PaperFile.DoesNotExist, Exception):
            pass


class CollaborationConsumer(AsyncJsonWebsocketConsumer):
    """Generic collaboration consumer for paper editing & team chat."""

    async def connect(self):
        kwargs = self.scope.get("url_route", {}).get("kwargs", {})
        self.doc_id = kwargs.get("doc_id") or kwargs.get("project_id", "default")
        self.room_group_name = f"collab_{self.doc_id}"

        # Resolve user authentication from scope or query string
        raw_user = self.scope.get("user")
        if raw_user is None or isinstance(raw_user, AnonymousUser) or getattr(raw_user, "is_anonymous", True):
            raw_query = self.scope.get("query_string", b"").decode("utf-8")
            if raw_query:
                query_params = urllib.parse.parse_qs(raw_query)
                token_key = query_params.get("token", [None])[0]
                if token_key:
                    raw_user = await self._get_user(token_key)

        self.user = raw_user if (raw_user and getattr(raw_user, "is_authenticated", False)) else None

        if self.channel_layer:
            await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "room_group_name") and self.channel_layer:
            try:
                await self.channel_layer.group_discard(
                    self.room_group_name, self.channel_name
                )
            except Exception:
                pass

    async def receive_json(self, content):
        if not isinstance(content, dict):
            await self.send_json(
                {"type": "error", "error": "Invalid payload; expected a JSON object."}
            )
            return

        msg_type = content.get("type")
        if msg_type == "ping":
            await self.send_json({"type": "pong"})
            return

        username = (
            self.user.username
            if self.user and getattr(self.user, "is_authenticated", False)
            else content.get("user", "Anonymous")
        )

        if msg_type == "file_edit":
            if self.channel_layer:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        "type": "broadcast_edit",
                        "file": content.get("file"),
                        "content": content.get("content"),
                        "user": username,
                    },
                )
        elif msg_type in ("cursor_move", "cursor"):
            if self.channel_layer:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        "type": "broadcast_cursor",
                        "file": content.get("file"),
                        "line": content.get("line"),
                        "col": content.get("col"),
                        "user": username,
                    },
                )
        elif msg_type == "chat":
            if self.channel_layer:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        "type": "broadcast_chat",
                        "message": content.get("message"),
                        "user": username,
                    },
                )

    async def broadcast_edit(self, event):
        await self.send_json(
            {
                "type": "edit",
                "file": event.get("file"),
                "content": event.get("content"),
                "user": event.get("user"),
            }
        )

    async def broadcast_cursor(self, event):
        await self.send_json(
            {
                "type": "cursor",
                "file": event.get("file"),
                "line": event.get("line"),
                "col": event.get("col"),
                "user": event.get("user"),
            }
        )

    async def broadcast_chat(self, event):
        await self.send_json(
            {
                "type": "chat",
                "message": event.get("message"),
                "user": event.get("user"),
            }
        )

    @database_sync_to_async
    def _get_user(self, token_key):
        if not token_key:
            return None
        try:
            return Token.objects.select_related("user").get(key=token_key).user
        except (Token.DoesNotExist, Exception):
            return None
