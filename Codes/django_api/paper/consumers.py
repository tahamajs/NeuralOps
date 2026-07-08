# paper/consumers.py
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token

from .models import PaperFile, PaperProject


class PaperCollabConsumer(AsyncJsonWebsocketConsumer):
    """Real-time collaborative editing for paper files."""

    async def connect(self):
        self.project_id = self.scope["url_route"]["kwargs"]["project_id"]
        self.group_name = f"paper_collab_{self.project_id}"
        token_key = self.scope.get("query_string", b"").decode().replace("token=", "")
        user = await self._get_user(token_key)
        if user is None or isinstance(user, AnonymousUser):
            await self.close(code=4001)
            return
        self.user = user
        has_access = await self._check_access()
        if not has_access:
            await self.close(code=4003)
            return
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive_json(self, content):
        msg_type = content.get("type")
        if msg_type == "file_edit":
            filename = content.get("filename")
            new_content = content.get("content", "")
            await self._save_file(filename, new_content)
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "broadcast_edit",
                    "filename": filename,
                    "content": new_content,
                    "username": self.user.username,
                },
            )
        elif msg_type == "cursor":
            await self.channel_layer.group_send(
                self.group_name,
                {
                    "type": "broadcast_cursor",
                    "filename": content.get("filename"),
                    "cursor": content.get("cursor", {}),
                    "username": self.user.username,
                },
            )

    async def broadcast_edit(self, event):
        await self.send_json(
            {
                "type": "edit",
                "filename": event["filename"],
                "content": event["content"],
                "username": event["username"],
            }
        )

    async def broadcast_cursor(self, event):
        await self.send_json(
            {
                "type": "cursor",
                "filename": event["filename"],
                "cursor": event["cursor"],
                "username": event["username"],
            }
        )

    @database_sync_to_async
    def _get_user(self, token_key):
        try:
            return Token.objects.select_related("user").get(key=token_key).user
        except Token.DoesNotExist:
            return AnonymousUser()

    @database_sync_to_async
    def _check_access(self):
        try:
            project = PaperProject.objects.get(id=self.project_id)
            return (
                project.owner == self.user
                or project.collaborators.filter(id=self.user.id).exists()
            )
        except PaperProject.DoesNotExist:
            return False

    @database_sync_to_async
    def _save_file(self, filename, content):
        try:
            f = PaperFile.objects.get(project_id=self.project_id, filename=filename)
            f.content = content
            f.version += 1
            f.save(update_fields=["content", "version", "updated_at"])
        except PaperFile.DoesNotExist:
            pass


class CollaborationConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.project_id = self.scope["url_route"]["kwargs"]["project_id"]
        self.room_group_name = f"collab_{self.project_id}"
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def receive_json(self, content):
        msg_type = content.get("type")
        if msg_type == "file_edit":
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "broadcast_edit",
                    "file": content["file"],
                    "content": content["content"],
                    "user": self.scope["user"].username,
                },
            )
        elif msg_type == "cursor_move":
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "broadcast_cursor",
                    "file": content["file"],
                    "line": content["line"],
                    "col": content["col"],
                    "user": self.scope["user"].username,
                },
            )
        elif msg_type == "chat":
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "broadcast_chat",
                    "message": content["message"],
                    "user": self.scope["user"].username,
                },
            )

    async def broadcast_edit(self, event):
        await self.send_json(
            {
                "type": "edit",
                "file": event["file"],
                "content": event["content"],
                "user": event["user"],
            }
        )

    async def broadcast_cursor(self, event):
        await self.send_json(
            {
                "type": "cursor",
                "file": event["file"],
                "line": event["line"],
                "col": event["col"],
                "user": event["user"],
            }
        )

    async def broadcast_chat(self, event):
        await self.send_json(
            {"type": "chat", "message": event["message"], "user": event["user"]}
        )
