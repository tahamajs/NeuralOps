import logging
import uuid
import urllib.parse

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token

from .models import ChatSession
from .orchestrator import LangGraphAgent

logger = logging.getLogger(__name__)


class AgentConsumer(AsyncJsonWebsocketConsumer):
    """Real-time chat consumer with LangGraph agent support."""

    async def connect(self):
        user = self.scope.get("user")
        if user is None or isinstance(user, AnonymousUser) or getattr(user, "is_anonymous", True):
            raw_query = self.scope.get("query_string", b"").decode("utf-8")
            if raw_query:
                query_params = urllib.parse.parse_qs(raw_query)
                token_key = query_params.get("token", [None])[0]
                if token_key:
                    user = await self._get_user_by_token(token_key)

        if user is None or isinstance(user, AnonymousUser) or getattr(user, "is_anonymous", True):
            await self.close(code=4001)
            return

        self.user = user
        self.session_id = (
            self.scope.get("url_route", {}).get("kwargs", {}).get("session_id", "default")
        )
        self.group_name = f"chat_{self.session_id}"

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

        msg_type = content.get("type", "message")
        if msg_type == "ping":
            await self.send_json({"type": "pong"})
            return

        if msg_type == "message":
            user_text = content.get("text", "")
            if not user_text:
                return

            try:
                # Process via LangGraph agent
                result = await self._run_agent(user_text)
                if not isinstance(result, dict):
                    result = {"messages": [], "text": str(result)}

                # Send updated messages
                await self.send_json(
                    {
                        "type": "messages_sync",
                        "messages": result.get("messages", []),
                    }
                )
                await self.send_json(
                    {
                        "type": "agent_response",
                        "text": result.get("text", ""),
                    }
                )
            except Exception as e:
                logger.exception("Error running agent for session %s: %s", self.session_id, e)
                await self.send_json(
                    {
                        "type": "error",
                        "error": f"Agent error: {str(e)}",
                    }
                )

    async def chat_message(self, event):
        """Broadcast handler for chat messages."""
        data = event.get("data", event)
        await self.send_json(data if isinstance(data, dict) else {"text": str(data)})

    async def agent_message(self, event):
        """Broadcast handler for agent messages."""
        data = event.get("data", event)
        await self.send_json(data if isinstance(data, dict) else {"text": str(data)})

    @database_sync_to_async
    def _run_agent(self, text):
        try:
            val_id = uuid.UUID(str(self.session_id))
        except (ValueError, AttributeError, TypeError):
            val_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(self.session_id))

        try:
            session, _ = ChatSession.objects.get_or_create(
                id=val_id, defaults={"user": self.user}
            )
        except Exception as e:
            logger.error("Failed to get/create ChatSession: %s", e)
            session = None

        agent = LangGraphAgent(str(val_id))
        return agent.run(text)

    @database_sync_to_async
    def _get_user_by_token(self, token_key):
        if not token_key:
            return None
        try:
            return Token.objects.select_related("user").get(key=token_key).user
        except (Token.DoesNotExist, Exception):
            return None


# Backward-compatible alias
AgentChatConsumer = AgentConsumer
