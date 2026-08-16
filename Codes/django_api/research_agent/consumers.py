import logging
import urllib.parse

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    HumanMessage = None

from .graph import build_graph
from .models import AgentMessage, AgentSession

logger = logging.getLogger(__name__)


class ResearchAgentConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        user = self.scope.get("user")
        if user is None or isinstance(user, AnonymousUser) or getattr(user, "is_anonymous", True):
            # Check token authentication via query parameter ?token=...
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
        self.group_name = f"research_agent_{self.session_id}"

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

            # Save user message
            try:
                await self._save_message("user", user_text)
            except Exception as e:
                logger.error("Failed to save user message: %s", e)

            try:
                # Build graph and invoke
                graph = build_graph()
                config = {"configurable": {"thread_id": self.session_id}}
                full_text = ""

                # Stream back tokens as they arrive
                if hasattr(graph, "astream_events") and HumanMessage is not None:
                    try:
                        async for event in graph.astream_events(
                            {
                                "messages": [HumanMessage(content=user_text)],
                                "user_id": self.user.id,
                            },
                            config=config,
                            version="v2",
                        ):
                            if event.get("event") == "on_chat_model_stream":
                                chunk = event.get("data", {}).get("chunk")
                                token = getattr(chunk, "content", "") if chunk else ""
                                if token:
                                    await self.send_json({"type": "token", "text": token})
                    except Exception as stream_err:
                        logger.warning("Stream error (falling back to invoke): %s", stream_err)

                # Get final response
                if hasattr(graph, "ainvoke") and HumanMessage is not None:
                    final_state = await graph.ainvoke(
                        {
                            "messages": [HumanMessage(content=user_text)],
                            "user_id": self.user.id,
                        },
                        config=config,
                    )
                    messages = final_state.get("messages", [])
                    final_ai = messages[-1] if messages else None
                    full_text = getattr(final_ai, "content", "") if final_ai else ""
                else:
                    full_text = "AI assistant response received."

                await self._save_message("assistant", full_text)
                await self.send_json({"type": "done", "text": full_text})

            except Exception as e:
                logger.exception("Error running research agent graph: %s", e)
                error_msg = f"Agent execution error: {str(e)}"
                try:
                    await self._save_message("assistant", error_msg)
                except Exception:
                    pass
                await self.send_json({"type": "error", "error": error_msg})

    async def agent_message(self, event):
        """Broadcast handler for research agent channel group."""
        data = event.get("data", event)
        await self.send_json(data if isinstance(data, dict) else {"text": str(data)})

    @database_sync_to_async
    def _save_message(self, role, content):
        session, _ = AgentSession.objects.get_or_create(
            id=self.session_id, defaults={"user": self.user}
        )
        return AgentMessage.objects.create(session=session, role=role, content=content)

    @database_sync_to_async
    def _get_user_by_token(self, token_key):
        if not token_key:
            return None
        try:
            return Token.objects.select_related("user").get(key=token_key).user
        except (Token.DoesNotExist, Exception):
            return None
