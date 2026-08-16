import logging

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    HumanMessage = None

from .graph import build_graph
from .models import AgentMessage, AgentSession

logger = logging.getLogger(__name__)


class ResearchAgentConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        if self.user.is_anonymous:
            await self.close()
            return
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        await self.accept()

    async def receive_json(self, content):
        msg_type = content.get("type", "message")
        if msg_type == "message":
            user_text = content.get("text", "")
            # Save user message
            await self._save_message("user", user_text)

            # Build graph and invoke
            graph = build_graph()
            config = {"configurable": {"thread_id": self.session_id}}

            # Stream back tokens as they arrive
            async for event in graph.astream_events(
                {
                    "messages": [HumanMessage(content=user_text)],
                    "user_id": self.user.id,
                },
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        await self.send_json({"type": "token", "text": token})

            # After streaming, get final state and save assistant message
            final_state = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_text)],
                    "user_id": self.user.id,
                },
                config=config,
            )
            final_ai = final_state["messages"][-1]
            full_text = final_ai.content if hasattr(final_ai, "content") else ""
            await self._save_message("assistant", full_text)
            await self.send_json({"type": "done"})

    @database_sync_to_async
    def _save_message(self, role, content):
        session, _ = AgentSession.objects.get_or_create(
            id=self.session_id, defaults={"user": self.user}
        )
        return AgentMessage.objects.create(session=session, role=role, content=content)
