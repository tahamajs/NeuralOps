from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from .models import ChatSession
from .orchestrator import LangGraphAgent


class AgentChatConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        user = self.scope.get("user")
        if user is None or user.is_anonymous:
            await self.close(code=4001)
            return
        self.user = user
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        await self.accept()

    async def receive_json(self, content):
        msg_type = content.get("type", "message")
        if msg_type == "message":
            user_text = content.get("text", "")
            # Process via LangGraph agent
            result = await self._run_agent(user_text)
            # Send updated messages
            await self.send_json(
                {
                    "type": "messages_sync",
                    "messages": result["messages"],
                }
            )
            await self.send_json(
                {
                    "type": "agent_response",
                    "text": result["text"],
                }
            )

    @database_sync_to_async
    def _run_agent(self, text):
        session, _ = ChatSession.objects.get_or_create(
            id=self.session_id, defaults={"user": self.user}
        )
        agent = LangGraphAgent(str(session.id))
        return agent.run(text)
