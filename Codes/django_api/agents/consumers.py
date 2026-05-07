import json
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from .models import ChatSession
from .orchestrator import AgentOrchestrator

class AgentChatConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        user = self.scope.get('user')
        if user is None or user.is_anonymous:
            await self.close(code=4001)
            return
        self.user = user
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        await self.accept()

    async def receive_json(self, content):
        msg_type = content.get("type", "message")
        if msg_type == "message":
            user_text = content.get("text", "")
            # Process via orchestrator
            orchestrator = await self._get_orchestrator()
            result = await database_sync_to_async(orchestrator.process_user_message)(user_text)
            # Send response back
            await self.send_json({
                "type": "agent_response",
                "text": result.get("text", ""),
                "tool_calls": result.get("tool_calls", []),
            })
            # Send updated messages
            messages = await self._get_messages()
            await self.send_json({
                "type": "messages_sync",
                "messages": messages,
            })

    @database_sync_to_async
    def _get_orchestrator(self):
        session, _ = ChatSession.objects.get_or_create(id=self.session_id, defaults={'user': self.user})
        return AgentOrchestrator(session)

    @database_sync_to_async
    def _get_messages(self):
        session = ChatSession.objects.get(id=self.session_id)
        return [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "metadata": m.metadata,
                "created_at": m.created_at.isoformat(),
            }
            for m in session.messages.order_by('created_at')
        ]