"""
WebSocket consumer for real-time AI Research Coach chat.
"""
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from rest_framework.authtoken.models import Token
from .models import ChatSession, ChatMessage
from .agent import ResearchCoachAgent


class ResearchCoachConsumer(AsyncWebsocketConsumer):
    """Handles real-time chat with the AI Research Coach."""
    
    async def connect(self):
        self.user = None
        token_key = self.scope.get('query_string', b'').decode().replace('token=', '')
        if token_key:
            self.user = await self._get_user_from_token(token_key)
        
        if not self.user:
            await self.close(code=4001)
            return
        
        self.session_id = self.scope['url_route']['kwargs'].get('session_id')
        self.room_group_name = f"coach_{self.user.id}"
        
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
    
    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        msg_type = data.get('type', 'message')
        
        if msg_type == 'message':
            user_text = data.get('content', '')
            if not user_text.strip():
                return
            
            # Save user message
            session = await self._get_or_create_session()
            await self._save_message(session, 'user', user_text)
            
            # Get history
            history = await self._get_chat_history(session)
            
            # Call agent
            agent = ResearchCoachAgent(self.user.id)
            response = agent.chat(user_text, session_history=history)
            
            # Save assistant message
            await self._save_message(session, 'assistant', response)
            
            # Send back
            await self.send(text_data=json.dumps({
                'type': 'response',
                'content': response,
                'session_id': str(session.id),
            }))
    
    @database_sync_to_async
    def _get_user_from_token(self, token_key):
        try:
            token = Token.objects.select_related('user').get(key=token_key)
            return token.user
        except Token.DoesNotExist:
            return None
    
    @database_sync_to_async
    def _get_or_create_session(self):
        if self.session_id:
            try:
                return ChatSession.objects.get(id=self.session_id, user=self.user)
            except ChatSession.DoesNotExist:
                pass
        return ChatSession.objects.create(user=self.user, title="Research Coach Chat")
    
    @database_sync_to_async
    def _save_message(self, session, role, content):
        return ChatMessage.objects.create(session=session, role=role, content=content)
    
    @database_sync_to_async
    def _get_chat_history(self, session):
        messages = session.messages.order_by('created_at')[:20]
        from langchain_core.messages import HumanMessage, AIMessage
        history = []
        for msg in messages:
            if msg.role == 'user':
                history.append(HumanMessage(content=msg.content))
            elif msg.role == 'assistant':
                history.append(AIMessage(content=msg.content))
        return history