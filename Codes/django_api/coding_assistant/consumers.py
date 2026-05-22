"""
WebSocket consumer for real‑time code generation (Copilot‑style streaming).
"""
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from rest_framework.authtoken.models import Token
from .llm_client import get_llm_response
from .models import CodeGenerationSession, CodeMessage


class CodeAssistantConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        token_key = self.scope.get('query_string', b'').decode().replace('token=', '')
        user = await self._get_user(token_key)

        if user is None or isinstance(user, AnonymousUser):
            await self.close(code=4001)
            return

        self.user = user
        session = await self._get_session()
        if session is None or session.user_id != user.id:
            await self.close(code=4003)
            return

        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        msg_type = data.get('type', 'message')

        if msg_type == 'message':
            user_content = data.get('content', '')
            language = data.get('language', 'python')

            if not user_content.strip():
                return

            # Save user message
            session = await self._get_session()
            await self._save_message('user', user_content, language)

            # Build history
            history = await self._get_history()
            messages = [
                {'role': 'system', 'content': f'You are an expert {language} developer. Write clean, well-documented code.'}
            ]
            for msg in history:
                messages.append({'role': msg.role, 'content': msg.content})

            # Stream response
            full_response = ''
            for chunk in get_llm_response(messages, stream=True):
                full_response += chunk
                await self.send(text_data=json.dumps({'type': 'chunk', 'content': chunk}))

            # Save final response
            await self._save_message('assistant', full_response, language)
            await self.send(text_data=json.dumps({'type': 'done'}))

    async def _get_user(self, token_key):
        try:
            token = await Token.objects.select_related('user').aget(key=token_key)
            return token.user
        except Token.DoesNotExist:
            return AnonymousUser()

    async def _get_session(self):
        try:
            return await CodeGenerationSession.objects.aget(id=self.session_id)
        except CodeGenerationSession.DoesNotExist:
            return None

    async def _save_message(self, role, content, language):
        session = await self._get_session()
        await CodeMessage.objects.acreate(
            session=session, role=role, content=content, language=language
        )

    async def _get_history(self):
        session = await self._get_session()
        msgs = []
        async for msg in session.messages.order_by('created_at').select_related('session'):
            msgs.append(msg)
        return msgs