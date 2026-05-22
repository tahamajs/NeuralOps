import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import SupportTicket, TicketMessage
from .tasks import process_ticket_task

logger = logging.getLogger(__name__)


class SupportConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for a support ticket.
    Streams AI responses back to the user as they are generated.
    """
    async def connect(self):
        self.ticket_id = self.scope['url_route']['kwargs']['ticket_id']
        self.group_name = f'support_ticket_{self.ticket_id}'

        # Authenticate user via token
        user = await self._get_user()
        if user is None or not user.is_authenticated:
            await self.close(code=4001)
            return

        self.user = user

        # Verify ticket belongs to user
        ticket = await self._get_ticket()
        if ticket is None or ticket.user_id != user.id:
            await self.close(code=4003)
            return

        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

        # Send existing messages
        messages = await self._get_messages()
        await self.send(text_data=json.dumps({
            'type': 'history',
            'messages': messages,
        }))

    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        msg_type = data.get('type')

        if msg_type == 'user_message':
            content = data.get('content', '')
            if content.strip():
                # Save user message
                ticket = await self._get_ticket()
                await self._save_message(ticket, 'user', content)

                # Broadcast to group
                await self.channel_layer.group_send(
                    self.group_name,
                    {
                        'type': 'chat_message',
                        'role': 'user',
                        'content': content,
                    }
                )

                # Trigger AI processing (async task)
                from .tasks import process_ticket_message_task
                process_ticket_message_task.delay(str(self.ticket_id), content)

    async def chat_message(self, event):
        """Send a chat message to the WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'message',
            'role': event['role'],
            'content': event['content'],
        }))

    async def ai_stream(self, event):
        """Stream a chunk of AI response."""
        await self.send(text_data=json.dumps({
            'type': 'stream',
            'content': event['content'],
        }))

    async def ticket_update(self, event):
        """Notify that ticket metadata changed."""
        await self.send(text_data=json.dumps({
            'type': 'ticket_update',
            'data': event['data'],
        }))

    @database_sync_to_async
    def _get_user(self):
        # Extract token from query string
        query_string = self.scope.get('query_string', b'').decode()
        token = None
        if 'token=' in query_string:
            token = query_string.split('token=')[1]
        if token:
            from rest_framework.authtoken.models import Token
            try:
                return Token.objects.select_related('user').get(key=token).user
            except Token.DoesNotExist:
                pass
        return None

    @database_sync_to_async
    def _get_ticket(self):
        try:
            return SupportTicket.objects.get(id=self.ticket_id)
        except SupportTicket.DoesNotExist:
            return None

    @database_sync_to_async
    def _get_messages(self):
        ticket = SupportTicket.objects.get(id=self.ticket_id)
        return list(ticket.messages.values('role', 'content', 'created_at'))

    @database_sync_to_async
    def _save_message(self, ticket, role, content):
        TicketMessage.objects.create(ticket=ticket, role=role, content=content)