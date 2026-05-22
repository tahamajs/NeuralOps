import json
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from .models import PromptVersion

class PromptCollabConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.version_id = self.scope['url_route']['kwargs']['version_id']
        self.group_name = f'prompt_collab_{self.version_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive_json(self, content):
        msg_type = content.get('type')
        if msg_type == 'edit':
            new_content = content['content']
            # Save to database
            await self._save_content(new_content)
            # Broadcast to others
            await self.channel_layer.group_send(self.group_name, {
                'type': 'broadcast_edit',
                'content': new_content,
                'sender': self.scope['user'].username,
            })
        elif msg_type == 'cursor':
            await self.channel_layer.group_send(self.group_name, {
                'type': 'broadcast_cursor',
                'cursor': content['cursor'],
                'sender': self.scope['user'].username,
            })

    async def broadcast_edit(self, event):
        await self.send_json({
            'type': 'edit',
            'content': event['content'],
            'sender': event['sender'],
        })

    async def broadcast_cursor(self, event):
        await self.send_json({
            'type': 'cursor',
            'cursor': event['cursor'],
            'sender': event['sender'],
        })

    @database_sync_to_async
    def _save_content(self, content):
        PromptVersion.objects.filter(id=self.version_id).update(content=content)