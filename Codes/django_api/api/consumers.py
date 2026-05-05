# api/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class JobConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Support both experiment_id and job_id URL parameters
        kwargs = self.scope['url_route']['kwargs']
        self.job_id = kwargs.get('experiment_id') or kwargs.get('job_id')
        self.group_name = f'job_{self.job_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def job_update(self, event):
        await self.send(text_data=json.dumps(event['data']))

class TerminalConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "terminal"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def terminal_message(self, event):
        await self.send(text_data=json.dumps(event) if isinstance(event, dict) else {"data": event['data']})