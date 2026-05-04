import json
import asyncio
import os
import shlex
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import async_to_sync

class JobConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.experiment_id = self.scope['url_route']['kwargs'].get('experiment_id') or \
                             self.scope['url_route']['kwargs'].get('job_id')
        if not self.experiment_id:
            await self.close()
            return
        self.group_name = f'job_{self.experiment_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def job_update(self, event):
        await self.send(text_data=json.dumps(event['data']))


import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TerminalConsumer(AsyncWebsocketConsumer):
    """
    WebSocket that receives log messages from the training process
    and sends them to all connected clients.
    No shell – just log forwarding.
    """
    async def connect(self):
        self.group_name = "terminal"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def terminal_message(self, event):
        # Send the log line to the browser
        await self.send(text_data=event['data'])