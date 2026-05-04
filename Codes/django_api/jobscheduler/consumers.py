# jobscheduler/consumers.py
import asyncio
import os
from channels.generic.websocket import AsyncWebsocketConsumer

class JobLogConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer that tails job log files and sends updates."""
    async def connect(self):
        self.job_id = self.scope['url_route']['kwargs']['job_id']
        self.group_name = f'job_{self.job_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        # Optionally send existing logs
        self.log_path = f'/tmp/job_logs/{self.job_id}.txt'
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                await self.send(text_data=f.read())
        self.task = asyncio.create_task(self.tail_log())

    async def tail_log(self):
        """Tail the log file and send new lines."""
        while True:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    f.seek(0, 2)
                    line = f.readline()
                    if line:
                        await self.send(text_data=line)
            await asyncio.sleep(0.5)

    async def disconnect(self, close_code):
        self.task.cancel()
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def job_update(self, event):
        """Receive updates from the training process via channel layer."""
        await self.send(text_data=event['data'])