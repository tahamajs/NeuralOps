import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import Experiment

class JobConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.job_id = self.scope['url_route']['kwargs']['job_id']
        self.group_name = f"job_{self.job_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        exp = await self.get_experiment()
        if exp:
            await self.send(text_data=json.dumps({"type": "status", "status": exp.status}))
            if exp.logs:
                await self.send(text_data=json.dumps({"type": "log", "data": exp.logs[-1000:]}))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def send_message(self, event):
        await self.send(text_data=json.dumps(event["message"]))

    @database_sync_to_async
    def get_experiment(self):
        try:
            return Experiment.objects.get(id=self.job_id)
        except Experiment.DoesNotExist:
            return None