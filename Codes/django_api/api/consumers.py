import json
import logging

from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


class JobConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        kwargs = self.scope.get("url_route", {}).get("kwargs", {})
        self.job_id = kwargs.get("experiment_id") or kwargs.get("job_id", "default")
        self.group_name = f"job_{self.job_id}"
        if self.channel_layer:
            await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "group_name") and self.channel_layer:
            try:
                await self.channel_layer.group_discard(
                    self.group_name, self.channel_name
                )
            except Exception:
                pass

    async def job_update(self, event):
        data = event.get("data", event)
        if isinstance(data, dict):
            await self.send(text_data=json.dumps(data))
        else:
            await self.send(text_data=str(data))


class TerminalConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "terminal"
        if self.channel_layer:
            await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, "group_name") and self.channel_layer:
            try:
                await self.channel_layer.group_discard(
                    self.group_name, self.channel_name
                )
            except Exception:
                pass

    async def terminal_message(self, event):
        data = event.get("data", event)
        if isinstance(data, dict):
            await self.send(text_data=json.dumps(data))
        else:
            await self.send(text_data=str(data))
