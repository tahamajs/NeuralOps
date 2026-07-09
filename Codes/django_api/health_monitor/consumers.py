import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class GlobalHealthConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.dashboard_id = self.scope['url_route']['kwargs'].get('dashboard_id', 'global')
        self.group_name = f"health_{self.dashboard_id}"
        
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        logger.info(f"Health WebSocket connected for dashboard: {self.dashboard_id}")

        # Send initial health snapshot immediately
        await self.send(text_data=json.dumps({
            "type": "health_update",
            "data": {
                "status": "healthy",
                "resources": {
                    "cpu": 45.2,
                    "memory": 62.1,
                    "disk": 51.4
                },
                "services": {
                    "database": { "status": "healthy", "latency": 2 },
                    "redis": { "status": "healthy", "latency": 1 },
                    "celery": { "status": "healthy" }
                }
            }
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
        logger.info(f"Health WebSocket disconnected: {self.dashboard_id}")

    async def receive(self, text_data):
        pass

    async def health_update(self, event):
        await self.send(text_data=json.dumps({
            "type": "health_update",
            "data": event["data"]
        }))

    async def alert(self, event):
        await self.send(text_data=json.dumps({
            "type": "alert",
            "data": event["data"]
        }))
