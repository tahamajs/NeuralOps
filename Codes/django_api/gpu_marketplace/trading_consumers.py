import json

from channels.generic.websocket import AsyncWebsocketConsumer


class TradingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.gpu_model = self.scope["url_route"]["kwargs"]["gpu_model"]
        self.room_group_name = f"trading_{self.gpu_model}"
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        # Clients don't send messages; all updates come from signals
        pass

    async def order_book_update(self, event):
        await self.send(text_data=json.dumps(event["data"]))

    async def trade_update(self, event):
        await self.send(text_data=json.dumps(event["data"]))
