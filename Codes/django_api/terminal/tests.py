import json
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.authtoken.models import Token

from django_api.asgi import application

User = get_user_model()


class TerminalWebSocketTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="term_user",
            email="term@example.com",
            password="password123",
        )
        self.token, _ = Token.objects.get_or_create(user=self.user)

    def test_terminal_consumer_connect_and_broadcast(self):
        async def test():
            communicator = WebsocketCommunicator(
                application, f"/ws/terminal/?token={self.token.key}"
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Receive welcome banner
            msg = await communicator.receive_from()
            data = json.loads(msg)
            self.assertEqual(data.get("type"), "output")

            # Test channel layer broadcast
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                "terminal", {"type": "terminal_message", "data": "Hello Terminal"}
            )
            res = await communicator.receive_from()
            self.assertEqual(res, "Hello Terminal")

            # Test ping-pong
            await communicator.send_json_to({"type": "ping"})
            res_ping = await communicator.receive_json_from()
            self.assertEqual(res_ping.get("type"), "pong")

            # Test resize without crash
            await communicator.send_json_to({"type": "resize", "cols": 120, "rows": 40})

            # Test invalid non-dict payload doesn't crash
            await communicator.send_to(text_data="not json raw string")

            await communicator.disconnect()

        async_to_sync(test)()
