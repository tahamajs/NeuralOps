from unittest.mock import patch, MagicMock
from asgiref.sync import async_to_sync
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.authtoken.models import Token

from django_api.asgi import application

User = get_user_model()


class AgentConsumerWebSocketTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="chat_user",
            email="chat_user@example.com",
            password="password123",
        )
        self.token, _ = Token.objects.get_or_create(user=self.user)

    def test_agent_chat_unauthorized(self):
        async def test():
            communicator = WebsocketCommunicator(
                application, "/ws/chat/test-sess-123/"
            )
            connected, close_code = await communicator.connect()
            self.assertFalse(connected)
            self.assertEqual(close_code, 4001)

        async_to_sync(test)()

    def test_agent_chat_connect_and_ping(self):
        async def test():
            communicator = WebsocketCommunicator(
                application,
                f"/ws/chat/test-sess-123/?token={self.token.key}",
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Test ping
            await communicator.send_json_to({"type": "ping"})
            res = await communicator.receive_json_from()
            self.assertEqual(res.get("type"), "pong")

            # Test invalid non-dict format
            await communicator.send_json_to(["invalid", "payload"])
            res_err = await communicator.receive_json_from()
            self.assertEqual(res_err.get("type"), "error")

            await communicator.disconnect()

        async_to_sync(test)()

    @patch("agents.consumers.LangGraphAgent")
    def test_agent_chat_send_message(self, mock_agent_class):
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = {
            "messages": [{"role": "assistant", "content": "I am here to help!"}],
            "text": "I am here to help!",
        }
        mock_agent_class.return_value = mock_agent_instance

        async def test():
            communicator = WebsocketCommunicator(
                application,
                f"/ws/chat/test-sess-456/?token={self.token.key}",
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            await communicator.send_json_to({"type": "message", "text": "Hello Agent"})
            
            # Should receive sync
            res_sync = await communicator.receive_json_from()
            self.assertEqual(res_sync.get("type"), "messages_sync")
            self.assertEqual(len(res_sync.get("messages", [])), 1)

            # Should receive agent_response
            res_resp = await communicator.receive_json_from()
            self.assertEqual(res_resp.get("type"), "agent_response")
            self.assertEqual(res_resp.get("text"), "I am here to help!")

            await communicator.disconnect()

        async_to_sync(test)()
