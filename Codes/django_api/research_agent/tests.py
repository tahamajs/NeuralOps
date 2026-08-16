from unittest.mock import patch, MagicMock
from asgiref.sync import async_to_sync
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.authtoken.models import Token

from django_api.asgi import application

User = get_user_model()


class ResearchAgentWebSocketTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="agent_user",
            email="agent_user@example.com",
            password="password123",
        )
        self.token, _ = Token.objects.get_or_create(user=self.user)

    def test_research_agent_unauthorized(self):
        async def test():
            communicator = WebsocketCommunicator(
                application, "/ws/agent/test-session-123/"
            )
            connected, close_code = await communicator.connect()
            self.assertFalse(connected)
            self.assertEqual(close_code, 4001)

        async_to_sync(test)()

    def test_research_agent_connect_and_ping(self):
        async def test():
            communicator = WebsocketCommunicator(
                application,
                f"/ws/agent/test-session-123/?token={self.token.key}",
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Test ping
            await communicator.send_json_to({"type": "ping"})
            res = await communicator.receive_json_from()
            self.assertEqual(res.get("type"), "pong")

            # Test invalid payload format
            await communicator.send_json_to(["invalid", "list"])
            res_err = await communicator.receive_json_from()
            self.assertEqual(res_err.get("type"), "error")

            await communicator.disconnect()

        async_to_sync(test)()

    @patch("research_agent.consumers.build_graph")
    def test_research_agent_send_message(self, mock_build_graph):
        mock_graph = MagicMock()
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Research findings summary."
        mock_graph.ainvoke = MagicMock()
        
        async def mock_ainvoke(inputs, config=None):
            return {"messages": [mock_ai_message]}
            
        async def mock_astream_events(inputs, config=None, version="v2"):
            if False:
                yield {}

        mock_graph.ainvoke.side_effect = mock_ainvoke
        mock_graph.astream_events.side_effect = mock_astream_events
        mock_build_graph.return_value = mock_graph

        async def test():
            communicator = WebsocketCommunicator(
                application,
                f"/ws/agent/test-session-456/?token={self.token.key}",
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            await communicator.send_json_to({"type": "message", "text": "What is LEKA attention?"})
            res = await communicator.receive_json_from()
            self.assertEqual(res.get("type"), "done")
            self.assertEqual(res.get("text"), "Research findings summary.")

            await communicator.disconnect()

        async_to_sync(test)()
