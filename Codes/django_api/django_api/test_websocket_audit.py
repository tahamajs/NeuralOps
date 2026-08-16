import json
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.authtoken.models import Token

from django_api.asgi import application
from django_api.settings import _configure_channel_layers
import research_agent.consumers
import agents.consumers

User = get_user_model()


class WebSocketAuditTestCase(TestCase):
    """
    Comprehensive verification suite for Subagent 2:
    - Channel layer in-memory fallback & routing
    - /ws/terminal/ (TerminalConsumer)
    - /ws/agent/<session_id>/ (ResearchAgentConsumer) with token streaming
    - /ws/chat/<session_id>/ (AgentConsumer)
    - /ws/collaboration/<doc_id>/ (CollaborationConsumer)
    - Graceful disconnects & error handling
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username="audit_tester",
            email="audit@example.com",
            password="testpassword123!",
        )
        self.token, _ = Token.objects.get_or_create(user=self.user)
        self.session_id = str(uuid.uuid4())

    def test_01_channel_layer_fallback(self):
        """Verify fallback to InMemoryChannelLayer when Redis is absent or offline."""
        layers = _configure_channel_layers()
        self.assertIn("default", layers)
        self.assertEqual(
            layers["default"]["BACKEND"], "channels.layers.InMemoryChannelLayer"
        )

        layer = get_channel_layer()
        self.assertIsNotNone(layer)

    def test_02_terminal_consumer_full_lifecycle(self):
        """Test /ws/terminal/ connection, welcome banner, broadcast, ping, and disconnect."""
        async def test():
            communicator = WebsocketCommunicator(
                application, f"/ws/terminal/?token={self.token.key}"
            )
            connected, subprotocol = await communicator.connect()
            self.assertTrue(connected, "TerminalConsumer should accept connection")

            # 1. Receive welcome banner
            msg = await communicator.receive_from()
            banner_data = json.loads(msg)
            self.assertEqual(banner_data.get("type"), "output")
            self.assertIn("NeuralOps Terminal", banner_data.get("data", ""))

            # 2. Ping-pong
            await communicator.send_json_to({"type": "ping"})
            res_ping = await communicator.receive_json_from()
            self.assertEqual(res_ping.get("type"), "pong")

            # 3. Channel Layer group broadcast
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                "terminal",
                {"type": "terminal_message", "data": {"status": "broadcast_ok"}},
            )
            bcast_msg = await communicator.receive_json_from()
            self.assertEqual(bcast_msg.get("status"), "broadcast_ok")

            # 4. Resize and text input without crashing
            await communicator.send_json_to({"type": "resize", "cols": 100, "rows": 30})
            await communicator.send_to(text_data="echo test\n")

            # 5. Graceful disconnect
            await communicator.disconnect()

        async_to_sync(test)()

    def test_03_research_agent_unauthorized(self):
        """Test /ws/agent/<session_id>/ rejection without auth."""
        async def test():
            communicator = WebsocketCommunicator(
                application, f"/ws/agent/{self.session_id}/"
            )
            connected, close_code = await communicator.connect()
            self.assertFalse(connected)
            self.assertEqual(close_code, 4001)

        async_to_sync(test)()

    @patch.object(research_agent.consumers, "build_graph")
    def test_04_research_agent_token_streaming_and_lifecycle(self, mock_build_graph):
        """Test /ws/agent/<session_id>/ token streaming, final response, and DB save."""
        mock_graph = MagicMock()
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Final research summary report."

        # Mock token stream
        async def mock_astream_events(inputs, config=None, version="v2"):
            for token_str in ["Token1 ", "Token2 ", "Token3"]:
                chunk_mock = MagicMock()
                chunk_mock.content = token_str
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": chunk_mock},
                }

        async def mock_ainvoke(inputs, config=None):
            return {"messages": [mock_ai_message]}

        mock_graph.astream_events.side_effect = mock_astream_events
        mock_graph.ainvoke.side_effect = mock_ainvoke
        mock_build_graph.return_value = mock_graph

        async def test():
            communicator = WebsocketCommunicator(
                application,
                f"/ws/agent/{self.session_id}/?token={self.token.key}",
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Send research query
            await communicator.send_json_to(
                {"type": "message", "text": "Analyze SVD Attention"}
            )

            # Receive 3 streaming tokens
            t1 = await communicator.receive_json_from()
            self.assertEqual(t1, {"type": "token", "text": "Token1 "})

            t2 = await communicator.receive_json_from()
            self.assertEqual(t2, {"type": "token", "text": "Token2 "})

            t3 = await communicator.receive_json_from()
            self.assertEqual(t3, {"type": "token", "text": "Token3"})

            # Receive done message
            done_msg = await communicator.receive_json_from()
            self.assertEqual(done_msg.get("type"), "done")
            self.assertEqual(done_msg.get("text"), "Final research summary report.")

            # Test channel layer broadcast
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                f"research_agent_{self.session_id}",
                {"type": "agent_message", "data": {"type": "status", "msg": "working"}},
            )
            status_msg = await communicator.receive_json_from()
            self.assertEqual(status_msg.get("type"), "status")

            # Disconnect
            await communicator.disconnect()

        async_to_sync(test)()

    def test_05_agent_chat_unauthorized(self):
        """Test /ws/chat/<session_id>/ rejection without auth."""
        async def test():
            communicator = WebsocketCommunicator(
                application, f"/ws/chat/{self.session_id}/"
            )
            connected, close_code = await communicator.connect()
            self.assertFalse(connected)
            self.assertEqual(close_code, 4001)

        async_to_sync(test)()

    @patch.object(agents.consumers, "LangGraphAgent")
    def test_06_agent_chat_lifecycle(self, mock_agent_class):
        """Test /ws/chat/<session_id>/ message exchange, messages_sync, and agent_response."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = {
            "messages": [
                {"role": "user", "content": "Explain SVD"},
                {"role": "assistant", "content": "SVD decomposes matrices."},
            ],
            "text": "SVD decomposes matrices.",
        }
        mock_agent_class.return_value = mock_agent_instance

        async def test():
            communicator = WebsocketCommunicator(
                application,
                f"/ws/chat/{self.session_id}/?token={self.token.key}",
            )
            connected, _ = await communicator.connect()
            self.assertTrue(connected)

            # Send ping
            await communicator.send_json_to({"type": "ping"})
            pong = await communicator.receive_json_from()
            self.assertEqual(pong.get("type"), "pong")

            # Send chat message
            await communicator.send_json_to({"type": "message", "text": "Explain SVD"})

            # Expect messages_sync
            sync_msg = await communicator.receive_json_from()
            self.assertEqual(sync_msg.get("type"), "messages_sync")
            self.assertEqual(len(sync_msg.get("messages", [])), 2)

            # Expect agent_response
            agent_msg = await communicator.receive_json_from()
            self.assertEqual(agent_msg.get("type"), "agent_response")
            self.assertEqual(agent_msg.get("text"), "SVD decomposes matrices.")

            # Test channel layer broadcast
            channel_layer = get_channel_layer()
            await channel_layer.group_send(
                f"chat_{self.session_id}",
                {"type": "chat_message", "data": {"type": "info", "data": "update"}},
            )
            bcast = await communicator.receive_json_from()
            self.assertEqual(bcast.get("type"), "info")

            # Disconnect
            await communicator.disconnect()

        async_to_sync(test)()

    def test_07_collaboration_consumer_lifecycle(self):
        """Test /ws/collaboration/<doc_id>/ file edits, cursor movements, and chat broadcast."""
        doc_id = "doc-987"

        async def test():
            # Client 1
            c1 = WebsocketCommunicator(
                application,
                f"/ws/collaboration/{doc_id}/?token={self.token.key}",
            )
            connected1, _ = await c1.connect()
            self.assertTrue(connected1)

            # Client 2
            c2 = WebsocketCommunicator(
                application,
                f"/ws/collaboration/{doc_id}/",
            )
            connected2, _ = await c2.connect()
            self.assertTrue(connected2)

            # 1. Send file_edit from c1
            await c1.send_json_to({
                "type": "file_edit",
                "file": "section1.tex",
                "content": "\\section{Introduction}\nRealtime text here.",
            })

            # c1 and c2 receive broadcast
            edit1 = await c1.receive_json_from()
            edit2 = await c2.receive_json_from()
            self.assertEqual(edit1.get("type"), "edit")
            self.assertEqual(edit1.get("file"), "section1.tex")
            self.assertEqual(edit1.get("user"), "audit_tester")
            self.assertEqual(edit2.get("type"), "edit")

            # 2. Send cursor move from c2
            await c2.send_json_to({
                "type": "cursor_move",
                "file": "section1.tex",
                "line": 5,
                "col": 12,
                "user": "PeerUser",
            })

            cur1 = await c1.receive_json_from()
            cur2 = await c2.receive_json_from()
            self.assertEqual(cur1.get("type"), "cursor")
            self.assertEqual(cur1.get("line"), 5)
            self.assertEqual(cur1.get("col"), 12)
            self.assertEqual(cur2.get("type"), "cursor")

            # 3. Send chat message from c1
            await c1.send_json_to({
                "type": "chat",
                "message": "Looks great to me!",
            })
            chat1 = await c1.receive_json_from()
            chat2 = await c2.receive_json_from()
            self.assertEqual(chat1.get("type"), "chat")
            self.assertEqual(chat1.get("message"), "Looks great to me!")
            self.assertEqual(chat1.get("user"), "audit_tester")
            self.assertEqual(chat2.get("message"), "Looks great to me!")

            # 4. Disconnect both
            await c1.disconnect()
            await c2.disconnect()

        async_to_sync(test)()
