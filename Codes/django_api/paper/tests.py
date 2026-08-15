# paper/tests.py
import json
import pytest
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.authtoken.models import Token
from channels.testing import WebsocketCommunicator
from channels.routing import URLRouter
from django.urls import re_path

from .models import PaperProject, PaperFile
from .consumers import PaperCollabConsumer, CollaborationConsumer

User = get_user_model()


class PaperWebSocketTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="testauthor",
            email="author@example.com",
            password="testpassword123",
        )
        self.token, _ = Token.objects.get_or_create(user=self.user)
        self.project = PaperProject.objects.create(
            owner=self.user,
            name="Test Paper Project",
            root_file="main.tex",
        )
        self.file = PaperFile.objects.create(
            project=self.project,
            filename="main.tex",
            content="\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}",
        )

    async def test_collaboration_consumer_basic_flow(self):
        application = URLRouter([
            re_path(r"ws/collab/(?P<project_id>[^/]+)/$", CollaborationConsumer.as_asgi()),
        ])

        communicator = WebsocketCommunicator(
            application, f"/ws/collab/{self.project.id}/"
        )
        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Test sending edit
        await communicator.send_json_to({
            "type": "file_edit",
            "file": "main.tex",
            "content": "\\documentclass{article}\n\\begin{document}\nUpdated\n\\end{document}",
            "user": "testauthor",
        })

        response = await communicator.receive_json_from()
        self.assertEqual(response["type"], "edit")
        self.assertEqual(response["file"], "main.tex")
        self.assertIn("Updated", response["content"])

        # Test sending chat
        await communicator.send_json_to({
            "type": "chat",
            "message": "Looks great!",
            "user": "testauthor",
        })
        chat_res = await communicator.receive_json_from()
        self.assertEqual(chat_res["type"], "chat")
        self.assertEqual(chat_res["message"], "Looks great!")

        await communicator.disconnect()

    async def test_paper_collab_consumer_authenticated_flow(self):
        application = URLRouter([
            re_path(
                r"ws/paper/(?P<project_id>[^/]+)/$",
                PaperCollabConsumer.as_asgi(),
            ),
        ])

        # Connect with valid token query param
        communicator = WebsocketCommunicator(
            application,
            f"/ws/paper/{self.project.id}/?token={self.token.key}",
        )
        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Send file edit
        new_content = "\\documentclass{article}\n\\begin{document}\nRealtime Content\n\\end{document}"
        await communicator.send_json_to({
            "type": "file_edit",
            "filename": "main.tex",
            "content": new_content,
        })

        edit_broadcast = await communicator.receive_json_from()
        self.assertEqual(edit_broadcast["type"], "edit")
        self.assertEqual(edit_broadcast["filename"], "main.tex")
        self.assertEqual(edit_broadcast["content"], new_content)

        # Send cursor update
        await communicator.send_json_to({
            "type": "cursor",
            "filename": "main.tex",
            "cursor": {"line": 2, "ch": 5},
        })
        cursor_broadcast = await communicator.receive_json_from()
        self.assertEqual(cursor_broadcast["type"], "cursor")
        self.assertEqual(cursor_broadcast["filename"], "main.tex")

        await communicator.disconnect()

    async def test_paper_collab_consumer_unauthorized(self):
        application = URLRouter([
            re_path(
                r"ws/paper/(?P<project_id>[^/]+)/$",
                PaperCollabConsumer.as_asgi(),
            ),
        ])

        # Attempt connect with invalid token
        communicator = WebsocketCommunicator(
            application,
            f"/ws/paper/{self.project.id}/?token=invalidtoken123",
        )
        connected, close_code = await communicator.connect()
        self.assertFalse(connected)
        self.assertEqual(close_code, 4001)
