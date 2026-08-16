import asyncio
import fcntl
import json
import logging
import os
import pty
import queue
import select
import signal
import struct
import termios
import threading
import urllib.parse

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.utils import timezone
from rest_framework.authtoken.models import Token

from .models import TerminalSession

logger = logging.getLogger(__name__)

MAX_SESSIONS_PER_USER = getattr(settings, "TERMINAL_MAX_SESSIONS_PER_USER", 5)
SESSION_IDLE_TIMEOUT = getattr(settings, "TERMINAL_SESSION_IDLE_TIMEOUT", 0)


class TerminalConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Initialise all attributes
        self.user = None
        self.child_pid = None
        self.fd = None
        self.session = None
        self.group_name = "terminal"
        self._stop_event = threading.Event()
        self._output_queue = queue.Queue()
        self._reader_thread = None
        self._drain_task = None
        self._idle_task = None

        # Accept the connection first
        await self.accept()

        # Join channel layer group if available
        if self.channel_layer:
            await self.channel_layer.group_add(self.group_name, self.channel_name)

        # Resolve user – check scope or query param token
        raw_user = self.scope.get("user")
        if (
            raw_user is None
            or isinstance(raw_user, AnonymousUser)
            or getattr(raw_user, "is_anonymous", True)
        ):
            raw_query = self.scope.get("query_string", b"").decode("utf-8")
            if raw_query:
                query_params = urllib.parse.parse_qs(raw_query)
                token_key = query_params.get("token", [None])[0]
                if token_key:
                    self.user = await self._get_user_by_token(token_key)
            if self.user is None:
                self.user = None
                logger.info(
                    "Terminal WebSocket connected – anonymous user (no auth cookie or token)"
                )
        else:
            self.user = await database_sync_to_async(lambda: raw_user)()

        # Enforce session limit (only for authenticated users)
        if self.user is not None and getattr(self.user, "is_authenticated", False):
            active_count = await database_sync_to_async(
                TerminalSession.objects.filter(user=self.user, is_active=True).count
            )()
            if active_count >= MAX_SESSIONS_PER_USER:
                await self.send(
                    text_data=json.dumps(
                        {
                            "type": "error",
                            "data": f"Maximum {MAX_SESSIONS_PER_USER} terminal sessions allowed.",
                        }
                    )
                )
                await self.close()
                return

        # Spawn a pseudo-terminal (PTY)
        try:
            self.child_pid, self.fd = pty.fork()
        except OSError as e:
            logger.error("Failed to spawn PTY: %s", e)
            await self.send(text_data=json.dumps({"type": "error", "data": str(e)}))
            await self.close()
            return

        if self.child_pid == 0:
            # Child process – run the shell
            os.environ["TERM"] = "xterm-256color"
            shell = os.environ.get("SHELL", "/bin/bash")
            os.execvp(shell, [shell])
            os._exit(1)
        else:
            # Parent – track session if authenticated
            if self.user is not None and getattr(self.user, "is_authenticated", False):
                self.session = await database_sync_to_async(
                    TerminalSession.objects.create
                )(user=self.user, pid=self.child_pid, fd=self.fd)
            else:
                self.session = None

            self._reader_thread = threading.Thread(target=self._read_pty, daemon=True)
            self._reader_thread.start()
            self._drain_task = asyncio.ensure_future(self._drain_output())

            # Welcome banner
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "output",
                        "data": "\x1b[1;32m╔══════════════════════════════════════╗\x1b[0m\r\n"
                        "\x1b[1;32m║   NeuralOps Terminal                 ║\x1b[0m\r\n"
                        "\x1b[1;32m╚══════════════════════════════════════╝\x1b[0m\r\n\r\n",
                    }
                )
            )

            if SESSION_IDLE_TIMEOUT > 0:
                self._idle_task = asyncio.ensure_future(self._idle_timeout())

    async def disconnect(self, close_code):
        self._stop_event.set()
        if hasattr(self, "group_name") and self.channel_layer:
            try:
                await self.channel_layer.group_discard(
                    self.group_name, self.channel_name
                )
            except Exception:
                pass

        if self._drain_task:
            self._drain_task.cancel()
        if self._idle_task:
            self._idle_task.cancel()
        if self.session:
            try:
                await database_sync_to_async(self.session.mark_inactive)()
            except Exception:
                pass
        if self.child_pid:
            try:
                os.kill(self.child_pid, signal.SIGKILL)
            except OSError:
                pass
        if self.fd:
            try:
                os.close(self.fd)
            except OSError:
                pass

    async def receive(self, text_data=None, bytes_data=None):
        if text_data is None and bytes_data is None:
            return

        if SESSION_IDLE_TIMEOUT > 0:
            if self._idle_task and not self._idle_task.done():
                self._idle_task.cancel()
            self._idle_task = asyncio.ensure_future(self._idle_timeout())

        # Handle binary input directly
        if bytes_data is not None and self.fd:
            try:
                os.write(self.fd, bytes_data)
                await self._touch_activity()
            except OSError:
                pass
            return

        # Attempt to parse as JSON; fallback to raw text writing
        try:
            data = json.loads(text_data)
        except (json.JSONDecodeError, TypeError, ValueError):
            if isinstance(text_data, str) and self.fd:
                try:
                    os.write(self.fd, text_data.encode("utf-8", errors="replace"))
                    await self._touch_activity()
                except OSError:
                    pass
            return

        if not isinstance(data, dict):
            if self.fd:
                try:
                    os.write(self.fd, str(data).encode("utf-8", errors="replace"))
                    await self._touch_activity()
                except OSError:
                    pass
            return

        msg_type = data.get("type")
        if msg_type == "input":
            payload = data.get("data", "")
            if isinstance(payload, str) and self.fd:
                try:
                    os.write(self.fd, payload.encode("utf-8", errors="replace"))
                    await self._touch_activity()
                except OSError:
                    pass
        elif msg_type == "resize":
            if not self.fd:
                return
            try:
                cols = int(data.get("cols", 80))
                rows = int(data.get("rows", 24))
                winsize = struct.pack("HHHH", rows, cols, 0, 0)
                fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)
            except Exception:
                pass
        elif msg_type == "ping":
            await self.send(text_data=json.dumps({"type": "pong"}))

    async def terminal_message(self, event):
        """Handle broadcast messages sent to 'terminal' group."""
        data = event.get("data", event)
        if isinstance(data, dict):
            await self.send(text_data=json.dumps(data))
        else:
            await self.send(text_data=str(data))

    def _read_pty(self):
        while not self._stop_event.is_set():
            if not self.fd:
                break
            try:
                r, _, _ = select.select([self.fd], [], [], 0.5)
                if r:
                    data = os.read(self.fd, 4096)
                    if data:
                        self._output_queue.put(data)
                    else:
                        break
            except (OSError, Exception):
                break

    async def _drain_output(self):
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            try:
                data = await loop.run_in_executor(
                    None, self._output_queue.get, True, 0.5
                )
                if data:
                    await self.send(
                        text_data=json.dumps(
                            {
                                "type": "output",
                                "data": data.decode("utf-8", errors="replace"),
                            }
                        )
                    )
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Drain output error: %s", e)
                break

    async def _idle_timeout(self):
        try:
            await asyncio.sleep(SESSION_IDLE_TIMEOUT)
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "output",
                        "data": "\r\n\x1b[1;31mSession timed out due to inactivity.\x1b[0m\r\n",
                    }
                )
            )
            await self.close()
        except asyncio.CancelledError:
            pass

    async def _touch_activity(self):
        if self.session:
            try:
                await database_sync_to_async(
                    TerminalSession.objects.filter(id=self.session.id).update
                )(last_activity_at=timezone.now())
            except Exception:
                pass

    @database_sync_to_async
    def _get_user_by_token(self, token_key):
        if not token_key:
            return None
        try:
            return Token.objects.select_related("user").get(key=token_key).user
        except (Token.DoesNotExist, Exception):
            return None
