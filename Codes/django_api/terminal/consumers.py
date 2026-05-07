import json
import os
import pty
import select
import struct
import termios
import fcntl
import signal
import threading
import queue
import asyncio
from datetime import timedelta

from channels.generic.websocket import AsyncWebsocketConsumer
from django.utils import timezone
from django.conf import settings

from .models import TerminalSession

MAX_SESSIONS_PER_USER = getattr(settings, 'TERMINAL_MAX_SESSIONS_PER_USER', 5)
SESSION_IDLE_TIMEOUT = getattr(settings, 'TERMINAL_SESSION_IDLE_TIMEOUT', 0)


class TerminalConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        user = self.scope.get('user')
        if user is None or user.is_anonymous:
            await self.close(code=4001)
            return

        # Enforce session limit
        from channels.db import database_sync_to_async
        active_count = await database_sync_to_async(
            TerminalSession.objects.filter(user=user, is_active=True).count
        )()
        if active_count >= MAX_SESSIONS_PER_USER:
            await self.close(code=4002)
            return

        await self.accept()
        self.user = user
        self.child_pid = None
        self.fd = None
        self.session = None
        self._stop_event = threading.Event()
        self._output_queue = queue.Queue()

        # Spawn PTY
        try:
            self.child_pid, self.fd = pty.fork()
        except OSError as e:
            await self.send(text_data=json.dumps({
                'type': 'error', 'data': str(e)
            }))
            await self.close()
            return

        if self.child_pid == 0:
            os.environ['TERM'] = 'xterm-256color'
            shell = os.environ.get('SHELL', '/bin/bash')
            os.execvp(shell, [shell])
            os._exit(1)
        else:
            self.session = await database_sync_to_async(
                TerminalSession.objects.create
            )(user=self.user, channel_name=self.channel_name,
              pid=self.child_pid, fd=self.fd)

            self._reader_thread = threading.Thread(
                target=self._read_pty, daemon=True
            )
            self._reader_thread.start()

            # Start drain task
            self._drain_task = asyncio.ensure_future(self._drain_output())

            # Welcome
            await self.send(text_data=json.dumps({
                'type': 'output',
                'data': '\x1b[1;32m╔══════════════════════════════════════╗\x1b[0m\r\n'
                        '\x1b[1;32m║   NeuralOps Terminal                ║\x1b[0m\r\n'
                        '\x1b[1;32m╚══════════════════════════════════════╝\x1b[0m\r\n\r\n',
            }))

            if SESSION_IDLE_TIMEOUT > 0:
                self._idle_task = asyncio.ensure_future(self._idle_timeout())

    async def disconnect(self, close_code):
        self._stop_event.set()
        if hasattr(self, '_drain_task'):
            self._drain_task.cancel()
        if hasattr(self, '_idle_task'):
            self._idle_task.cancel()
        if self.session:
            self.session.mark_inactive()
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
        data = json.loads(text_data)
        if data.get('type') == 'input' and self.fd:
            os.write(self.fd, data['data'].encode())
            await self._touch_activity()
        elif data.get('type') == 'resize' and self.fd:
            cols = data.get('cols', 80)
            rows = data.get('rows', 24)
            try:
                winsize = struct.pack('HHHH', rows, cols, 0, 0)
                fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)
            except Exception:
                pass

    def _read_pty(self):
        while not self._stop_event.is_set():
            try:
                r, _, _ = select.select([self.fd], [], [], 0.5)
                if r:
                    data = os.read(self.fd, 4096)
                    if data:
                        self._output_queue.put(data)
            except (OSError, Exception):
                break

    async def _drain_output(self):
        while True:
            try:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self._output_queue.get, True, 1
                )
                await self.send(text_data=json.dumps({
                    'type': 'output',
                    'data': data.decode('utf-8', errors='replace'),
                }))
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _idle_timeout(self):
        await asyncio.sleep(SESSION_IDLE_TIMEOUT)
        await self.send(text_data=json.dumps({
            'type': 'output',
            'data': '\r\n\x1b[1;31mSession timed out due to inactivity.\x1b[0m\r\n',
        }))
        await self.close()

    async def _touch_activity(self):
        if self.session:
            from channels.db import database_sync_to_async
            await database_sync_to_async(
                TerminalSession.objects.filter(id=self.session.id).update
            )(last_activity_at=timezone.now())