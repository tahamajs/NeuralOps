import json
import pty
import os
import select
import threading
from channels.generic.websocket import AsyncWebsocketConsumer

class TerminalConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        user = self.scope.get('user')
        if user is None or user.is_anonymous:
            await self.close(code=4001)
            return

        await self.accept()
        self.fd, self.child_pid = pty.fork()
        if self.fd == 0:
            os.environ['TERM'] = 'xterm-256color'
            os.execvp('/bin/bash', ['/bin/bash'])
        else:
            self.thread = threading.Thread(target=self._read_pty, daemon=True)
            self.thread.start()

    def _read_pty(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            try:
                r, _, _ = select.select([self.fd], [], [], 0.1)
                if r:
                    data = os.read(self.fd, 4096)
                    if data:
                        async def send_output():
                            await self.send(text_data=json.dumps({
                                'type': 'output',
                                'data': data.decode('utf-8', errors='replace'),
                            }))
                        loop.run_until_complete(send_output())
            except (OSError, Exception):
                break

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        if data.get('type') == 'input':
            os.write(self.fd, data.get('data', '').encode())
        elif data.get('type') == 'resize':
            cols = data.get('cols', 80)
            rows = data.get('rows', 24)
            try:
                import fcntl, termios, struct
                winsize = struct.pack('HHHH', rows, cols, 0, 0)
                fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)
            except:
                pass

    async def disconnect(self, close_code):
        try:
            os.close(self.fd)
        except:
            pass
        try:
            os.kill(self.child_pid, 9)
        except:
            pass