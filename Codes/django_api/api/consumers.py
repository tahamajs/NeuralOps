import json
import asyncio
import os
import shlex
from channels.generic.websocket import AsyncWebsocketConsumer

class JobConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.experiment_id = self.scope['url_route']['kwargs'].get('experiment_id') or \
                             self.scope['url_route']['kwargs'].get('job_id')
        if not self.experiment_id:
            await self.close()
            return
        self.group_name = f'job_{self.experiment_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def job_update(self, event):
        await self.send(text_data=json.dumps(event['data']))


class TerminalConsumer(AsyncWebsocketConsumer):
    """WebSocket that spawns a bash shell and forwards I/O."""
    async def connect(self):
        self.process = None
        self.read_task = None
        await self.accept()
        # Spawn a restricted bash shell (change to /bin/bash or /bin/sh)
        # You can limit commands by using a custom wrapper or rbash.
        self.process = await asyncio.create_subprocess_shell(
            "/bin/bash",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/tmp",   # start in a safe directory
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"},
        )
        self.read_task = asyncio.create_task(self.read_output())

    async def disconnect(self, close_code):
        if self.process:
            self.process.terminate()
        if self.read_task:
            self.read_task.cancel()

    async def read_output(self):
        while True:
            try:
                data = await self.process.stdout.read(1024)
                if not data:
                    break
                await self.send(text_data=data.decode('utf-8', errors='replace'))
            except:
                break

    async def receive(self, text_data):
        if self.process and self.process.stdin:
            self.process.stdin.write(text_data.encode())
            await self.process.stdin.drain()