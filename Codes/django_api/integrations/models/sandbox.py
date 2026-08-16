import uuid

from django.conf import settings
from django.db import models


class SandboxSession(models.Model):
    """A sandboxed execution environment for an AI agent."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sandboxes')
    agent_session = models.ForeignKey('agents.AgentSession', on_delete=models.SET_NULL, null=True, blank=True)
    docker_image = models.CharField(max_length=300, default='python:3.11-slim')
    cpu_limit = models.FloatField(default=1.0)
    memory_limit_mb = models.IntegerField(default=512)
    network_access = models.BooleanField(default=False)
    allowed_commands = models.JSONField(default=list, blank=True)  # whitelist
    status = models.CharField(max_length=20, default='created',
        choices=[('created','Created'),('running','Running'),('stopped','Stopped'),('error','Error')])
    container_id = models.CharField(max_length=100, blank=True)
    logs = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    stopped_at = models.DateTimeField(null=True, blank=True)

class SandboxAuditLog(models.Model):
    """Every command executed in the sandbox."""
    session = models.ForeignKey(SandboxSession, on_delete=models.CASCADE, related_name='audit_logs')
    command = models.TextField()
    output = models.TextField(blank=True)
    exit_code = models.IntegerField(null=True)
    executed_at = models.DateTimeField(auto_now_add=True)