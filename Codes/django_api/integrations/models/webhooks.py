import uuid

from django.conf import settings
from django.db import models


class WebhookSubscription(models.Model):
    """A user's webhook registration."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='webhooks')
    url = models.URLField(max_length=500)
    events = models.JSONField(default=list, blank=True)  # e.g., ['experiment.completed', 'model.published']
    secret = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_triggered_at = models.DateTimeField(null=True, blank=True)

class WebhookDelivery(models.Model):
    """Log of each webhook delivery attempt."""
    subscription = models.ForeignKey(WebhookSubscription, on_delete=models.CASCADE, related_name='deliveries')
    event = models.CharField(max_length=100)
    payload = models.TextField()
    response_status = models.IntegerField(null=True)
    response_body = models.TextField(blank=True)
    success = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)