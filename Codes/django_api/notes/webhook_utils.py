import hashlib
import hmac
import json
import logging

import requests
from django.utils import timezone

from .models import WebhookSubscription

logger = logging.getLogger(__name__)


def manual_send(url, payload, secret=""):
    headers = {"Content-Type": "application/json"}
    if secret:
        signature = hmac.new(
            secret.encode(), json.dumps(payload).encode(), hashlib.sha256
        ).hexdigest()
        headers["X-Signature"] = signature
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        return resp.status_code < 400
    except Exception as e:
        logger.warning(f"Webhook delivery failed to {url}: {e}")
        return False


def trigger_webhook_event(event_type, note):
    all_subs = WebhookSubscription.objects.filter(is_active=True)
    subscriptions = [s for s in all_subs if event_type in s.events]

    payload = {
        "event": event_type,
        "note_id": str(note.id),
        "title": note.title,
        "author": note.author.username,
        "timestamp": str(note.updated_at or note.created_at),
    }
    for sub in subscriptions:
        success = manual_send(sub.url, payload, sub.secret)
        if success:
            sub.last_triggered_at = timezone.now()
            sub.save(update_fields=["last_triggered_at"])
