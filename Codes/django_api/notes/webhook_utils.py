# notes/webhook_utils.py
import json
import hmac
import hashlib
import logging
import requests

logger = logging.getLogger(__name__)


def manual_send(url, payload, secret=''):
    """
    Send a webhook payload synchronously.
    Returns True if the request was successful (status 2xx), else False.
    """
    headers = {'Content-Type': 'application/json'}
    if secret:
        signature = hmac.new(
            secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        headers['X-Signature'] = signature
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        return resp.status_code < 400
    except Exception as e:
        logger.warning(f"Webhook delivery failed to {url}: {e}")
        return False


def trigger_webhook_event(event_type, note):
    """
    Find all active webhooks subscribed to the given event type
    and send them a notification payload. (Placeholder implementation)
    """
    from .models import WebhookSubscription
    subscriptions = WebhookSubscription.objects.filter(
        events__contains=[event_type], is_active=True
    )
    # Fetch all active subscriptions and filter in Python (avoids SQLite JSON contains issue)
    all_subs = WebhookSubscription.objects.filter(is_active=True)
    subscriptions = [s for s in all_subs if event_type in s.events]

    payload = {
        'event': event_type,
        'note_id': str(note.id),
        'title': note.title,
        'author': note.author.username,
        'timestamp': str(note.updated_at or note.created_at),
    }
    for sub in subscriptions:
        success = manual_send(sub.url, payload, sub.secret)
        if success:
            sub.last_triggered_at = timezone.now()
            sub.save(update_fields=['last_triggered_at'])
            logger.info(f"Webhook delivered to {sub.url}")
        else:
            logger.warning(f"Webhook delivery failed to {sub.url}")