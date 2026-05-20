import json
import logging
from django.dispatch import Signal
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth import get_user_model

from .models import ModelVersion, ModelVersionDiff
from .services import migrate_storage_tier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Custom signals
# ------------------------------------------------------------------
version_created = Signal()          # args: version, user
version_published = Signal()        # args: version, user, hub_url
version_archived = Signal()          # args: version, user, storage_tier
version_restored = Signal()          # args: version, user
version_set_production = Signal()    # args: version, user
version_diff_created = Signal()      # args: diff, user

# ------------------------------------------------------------------
# Activity logging helper
# ------------------------------------------------------------------
def _log_activity(user, action_type, description, content_object=None, metadata=None):
    """Create an ActivityLog entry if the app is installed."""
    try:
        from experiment.activity import log_activity
        log_activity(
            user=user,
            action_type=action_type,
            description=description,
            content_object=content_object,
            metadata=metadata or {},
        )
    except ImportError:
        pass

# ------------------------------------------------------------------
# Email notification helper
# ------------------------------------------------------------------
def _send_notification(subject: str, message: str, recipient_list: list = None):
    if not recipient_list:
        recipient_list = [settings.DEFAULT_FROM_EMAIL]
    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=recipient_list,
            fail_silently=True,
        )
    except Exception as e:
        logger.warning(f"Failed to send notification email: {e}")

# ------------------------------------------------------------------
# Webhook helper (optional, URL defined in settings)
# ------------------------------------------------------------------
def _trigger_webhook(event: str, payload: dict):
    url = getattr(settings, 'MODEL_VERSIONING_WEBHOOK_URL', None)
    if not url:
        return
    try:
        import requests
        requests.post(url, json={'event': event, 'data': payload}, timeout=5)
    except Exception as e:
        logger.warning(f"Webhook failed: {e}")

# ------------------------------------------------------------------
# Default receiver: auto‑migrate storage tier on save
# ------------------------------------------------------------------
@receiver(post_save, sender=ModelVersion)
def auto_migrate_on_save(sender, instance, created, **kwargs):
    if not created:
        return
    # Asynchronously trigger migration (Celery task would be better, but for simplicity)
    try:
        migrate_storage_tier(instance)
    except Exception as e:
        logger.warning(f"Auto‑migration failed for version {instance.id}: {e}")

# ------------------------------------------------------------------
# Handlers for custom signals
# ------------------------------------------------------------------

@receiver(version_created)
def on_version_created(sender, version, user, **kwargs):
    _log_activity(user, 'create', f'Created model version {version.version}', version)
    _send_notification(
        f'New Model Version Created: {version.version}',
        f'Run: {version.run.name}\nVersion: {version.version}\nMetrics: {json.dumps(version.metrics)}'
    )
    _trigger_webhook('version_created', {
        'version_id': str(version.id),
        'run_id': str(version.run_id),
        'version': version.version,
    })

@receiver(version_published)
def on_version_published(sender, version, user, hub_url, **kwargs):
    _log_activity(user, 'publish', f'Published version {version.version} to HuggingFace Hub', version,
                  {'hub_url': hub_url})
    _send_notification(
        f'Model Version Published: {version.version}',
        f'Version {version.version} is now available at {hub_url}'
    )
    _trigger_webhook('version_published', {
        'version_id': str(version.id),
        'hub_url': hub_url,
    })

@receiver(version_archived)
def on_version_archived(sender, version, user, storage_tier, **kwargs):
    _log_activity(user, 'update', f'Archived version {version.version} to {storage_tier}', version,
                  {'storage_tier': storage_tier})
    _trigger_webhook('version_archived', {
        'version_id': str(version.id),
        'storage_tier': storage_tier,
    })

@receiver(version_restored)
def on_version_restored(sender, version, user, **kwargs):
    _log_activity(user, 'update', f'Restored version {version.version} from cold storage', version)
    _trigger_webhook('version_restored', {'version_id': str(version.id)})

@receiver(version_set_production)
def on_version_set_production(sender, version, user, **kwargs):
    _log_activity(user, 'update', f'Set version {version.version} as production', version)
    _send_notification(
        f'Production Version Changed: {version.version}',
        f'Version {version.version} is now the production model for run {version.run.name}.'
    )
    _trigger_webhook('version_set_production', {'version_id': str(version.id)})

@receiver(version_diff_created)
def on_version_diff_created(sender, diff, user, **kwargs):
    _log_activity(user, 'create', f'Created diff between {diff.from_version.version} and {diff.to_version.version}', diff)
    _trigger_webhook('version_diff_created', {
        'diff_id': str(diff.id),
        'from_version': str(diff.from_version_id),
        'to_version': str(diff.to_version_id),
    })