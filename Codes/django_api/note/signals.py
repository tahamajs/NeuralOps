# notes/signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import ResearcherNote
from .webhook_utils import trigger_webhook_event

@receiver(post_save, sender=ResearcherNote)
def note_post_save(sender, instance, created, **kwargs):
    if created:
        trigger_webhook_event('note.created', instance)
    else:
        trigger_webhook_event('note.updated', instance)

@receiver(post_delete, sender=ResearcherNote)
def note_post_delete(sender, instance, **kwargs):
    trigger_webhook_event('note.deleted', instance)