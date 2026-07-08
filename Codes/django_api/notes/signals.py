import re

from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import NoteMention, Notification, ResearcherNote
from .webhook_utils import trigger_webhook_event


@receiver(post_save, sender=ResearcherNote)
def note_post_save(sender, instance, created, **kwargs):
    """Handle note creation/update events."""
    if created:
        trigger_webhook_event("note.created", instance)
        _parse_mentions(instance)
    else:
        trigger_webhook_event("note.updated", instance)
        _parse_mentions(instance)


def _parse_mentions(note: ResearcherNote):
    """Parse @username mentions in note content and create notifications."""
    mentioned = re.findall(r"@(\w+)", note.content)
    for username in set(mentioned):
        try:
            user = User.objects.get(username=username)
            if user != note.author:
                mention, created = NoteMention.objects.get_or_create(
                    note=note, mentioned_user=user
                )
                if created:
                    Notification.objects.create(
                        recipient=user,
                        sender=note.author,
                        type="mention",
                        note=note,
                        content=f"You were mentioned in note: {note.title or 'Untitled'}",
                    )
        except User.DoesNotExist:
            pass
