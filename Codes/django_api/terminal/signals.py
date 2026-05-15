# terminal/signals.py
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from .models import TerminalSession

@receiver(pre_delete, sender='auth.User')
def cleanup_user_terminals(sender, instance, **kwargs):
    """When a user is deleted, kill all their active terminal sessions."""
    for session in TerminalSession.objects.filter(user=instance, is_active=True):
        session.mark_inactive()