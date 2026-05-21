# profiles/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings
from .models import Profile
import logging

logger = logging.getLogger(__name__)

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_user_profile(sender, instance, created, **kwargs):
    """Automatically create a Profile when a new User is created."""
    if created:
        try:
            Profile.objects.get_or_create(user=instance)
        except Exception as e:
            logger.warning("Could not create profile for user %s: %s", instance.username, e)