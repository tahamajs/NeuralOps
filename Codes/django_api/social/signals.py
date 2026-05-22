from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Connection, ActivityFeed, PublicationAlert


@receiver(post_save, sender=Connection)
def on_connection_accepted(sender, instance, created, **kwargs):
    if instance.status == 'ACCEPTED' and instance.connection_type == 'FOLLOW':
        ActivityFeed.objects.get_or_create(
            user=instance.from_user,
            activity_type='CONNECTION',
            title=f"Started following {instance.to_user.username}",
        )