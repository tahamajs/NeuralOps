"""Auto-update cluster stats when nodes change."""

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import FederationNode


@receiver(post_save, sender=FederationNode)
@receiver(post_delete, sender=FederationNode)
def update_cluster_resources(sender, instance, **kwargs):
    instance.cluster.recalculate_resources()
