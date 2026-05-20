
from django.db.models.signals import post_save
from django.dispatch import receiver
from experiment.models import TrainingRun
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

@receiver(post_save, sender=TrainingRun)
def broadcast_run_update(sender, instance, created, **kwargs):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'dashboard_{instance.user.id}' if hasattr(instance, 'user') else 'dashboard_all',
        {
            'type': 'dashboard_update',
            'data': {
                'id': str(instance.id),
                'status': instance.status,
                'name': instance.name,
            }
        }
    )