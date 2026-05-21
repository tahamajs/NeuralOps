"""Auto-update reputation scores and award badges on marketplace events."""
import logging
from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)


@receiver(post_save, sender='gpu_marketplace.ComputeJob')
def update_reputation_on_job_complete(sender, instance, **kwargs):
    """When a job completes or fails, update both parties' reputation scores."""
    if instance.status in ('completed', 'failed'):
        from .models import ReputationScore
        success = (instance.status == 'completed')

        # Provider score
        prov_score, _ = ReputationScore.objects.get_or_create(user=instance.provider)
        prov_score.update_provider_score(success)

        # Consumer score
        cons_score, _ = ReputationScore.objects.get_or_create(user=instance.consumer)
        cons_score.update_consumer_score(payment_on_time=success)

    if instance.status == 'disputed':
        from .models import ReputationScore
        # Downgrade both temporarily (resolved by dispute outcome later)
        for user in (instance.provider, instance.consumer):
            score, _ = ReputationScore.objects.get_or_create(user=user)
            score.provider_score = max(0, score.provider_score - 2)
            score.save(update_fields=['provider_score'])