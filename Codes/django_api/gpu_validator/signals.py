"""
Automatically create verification tasks when compute jobs complete.
"""
import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import VerificationTask, ProofOfCompute

logger = logging.getLogger(__name__)


@receiver(post_save, sender='gpu_marketplace.ComputeJob')
def create_verification_task_on_completion(sender, instance, created, **kwargs):
    """
    When a ComputeJob is completed, create a VerificationTask.
    """
    if instance.status == 'completed' and not hasattr(instance, 'verification_task'):
        task = VerificationTask.objects.create(
            job=instance,
            status='pending',
            required_validators=3,  # configurable
        )
        logger.info(f"Verification task created for job {instance.id}")

        # Also create a ProofOfCompute placeholder if not exists
        ProofOfCompute.objects.get_or_create(job=instance)