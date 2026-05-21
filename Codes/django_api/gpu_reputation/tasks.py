"""Celery tasks for reputation — EigenTrust computation, badge awards, dispute expiry."""
import logging
from celery import shared_task
from django.utils import timezone
from .models import ReputationScore, ReputationBadge, UserReputationBadge
from .scoring import build_reputation_graph, compute_eigentrust

logger = logging.getLogger(__name__)


@shared_task
def compute_daily_trust_graph():
    """Build and compute the EigenTrust reputation graph daily."""
    graph = build_reputation_graph(name=f"daily-{timezone.now().strftime('%Y%m%d')}")
    results = compute_eigentrust(graph)
    logger.info(f"Computed EigenTrust for {len(results)} users")
    return len(results)


@shared_task
def auto_award_badges():
    """Check all users and award reputation badges based on milestones."""
    scores = ReputationScore.objects.all()
    badges = ReputationBadge.objects.all()
    awarded = 0

    for score in scores:
        for badge in badges:
            # Check conditions
            conditions_met = True
            if badge.min_provider_score is not None and score.provider_score < badge.min_provider_score:
                conditions_met = False
            if badge.min_consumer_score is not None and score.consumer_score < badge.min_consumer_score:
                conditions_met = False
            if badge.min_jobs_completed is not None and score.total_jobs_completed < badge.min_jobs_completed:
                conditions_met = False
            if badge.max_disputes_lost is not None and score.total_disputes_lost > badge.max_disputes_lost:
                conditions_met = False

            if conditions_met:
                _, created = UserReputationBadge.objects.get_or_create(
                    user=score.user, badge=badge
                )
                if created:
                    awarded += 1

    return awarded


@shared_task
def expire_stale_disputes():
    """Auto-close disputes that have been open for more than 30 days."""
    from .models import Dispute
    cutoff = timezone.now() - timezone.timedelta(days=30)
    expired = Dispute.objects.filter(
        status__in=['open', 'evidence_gathering'], created_at__lt=cutoff
    ).update(status='withdrawn', resolution_reason='Auto-expired after 30 days')
    return expired