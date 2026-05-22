"""Celery tasks for reputation management."""
from celery import shared_task
from django.utils import timezone
from .models import ReputationScore, ReputationSnapshot, Recommendation


@shared_task
def daily_snapshot():
    """Take a daily snapshot of all reputation scores."""
    today = timezone.now().date()
    scores = ReputationScore.objects.select_related('user').all()
    count = 0

    for score in scores:
        _, created = ReputationSnapshot.objects.get_or_create(
            user=score.user, recorded_at=today,
            defaults={
                'total_score': score.total_score,
                'other_reads_score': score.other_reads_score,
                'full_text_reads_score': score.full_text_reads_score,
                'recommendations_score': score.recommendations_score,
                'citations_score': score.citations_score,
                'percentile': score.percentile,
            }
        )
        if created:
            count += 1

    # Update percentiles
    total = ReputationScore.objects.count()
    for i, score in enumerate(ReputationScore.objects.order_by('-total_score'), 1):
        score.percentile = round((1 - i / total) * 100, 2)
        score.save(update_fields=['percentile'])

    return count


@shared_task
def generate_daily_recommendations():
    """Generate recommendations for all active users."""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    from agents.models import ResearchPaper

    users = User.objects.filter(is_active=True)
    total_created = 0

    for user in users:
        saved_papers = ResearchPaper.objects.filter(saved_by=user).values_list('id', flat=True)
        if not saved_papers:
            continue

        similar_users = ResearchPaper.objects.filter(
            id__in=saved_papers
        ).values_list('saved_by', flat=True).distinct()

        recommended = ResearchPaper.objects.filter(
            saved_by__in=similar_users
        ).exclude(
            id__in=saved_papers
        ).distinct().order_by('-citation_count')[:10]

        for paper in recommended:
            shared_count = ResearchPaper.objects.filter(
                id__in=saved_papers, saved_by__in=[user]
            ).count()
            score = min(1.0, (paper.citation_count or 0) / 1000.0 + shared_count * 0.1)
            rec, created = Recommendation.objects.get_or_create(
                user=user, paper=paper,
                defaults={'score': round(score, 2), 'reason': 'Based on your saved papers and interests'}
            )
            if created:
                total_created += 1

    return total_created