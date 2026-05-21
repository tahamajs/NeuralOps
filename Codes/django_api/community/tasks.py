from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from .models import Question, DigestSubscription

@shared_task
def send_weekly_digest():
    """Send trending questions digest to subscribed users."""
    week_ago = timezone.now() - timedelta(days=7)
    trending = Question.objects.filter(
        created_at__gte=week_ago, status='open'
    ).order_by('-score')[:10]

    for sub in DigestSubscription.objects.filter(frequency='weekly'):
        body = "🔥 Trending This Week on NeuralOps Community\n\n"
        for q in trending:
            body += f"• {q.title} (+{q.score}) — {q.answer_count} answers\n"
        body += f"\nVisit {settings.PUBLIC_BASE_URL}/community to participate.\n"

        send_mail(
            subject="📊 Weekly Community Digest — Trending Questions",
            message=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[sub.user.email],
            fail_silently=True,
        )
        sub.last_sent_at = timezone.now()
        sub.save(update_fields=['last_sent_at'])