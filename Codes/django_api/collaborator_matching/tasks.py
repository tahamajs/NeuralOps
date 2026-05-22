"""
Celery tasks for embedding generation and match finding.
"""
import logging
from celery import shared_task
from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)


@shared_task
def update_user_embedding(user_id):
    """Regenerate a single user's interest embedding."""
    user = User.objects.get(id=user_id)
    from .embedding import generate_user_embedding
    from .models import ResearcherInterestProfile

    embedding = generate_user_embedding(user)
    if embedding is None:
        return "No papers found for user"

    profile, _ = ResearcherInterestProfile.objects.get_or_create(user=user)
    profile.embedding = embedding
    profile.paper_count = user.saved_papers.count()
    profile.save()
    return f"Updated embedding for {user.username}"


@shared_task
def generate_matches_for_user(user_id):
    """Find collaborator matches for a single user."""
    user = User.objects.get(id=user_id)
    from .models import ResearcherInterestProfile, CollaboratorMatch
    from .embedding import cosine_similarity

    # Ensure the user has a profile
    profile, _ = ResearcherInterestProfile.objects.get_or_create(user=user)
    if not profile.embedding:
        # Regenerate embedding first
        update_user_embedding(user_id)
        profile.refresh_from_db()
        if not profile.embedding:
            return "Could not generate embedding"

    user_emb = profile.embedding

    # Find other users with profiles
    other_profiles = ResearcherInterestProfile.objects.exclude(
        user=user
    ).exclude(
        embedding__exact=[]
    ).select_related('user')

    matches_created = 0
    for other in other_profiles:
        if not other.embedding:
            continue
        score = cosine_similarity(user_emb, other.embedding)
        if score < 0.3:  # minimum threshold
            continue

        # Determine common interests from paper tags
        user_tags = set()
        for paper in user.saved_papers.all():
            if paper.tags:
                user_tags.update(paper.tags)
        other_tags = set()
        for paper in other.user.saved_papers.all():
            if paper.tags:
                other_tags.update(paper.tags)
        common = list(user_tags & other_tags)

        match, created = CollaboratorMatch.objects.update_or_create(
            user=user,
            matched_user=other.user,
            defaults={
                'score': round(score, 4),
                'common_interests': common,
                'explanation': f"Based on similar research interests ({len(common)} common tags)" if common else "Based on semantic similarity of research outputs",
            }
        )
        if created:
            matches_created += 1

    return f"Generated {matches_created} new matches for {user.username}"


@shared_task
def regenerate_all_profiles():
    """Periodic task: regenerate interest profiles for all active users."""
    users = User.objects.filter(is_active=True)
    for user in users:
        update_user_embedding.delay(user.id)
    return f"Regeneration queued for {users.count()} users"


@shared_task
def generate_all_matches():
    """Periodic task: find matches for all active users."""
    users = User.objects.filter(is_active=True)
    for user in users:
        generate_matches_for_user.delay(user.id)
    return f"Match generation queued for {users.count()} users"