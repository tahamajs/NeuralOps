import logging
from celery import shared_task
from django.contrib.auth import get_user_model
from .models import ExpertProfile

logger = logging.getLogger(__name__)
User = get_user_model()

@shared_task(bind=True)
def auto_generate_expert_profile(self, user_id):
    """
    Background task to auto-generate / update an expert profile for a researcher.
    """
    try:
        user = User.objects.get(id=user_id)
        profile, created = ExpertProfile.objects.get_or_create(user=user)
        
        # Pull saved papers or other criteria
        saved_papers_count = user.saved_papers.count() if hasattr(user, 'saved_papers') else 0
        
        # Simple dynamic heuristics based on activity
        if saved_papers_count > 10:
            profile.expertise_level = "expert"
        elif saved_papers_count > 5:
            profile.expertise_level = "advanced"
        else:
            profile.expertise_level = "intermediate"
            
        profile.save()
        logger.info(f"Successfully updated expert profile for user {user.username}")
        return f"Updated profile for user {user.username}"
    except Exception as e:
        logger.error(f"Error generating expert profile: {str(e)}")
        return f"Error: {str(e)}"
