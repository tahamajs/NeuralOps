# profiles/models.py
import uuid
from django.db import models
from django.conf import settings


def avatar_upload_to(instance, filename):
    ext = filename.rsplit('.', 1)[-1]
    return f'avatars/{instance.user.id}.{ext}'


class Profile(models.Model):
    """Extended user profile – one-to-one with Django User."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile'
    )

    # Basic info
    display_name = models.CharField(max_length=100, blank=True)
    bio = models.TextField(max_length=500, blank=True)
    website = models.URLField(max_length=300, blank=True)
    location = models.CharField(max_length=150, blank=True)
    avatar = models.ImageField(upload_to=avatar_upload_to, blank=True, null=True)

    # Professional / academic
    affiliation = models.CharField(max_length=250, blank=True)
    research_interests = models.JSONField(default=list, blank=True)
    skills = models.JSONField(default=list, blank=True)

    # Social links
    github = models.URLField(max_length=300, blank=True)
    twitter = models.URLField(max_length=300, blank=True)
    linkedin = models.URLField(max_length=300, blank=True)
    google_scholar = models.URLField(max_length=300, blank=True)
    orcid = models.CharField(max_length=20, blank=True)

    # Stats
    total_experiments = models.PositiveIntegerField(default=0)
    total_compressions = models.PositiveIntegerField(default=0)
    total_papers = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['user__username']

    def __str__(self):
        return f"Profile of {self.user.username}"

    @property
    def avatar_url(self):
        if self.avatar:
            return self.avatar.url
        return None