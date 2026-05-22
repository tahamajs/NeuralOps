import uuid
from django.db import models
from django.conf import settings


class ExpertProfile(models.Model):
    """
    Extended profile for a researcher's expertise.
    Automatically generated / updated from publication history.
    """
    EXPERTISE_LEVEL_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
        ('expert', 'Expert'),
        ('world_class', 'World-Class'),
    ]
    COLLABORATION_PREFERENCE = [
        ('open', 'Open to collaboration'),
        ('selective', 'Selective'),
        ('not_available', 'Not Available'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='expert_profile'
    )
    skills = models.JSONField(default=list, blank=True)
    domains = models.JSONField(default=list, blank=True)
    expertise_level = models.CharField(
        max_length=20, choices=EXPERTISE_LEVEL_CHOICES, default='intermediate'
    )
    collaboration_preference = models.CharField(
        max_length=20, choices=COLLABORATION_PREFERENCE, default='open'
    )
    seeking_collaborators = models.BooleanField(default=False)
    bio = models.TextField(blank=True)
    orcid = models.CharField(max_length=20, blank=True)
    verified = models.BooleanField(default=False)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-expertise_level']

    def __str__(self):
        return f"Expert profile of {self.user.username}"


class Project(models.Model):
    """
    A research project seeking collaborators.
    """
    PROJECT_TYPE_CHOICES = [
        ('research', 'Research Project'),
        ('grant', 'Grant Proposal'),
        ('paper', 'Paper / Publication'),
        ('software', 'Software / Tool'),
        ('dataset', 'Dataset Creation'),
        ('workshop', 'Workshop / Event'),
    ]
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('active', 'Actively Recruiting'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('closed', 'Closed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='owned_projects'
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    project_type = models.CharField(max_length=20, choices=PROJECT_TYPE_CHOICES, default='research')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    skills_needed = models.JSONField(default=list, blank=True)
    domains_needed = models.JSONField(default=list, blank=True)
    collaborators_needed = models.PositiveIntegerField(default=1)
    current_collaborators = models.PositiveIntegerField(default=0)
    deadline = models.DateTimeField(null=True, blank=True)
    is_public = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title


class ProjectApplication(models.Model):
    """
    An application from a researcher to join a project.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='applications'
    )
    applicant = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='project_applications'
    )
    message = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['project', 'applicant']]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.applicant.username} → {self.project.title}"


class ExpertReview(models.Model):
    """
    A request for an expert to review a research proposal, paper, or draft.
    """
    REVIEW_TYPE_CHOICES = [
        ('proposal', 'Grant Proposal'),
        ('paper', 'Paper / Draft'),
        ('methodology', 'Methodology'),
        ('code', 'Code Review'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('completed', 'Completed'),
        ('declined', 'Declined'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    requester = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='requested_reviews'
    )
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='assigned_reviews'
    )
    review_type = models.CharField(max_length=20, choices=REVIEW_TYPE_CHOICES)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    attachment_url = models.URLField(blank=True, max_length=500)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    reviewer_feedback = models.TextField(blank=True)
    rating = models.PositiveSmallIntegerField(null=True, blank=True, help_text="1-5 stars")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Review: {self.title}"