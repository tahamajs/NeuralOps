import uuid

from django.conf import settings
from django.db import models

User = settings.AUTH_USER_MODEL


class Publication(models.Model):
    PUBLICATION_TYPES = [
        ("ARTICLE", "Journal Article"),
        ("CONFERENCE", "Conference Paper"),
        ("PREPRINT", "Preprint"),
        ("DATASET", "Dataset"),
        ("SOFTWARE", "Software"),
        ("POSTER", "Poster"),
        ("THESIS", "Thesis"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500)
    authors = models.ManyToManyField(
        User, related_name="authored_publications", through="Authorship"
    )
    abstract = models.TextField()
    publication_type = models.CharField(
        max_length=20, choices=PUBLICATION_TYPES, default="ARTICLE"
    )
    doi = models.CharField(max_length=100, blank=True, null=True)
    arxiv_id = models.CharField(max_length=50, blank=True, null=True)
    pdf_file = models.FileField(upload_to="publications/pdfs/", null=True, blank=True)
    supplementary_files = models.FileField(
        upload_to="publications/supplementary/", null=True, blank=True
    )
    repository_url = models.URLField(blank=True, null=True)
    journal = models.CharField(max_length=200, blank=True, null=True)
    conference = models.CharField(max_length=200, blank=True, null=True)
    publication_date = models.DateField()
    citation_count = models.IntegerField(default=0)
    view_count = models.IntegerField(default=0)
    download_count = models.IntegerField(default=0)
    tags = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-publication_date"]

    def __str__(self):
        return self.title


class Authorship(models.Model):
    publication = models.ForeignKey(Publication, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    author_order = models.PositiveIntegerField(default=0)
    corresponding_author = models.BooleanField(default=False)

    class Meta:
        ordering = ["author_order"]
        unique_together = [["publication", "author"]]


class Citation(models.Model):
    """引用关系"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source = models.ForeignKey(
        Publication, on_delete=models.CASCADE, related_name="citations_out"
    )
    target = models.ForeignKey(
        Publication, on_delete=models.CASCADE, related_name="citations_in"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["source", "target"]]


class Review(models.Model):
    """同行评审"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    publication = models.ForeignKey(
        Publication, on_delete=models.CASCADE, related_name="reviews"
    )
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE, related_name="reviews")
    content = models.TextField()
    rating = models.PositiveSmallIntegerField(default=5, help_text="1-10")
    is_anonymous = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["publication", "reviewer"]]

class ResearchStream(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
