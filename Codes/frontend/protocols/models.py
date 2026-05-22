class ResearchProtocol(models.Model):
    """A reusable experimental protocol."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='protocols')
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    steps = models.JSONField(default=list)
    required_gpu_model = models.CharField(max_length=100, blank=True)
    estimated_duration_hours = models.FloatField(null=True)
    tags = models.JSONField(default=list, blank=True)
    is_public = models.BooleanField(default=False)
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class SharedDataset(models.Model):
    """A pre‑processed dataset shared by the community."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='shared_datasets')
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='shared_datasets/')
    file_size_mb = models.FloatField(default=0)
    task_type = models.CharField(max_length=50, blank=True)
    license = models.CharField(max_length=50, default='MIT')
    download_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

class CodeSnippet(models.Model):
    """A reusable code snippet."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='code_snippets')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    language = models.CharField(max_length=30, default='python')
    code = models.TextField()
    tags = models.JSONField(default=list, blank=True)
    is_public = models.BooleanField(default=True)
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)