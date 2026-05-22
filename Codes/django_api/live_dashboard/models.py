import uuid
from django.db import models
from django.conf import settings


class Dashboard(models.Model):
    """
    A custom dashboard created by a user.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='dashboards'
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.name


class DashboardWidget(models.Model):
    """
    A single widget on a dashboard.
    """
    WIDGET_TYPES = [
        ('line_chart', 'Line Chart'),
        ('bar_chart', 'Bar Chart'),
        ('scatter_plot', 'Scatter Plot'),
        ('metric_card', 'Metric Card'),
        ('table', 'Table'),
        ('sql_query', 'SQL Query Result'),
        ('live_feed', 'Live Experiment Feed'),
        ('gpu_monitor', 'GPU Monitor'),
        ('notes_feed', 'Notes Feed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dashboard = models.ForeignKey(
        Dashboard,
        on_delete=models.CASCADE,
        related_name='widgets'
    )
    widget_type = models.CharField(max_length=30, choices=WIDGET_TYPES)
    title = models.CharField(max_length=200, blank=True)
    config = models.JSONField(default=dict)
    position_x = models.IntegerField(default=0)
    position_y = models.IntegerField(default=0)
    width = models.IntegerField(default=2)
    height = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['position_y', 'position_x']

    def __str__(self):
        return f"{self.title or self.widget_type}"


class SavedSQLQuery(models.Model):
    """
    A read‑only SQL query saved by a user for dashboard use.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='saved_sql_queries'
    )
    name = models.CharField(max_length=200)
    sql = models.TextField(help_text="Read‑only SQL query")
    description = models.TextField(blank=True)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name