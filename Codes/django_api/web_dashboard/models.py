import uuid
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class DashboardWidget(models.Model):
    WIDGET_TYPES = [
        ('recent_runs', 'Recent Runs'),
        ('gpu_monitor', 'GPU Monitor'),
        ('loss_chart', 'Loss Chart'),
        ('accuracy_chart', 'Accuracy Chart'),
        ('metric_comparison', 'Metric Comparison'),
        ('run_status', 'Run Status Overview'),
        ('notes_feed', 'Notes Feed'),
        ('model_registry', 'Model Registry'),
        ('scheduler_queue', 'Scheduler Queue'),
        ('custom_plot', 'Custom Plot'),
        ('terminal', 'Terminal'),
        ('sweep_monitor', 'Sweep Monitor'),
        ('experiment_comparison', 'Experiment Comparison'),
        ('system_health', 'System Health'),
        ('live_log', 'Live Log'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dashboard_widgets')
    widget_type = models.CharField(max_length=30, choices=WIDGET_TYPES)
    title = models.CharField(max_length=100, blank=True)
    config = models.JSONField(default=dict)
    position_x = models.IntegerField(default=0)
    position_y = models.IntegerField(default=0)
    width = models.IntegerField(default=1)
    height = models.IntegerField(default=1)
    is_visible = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['position_y', 'position_x']