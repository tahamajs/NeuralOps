"""
Dashboard widget system – customisable tile configurations.
"""
from django.db import models
import uuid
from django.contrib.auth import get_user_model

User = get_user_model()


class DashboardWidget(models.Model):
    """User-customisable dashboard widget."""
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
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dashboard_widgets')
    widget_type = models.CharField(max_length=30, choices=WIDGET_TYPES)
    title = models.CharField(max_length=100, blank=True)
    config = models.JSONField(default=dict)   # e.g., {"run_ids": [...], "metric": "eval_accuracy"}
    position_x = models.IntegerField(default=0)
    position_y = models.IntegerField(default=0)
    width = models.IntegerField(default=1)     # grid columns
    height = models.IntegerField(default=1)    # grid rows
    is_visible = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['position_y', 'position_x']


DEFAULT_WIDGETS = [
    {"widget_type": "recent_runs", "title": "Recent Runs", "position_x": 0, "position_y": 0, "width": 2, "height": 1},
    {"widget_type": "gpu_monitor", "title": "GPU Monitor", "position_x": 2, "position_y": 0, "width": 1, "height": 1},
    {"widget_type": "run_status", "title": "Run Status", "position_x": 0, "position_y": 1, "width": 1, "height": 1},
    {"widget_type": "notes_feed", "title": "Notes", "position_x": 1, "position_y": 1, "width": 2, "height": 1},
]


# ---- Views ----
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_dashboard_widgets(request):
    widgets = DashboardWidget.objects.filter(user=request.user).order_by('position_y', 'position_x')
    if not widgets.exists():
        # Create defaults for new users
        for w in DEFAULT_WIDGETS:
            DashboardWidget.objects.create(user=request.user, **w)
        widgets = DashboardWidget.objects.filter(user=request.user).order_by('position_y', 'position_x')

    data = [{
        'id': str(w.id), 'widget_type': w.widget_type, 'title': w.title,
        'config': w.config, 'position_x': w.position_x, 'position_y': w.position_y,
        'width': w.width, 'height': w.height, 'is_visible': w.is_visible,
    } for w in widgets]
    return Response(data)


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def save_dashboard_layout(request):
    """Save the entire dashboard layout."""
    widgets_data = request.data.get('widgets', [])
    existing_ids = set(str(w.id) for w in DashboardWidget.objects.filter(user=request.user))
    updated_ids = set()

    for wdata in widgets_data:
        wid = wdata.get('id')
        if wid and wid in existing_ids:
            widget = DashboardWidget.objects.get(id=wid, user=request.user)
            for field in ['position_x', 'position_y', 'width', 'height', 'is_visible', 'config', 'title']:
                if field in wdata:
                    setattr(widget, field, wdata[field])
            widget.save()
            updated_ids.add(wid)
        elif not wid:
            DashboardWidget.objects.create(user=request.user, **{
                k: v for k, v in wdata.items()
                if k in ['widget_type', 'title', 'config', 'position_x', 'position_y', 'width', 'height', 'is_visible']
            })

    # Remove widgets that were deleted by the user
    to_delete = existing_ids - updated_ids
    DashboardWidget.objects.filter(id__in=to_delete).delete()
    return Response({'status': 'saved'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def widget_data(request, widget_id):
    """Fetch live data for a specific widget."""
    widget = get_object_or_404(DashboardWidget, id=widget_id, user=request.user)
    data = _resolve_widget_data(widget)
    return Response(data)


def _resolve_widget_data(widget: DashboardWidget) -> dict:
    """Resolve live data for each widget type."""
    from experiment.models import TrainingRun
    from jobscheduler.models import Job

    if widget.widget_type == 'recent_runs':
        runs = TrainingRun.objects.order_by('-created_at')[:10]
        return {'runs': [{'id': str(r.id), 'name': r.name, 'status': r.status,
                           'created_at': r.created_at.isoformat()} for r in runs]}

    elif widget.widget_type == 'gpu_monitor':
        import torch
        if torch.cuda.is_available():
            return {
                'total': torch.cuda.device_count(),
                'allocated_mb': round(torch.cuda.memory_allocated()/1024**2, 1),
                'reserved_mb': round(torch.cuda.memory_reserved()/1024**2, 1),
            }
        return {'total': 0, 'allocated_mb': 0}

    elif widget.widget_type == 'run_status':
        runs = TrainingRun.objects.all()
        return {
            'total': runs.count(),
            'running': runs.filter(status='running').count(),
            'completed': runs.filter(status='completed').count(),
            'failed': runs.filter(status='failed').count(),
            'queued': runs.filter(status='queued').count(),
        }

    elif widget.widget_type == 'notes_feed':
        from notes.models import ResearcherNote
        notes = ResearcherNote.objects.filter(deleted_at__isnull=True, is_public=True).order_by('-created_at')[:5]
        return {'notes': [{'id': str(n.id), 'title': n.title, 'author': n.author.username,
                            'created_at': n.created_at.isoformat()} for n in notes]}

    elif widget.widget_type == 'loss_chart':
        run_ids = widget.config.get('run_ids', [])
        if not run_ids:
            runs = TrainingRun.objects.filter(status__in=['running', 'completed']).order_by('-created_at')[:1]
            run_ids = [str(r.id) for r in runs]
        history_data = {}
        for rid in run_ids:
            try:
                run = TrainingRun.objects.get(id=rid)
                h = run.training_history or {}
                history_data[rid] = {'steps': h.get('steps', []), 'losses': h.get('losses', [])}
            except TrainingRun.DoesNotExist:
                pass
        return {'history': history_data}

    return {'message': f'No data resolver for widget type: {widget.widget_type}'}