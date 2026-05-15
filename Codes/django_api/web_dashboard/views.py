import json
from datetime import timedelta

from django.shortcuts import render, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Avg, Value as V
from django.db.models.functions import Coalesce, TruncDate
from django.utils import timezone
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

from experiment.models import Experiment, TrainingRun


# ============================================================================
# HTML Page Views
# ============================================================================

def index(request):
    """Main landing page – list of recent runs."""
    runs = TrainingRun.objects.select_related('experiment').order_by('-created_at')[:20]
    return render(request, 'web_dashboard/index.html', {'runs': runs})


def create_job(request):
    """Job creation wizard page."""
    return render(request, 'web_dashboard/create_job.html')


@login_required(login_url='/login/')
def job_detail(request, job_id):
    """Detailed view of a single run (experiment monitor)."""
    run = get_object_or_404(TrainingRun, id=job_id)
    return render(request, 'web_dashboard/experiment_monitor.html', {'run': run})


@login_required(login_url='/login/')
def dashboard(request):
    """Full dashboard (bootstrap version with real‑time updates)."""
    return render(request, 'web_dashboard/dashboard.html')


@login_required(login_url='/login/')
def statistics(request):
    """Statistics page with aggregated metrics and charts."""
    total = TrainingRun.objects.count()
    completed = TrainingRun.objects.filter(status='completed').count()
    failed = TrainingRun.objects.filter(status='failed').count()
    running = TrainingRun.objects.filter(status='running').count()
    success_rate = (completed / total * 100) if total else 0

    # Last 7 days activity
    week_ago = timezone.now() - timedelta(days=7)
    runs_by_day = (
        TrainingRun.objects
        .filter(created_at__gte=week_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )
    days_labels = [item['date'].strftime('%Y-%m-%d') for item in runs_by_day]
    days_counts = [item['count'] for item in runs_by_day]

    # Average metrics (handle SQLite JSON fields gracefully)
    try:
        avg_metrics = TrainingRun.objects.filter(
            status='completed',
            final_metrics__has_key='eval_accuracy'
        ).aggregate(
            avg_loss=Coalesce(Avg('final_metrics__eval_loss'), V(0.0)),
            avg_accuracy=Coalesce(Avg('final_metrics__eval_accuracy'), V(0.0)),
            avg_f1=Coalesce(Avg('final_metrics__eval_f1'), V(0.0)),
        )
    except Exception:
        avg_metrics = {'avg_loss': 0.0, 'avg_accuracy': 0.0, 'avg_f1': 0.0}

    model_usage = TrainingRun.objects.values('config__model_name').annotate(count=Count('id')).order_by('-count')[:5]
    task_usage = TrainingRun.objects.values('config__task_name').annotate(count=Count('id')).order_by('-count')[:5]

    context = {
        'total_experiments': total,
        'completed': completed,
        'failed': failed,
        'running': running,
        'success_rate': round(success_rate, 1),
        'days_labels': json.dumps(days_labels),
        'days_counts': json.dumps(days_counts),
        'avg_loss': round(avg_metrics.get('avg_loss') or 0, 4),
        'avg_accuracy': round(avg_metrics.get('avg_accuracy') or 0, 4),
        'avg_f1': round(avg_metrics.get('avg_f1') or 0, 4),
        'model_usage': model_usage,
        'task_usage': task_usage,
    }
    return render(request, 'web_dashboard/statistics.html', context)


# ============================================================================
# Authentication Pages (public)
# ============================================================================

def login_page(request):
    return render(request, 'web_dashboard/login.html')


def register_page(request):
    return render(request, 'web_dashboard/register.html')


# ============================================================================
# REST API Endpoints (used by the frontend JavaScript)
# ============================================================================

@api_view(["POST"])
@permission_classes([AllowAny])
def api_register(request):
    username = request.data.get("username")
    password = request.data.get("password")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)
    if len(username) > 150:
        return Response({"error": "Username too long"}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)

    user = User.objects.create_user(username=username, password=password)
    token = Token.objects.create(user=user)
    auth_login(request, user)

    response = Response({"token": token.key, "username": user.username}, status=201)
    response.set_cookie(
        key='authToken',
        value=token.key,
        httponly=False,
        samesite='Lax',
        max_age=60*60*24*30,
        path='/'
    )
    return response


@api_view(["POST"])
@permission_classes([AllowAny])
def api_login(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        auth_login(request, user)
        response = Response({"token": token.key, "username": user.username})
        response.set_cookie(
            key='authToken',
            value=token.key,
            httponly=False,
            samesite='Lax',
            max_age=60*60*24*30,
            path='/'
        )
        return response
    return Response({"error": "Invalid credentials"}, status=401)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def api_logout(request):
    if request.auth and hasattr(request.auth, 'delete'):
        request.auth.delete()
    else:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Token "):
            key = auth_header[6:]
            Token.objects.filter(key=key).delete()

    response = Response({"status": "logged out"})
    response.delete_cookie('authToken')
    auth_logout(request)
    return response


@api_view(["GET"])
def dashboard_api_jobs(request):
    """Return recent runs as JSON (used by the dashboard frontend)."""
    runs = TrainingRun.objects.select_related('experiment').order_by('-created_at')[:50]
    data = [{
        'id': str(run.id),
        'status': run.status,
        'experiment_name': run.experiment.name if run.experiment else '',
        'name': run.name,
        'config': run.config,
        'created_at': run.created_at.isoformat(),
        'started_at': run.started_at.isoformat() if run.started_at else None,
        'completed_at': run.completed_at.isoformat() if run.completed_at else None,
        'current_step': run.current_step,
        'total_steps': run.total_steps,
        'eta_seconds': run.estimated_remaining_seconds,
        'gpu_percent': run.gpu_usage,
        'gpu_memory_mb': run.gpu_memory_mb,
        'cpu_percent': run.cpu_usage,
        'ram_mb': run.ram_usage_mb,
    } for run in runs]
    return Response(data)


@api_view(["GET"])
def launch_config_options(request):
    """Return all available options for the launch page dropdowns."""
    from framework.core.registry import (
        ATTENTION_REGISTRY,
        COMPRESSOR_REGISTRY,
        TRAINER_REGISTRY,
    )
    from framework.trainer.methods.registry import METHOD_REGISTRY
    from framework.trainer.losses import LOSS_REGISTRY
    from framework.trainer.optimizers import OPTIMIZER_REGISTRY
    from framework.trainer.schedulers import SCHEDULER_REGISTRY
    from framework.callbacks import CALLBACK_REGISTRY
    from framework.data.augmentation import AUGMENTATION_REGISTRY

    return Response({
        "attentions": list(ATTENTION_REGISTRY.keys()),
        "compressors": list(COMPRESSOR_REGISTRY.keys()),
        "trainers": list(TRAINER_REGISTRY.keys()),
        "methods": list(METHOD_REGISTRY.keys()),
        "losses": list(LOSS_REGISTRY.keys()),
        "optimizers": list(OPTIMIZER_REGISTRY.keys()),
        "schedulers": list(SCHEDULER_REGISTRY.keys()),
        "callbacks": list(CALLBACK_REGISTRY.keys()),
        "augmentations": list(AUGMENTATION_REGISTRY.keys()),
        "model_variants": ["standard", "unsloth", "compressed", "hybrid_svd", "quantized", "lora"],
        "distributed_strategies": ["none", "ddp", "deepspeed", "fsdp"],
    })


# ============================================================================
# Widget & Dashboard Helper Views (JSON endpoints)
# ============================================================================

def get_dashboard_widgets(request):
    """Return available widget metadata."""
    return JsonResponse({
        "widgets": [
            {"id": "recent_experiments", "title": "Recent Experiments", "icon": "flask", "configurable": True},
            {"id": "running_jobs", "title": "Running Jobs", "icon": "play", "configurable": False},
            {"id": "system_status", "title": "System Status", "icon": "desktop", "configurable": True},
            {"id": "gpu_monitor", "title": "GPU Monitor", "icon": "chip", "configurable": False},
        ]
    })


def save_dashboard_layout(request):
    """Save user's dashboard widget layout (positions, sizes)."""
    if request.method == "POST":
        # Expecting JSON like {"layout": [...]}
        # For now, just echo success.
        return JsonResponse({"status": "ok", "message": "Layout saved"})
    return JsonResponse({"error": "POST required"}, status=405)


def get_dashboard_settings(request):
    """Return dashboard user settings."""
    return JsonResponse({
        "settings": {
            "theme": "dark",
            "refresh_interval": 5,   # seconds
            "auto_refresh": True,
        }
    })


def save_dashboard_settings(request):
    """Save dashboard user settings."""
    if request.method == "POST":
        return JsonResponse({"status": "ok", "message": "Settings saved"})
    return JsonResponse({"error": "POST required"}, status=405)


def get_widget_data(request, widget_id):
    """Return data for a specific widget (used by refresh endpoint)."""
    # Simple placeholder: return different data per widget type
    if widget_id == "recent_experiments":
        runs = TrainingRun.objects.order_by('-created_at')[:5].values('name', 'status', 'created_at')
        data = list(runs)
    elif widget_id == "running_jobs":
        jobs = TrainingRun.objects.filter(status='running').count()
        data = {"count": jobs}
    elif widget_id == "gpu_monitor":
        # Simulate GPU info (can be expanded with real NVIDIA/MPS stats)
        data = {"gpu_util": 0, "memory_used": 0, "memory_total": 0}
    else:
        data = {}
    return JsonResponse({"widget_id": widget_id, "data": data})


def refresh_widget(request, widget_id):
    """Alias for get_widget_data – used for manual refresh."""
    return get_widget_data(request, widget_id)


def widget_data(request, widget_id):
    """Same as get_widget_data – URL pattern uses this name."""
    return get_widget_data(request, widget_id)


def widget_config(request, widget_id):
    """Return configuration options for a widget (if configurable)."""
    configs = {
        "recent_experiments": {"limit": 5, "show_metrics": True},
        "system_status": {"show_cpu": True, "show_memory": True},
    }
    return JsonResponse(configs.get(widget_id, {}))


def dashboard_stats(request):
    """Return aggregated dashboard statistics (used by some frontend components)."""
    total = TrainingRun.objects.count()
    running = TrainingRun.objects.filter(status='running').count()
    completed = TrainingRun.objects.filter(status='completed').count()
    failed = TrainingRun.objects.filter(status='failed').count()
    return JsonResponse({
        "total_runs": total,
        "running": running,
        "completed": completed,
        "failed": failed,
        "success_rate": round(completed / total * 100, 1) if total else 0,
    })