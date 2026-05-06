# web_dashboard/views.py
import json
from datetime import timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Avg, Value as V
from django.db.models.functions import Coalesce
from django.utils import timezone
from django.http import JsonResponse, HttpResponse
from django.views.decorators.cache import cache_page
from django.views.decorators.csrf import csrf_exempt

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

from experiment.models import Experiment, TrainingRun

# ---------- Web Page Views ----------
def index(request):
    """Main landing page (list of recent runs)."""
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
    """Statistics page with aggregated metrics."""
    total = TrainingRun.objects.count()
    completed = TrainingRun.objects.filter(status='completed').count()
    failed = TrainingRun.objects.filter(status='failed').count()
    running = TrainingRun.objects.filter(status='running').count()
    success_rate = (completed / total * 100) if total else 0

    # Last 7 days activity
    from django.db.models.functions import TruncDate
    week_ago = timezone.now() - timedelta(days=7)
    runs_by_day = (
        TrainingRun.objects
        .filter(created_at__gte=week_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )
    days_labels = [item['date'].strftime('%Y-%m-%d') for item in runs_by_day]  # now works
    days_counts = [item['count'] for item in runs_by_day]

    # Average metrics – use a safe fallback for SQLite JSON fields
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
# ---------- Authentication Pages ----------
def login_page(request):
    return render(request, 'web_dashboard/login.html')

def register_page(request):
    return render(request, 'web_dashboard/register.html')

# ---------- API Endpoints for Frontend (token + cookie) ----------
@api_view(["POST"])
@permission_classes([AllowAny])
def api_register(request):
    username = request.data.get("username")
    password = request.data.get("password")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)
    user = User.objects.create_user(username=username, password=password)
    token = Token.objects.create(user=user)
    auth_login(request, user)   # also log in session
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
def api_logout(request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Token "):
        key = auth_header[6:]
        Token.objects.filter(key=key).delete()
    response = Response({"status": "logged out"})
    response.delete_cookie('authToken')
    auth_logout(request)
    return response

# ---------- API for Dashboard (runs list) ----------
@api_view(["GET"])
def dashboard_api_jobs(request):
    """Return recent runs as JSON (used by dashboard)."""
    runs = TrainingRun.objects.select_related('experiment').order_by('-created_at')[:50]
    data = [{
        'id': str(run.id),
        'status': run.status,
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