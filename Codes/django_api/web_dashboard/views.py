import json
from datetime import timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Avg
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.http import JsonResponse

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

from api.models import Experiment

# ---------- Web Page Views ----------
def index(request):
    experiments = Experiment.objects.only(
        'id', 'status', 'config', 'current_step', 'total_steps',
        'estimated_remaining_seconds', 'created_at'
    ).order_by('-created_at')[:20]
    return render(request, 'web_dashboard/index.html', {'experiments': experiments})

def create_job(request):
    """Job creation wizard (form)"""
    return render(request, 'web_dashboard/create_job.html')

#@login_required(login_url='/login/')
from api.models import TrainingRun

def job_detail(request, job_id):
    run = get_object_or_404(TrainingRun, id=job_id)
    return render(request, 'web_dashboard/job_detail_minimal.html', {'run': run})
#@login_required(login_url='/login/')
def dashboard(request):
    """Alternative dashboard (bootstrap version)"""
    return render(request, 'web_dashboard/dashboard.html')

#@login_required(login_url='/login/')
def statistics(request):
    """Statistics page – login required, shows aggregated metrics"""
    total_experiments = Experiment.objects.count()
    completed = Experiment.objects.filter(status='completed').count()
    failed = Experiment.objects.filter(status='failed').count()
    running = Experiment.objects.filter(status='running').count()
    success_rate = (completed / total_experiments * 100) if total_experiments else 0

    week_ago = timezone.now() - timedelta(days=7)
    experiments_by_day = Experiment.objects.filter(created_at__gte=week_ago) \
        .extra({'date': "date(created_at)"}) \
        .values('date') \
        .annotate(count=Count('id')) \
        .order_by('date')
    days_labels = [item['date'].strftime('%Y-%m-%d') for item in experiments_by_day]
    days_counts = [item['count'] for item in experiments_by_day]

    avg_metrics = Experiment.objects.filter(status='completed', config__has_key='final_metrics') \
        .aggregate(
            avg_loss=Avg('config__final_metrics__eval_loss'),
            avg_accuracy=Avg('config__final_metrics__eval_accuracy'),
            avg_f1=Avg('config__final_metrics__eval_f1'),
        )

    model_usage = Experiment.objects.values('config__model_name').annotate(count=Count('id')).order_by('-count')[:5]
    task_usage = Experiment.objects.values('config__task_name').annotate(count=Count('id')).order_by('-count')[:5]

    context = {
        'total_experiments': total_experiments,
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

# ---------- Login / Register Pages ----------
def login_page(request):
    return render(request, 'web_dashboard/login.html')

def register_page(request):
    return render(request, 'web_dashboard/register.html')

# ---------- API Authentication Endpoints (Token + Cookie) ----------
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
    auth_login(request, user)   # create session
    response = Response({"token": token.key, "username": user.username}, status=201)
    # Set cookie for token-based page auth
    response.set_cookie(
        key='authToken',
        value=token.key,
        httponly=False,
        samesite='Lax',
        max_age=60*60*24*30,
        path='/'
    )
    return response
from django.contrib.auth.views import LoginView

class CustomLoginView(LoginView):
    template_name = 'web_dashboard/login.html'
    redirect_authenticated_user = True

    def get_success_url(self):
        next_url = self.request.GET.get('next')
        if next_url:
            return next_url
        return '/'
@api_view(["POST"])
@permission_classes([AllowAny])
def api_login(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)
    if user is not None:
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
    # Also delete cookie
    response = Response({"status": "logged out"})
    response.delete_cookie('authToken')
    auth_logout(request)
    return response

# ---------- API for Dashboard Data ----------
@api_view(["GET"])
def dashboard_api_jobs(request):
    experiments = Experiment.objects.all().order_by('-created_at')[:20]
    data = []
    for exp in experiments:
        data.append({
            'id': str(exp.id),
            'status': exp.status,
            'config': exp.config,
            'created_at': exp.created_at.isoformat(),
            'started_at': exp.started_at.isoformat() if exp.started_at else None,
            'completed_at': exp.completed_at.isoformat() if exp.completed_at else None,
            'current_step': exp.current_step,
            'total_steps': exp.total_steps,
            'eta_seconds': exp.estimated_remaining_seconds,
            'gpu_percent': exp.gpu_usage,
            'gpu_memory_mb': exp.gpu_memory_mb,
            'cpu_percent': exp.cpu_usage,
            'ram_mb': exp.ram_usage_mb,
        })
    return Response(data)



from api.models import TrainingRun

def job_detail(request, job_id):
    run = get_object_or_404(TrainingRun, id=job_id)
    return render(request, 'web_dashboard/job_detail_minimal.html', {'run': run})




