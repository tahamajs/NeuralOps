from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from api.models import Experiment
from api.views import start_train, start_evaluate, start_compress, job_status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json

def index(request):
    """Main dashboard"""
    experiments = Experiment.objects.all().order_by('-created_at')
    return render(request, 'web_dashboard/index.html', {'experiments': experiments})

def job_detail(request, job_id):
    """Detailed job view with live training charts"""
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return redirect('index')
    return render(request, 'web_dashboard/job_detail.html', {'experiment': exp})

def create_job(request):
    """Job creation wizard"""
    return render(request, 'web_dashboard/create_job.html')

# API for dashboard (called by JavaScript)
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