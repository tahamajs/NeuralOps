# experiments/views.py
import os
import json
import tempfile
import zipfile
import psutil
import torch
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from django.db.models import F
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import Experiment, TrainingRun, TrainingSnapshot
from .serializers import ExperimentSerializer, TrainingRunSerializer, TrainingSnapshotSerializer
from .jobs import start_job, stop_job, stop_all_jobs, push_update
from .runners import run_training, run_evaluation, run_compression

# ----------------------------------------------------------------------
# Config endpoints (sample – replace with real registries)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_attentions(request):
    # Example: return a static list; replace with actual ATTENTION_REGISTRY
    return Response({"attentions": ["full", "goldfinch_ahn", "granite_hybrid", "griffin_ssd"]})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_models(request):
    return Response({"models": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_tasks(request):
    return Response({"tasks": ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "stsb"]})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_compressors(request):
    return Response({"compressors": ["slicegpt", "sparsegpt", "adasvd"]})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_methods(request):
    return Response({"methods": ["baseline", "lora", "rdrop", "r3f"]})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_presets(request):
    return Response({"presets": ["fast", "accurate", "extreme_compress"]})

# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def list_experiments(request):
    if request.method == 'GET':
        experiments = Experiment.objects.all().order_by('-created_at')
        serializer = ExperimentSerializer(experiments, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = ExperimentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_experiment(request):
    serializer = ExperimentSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def experiment_detail(request, exp_id):
    exp = get_object_or_404(Experiment, id=exp_id)
    return Response(ExperimentSerializer(exp).data)

# ----------------------------------------------------------------------
# Training Runs
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_runs(request, exp_id=None):
    if exp_id:
        runs = TrainingRun.objects.filter(experiment_id=exp_id).order_by('-created_at')
    else:
        runs = TrainingRun.objects.all().order_by('-created_at')
    serializer = TrainingRunSerializer(runs, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_status(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    return Response(TrainingRunSerializer(run).data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_train(request):
    data = request.data
    exp_id = data.get('experiment_id')
    if exp_id:
        experiment = get_object_or_404(Experiment, id=exp_id)
    else:
        experiment = Experiment.objects.create(
            name=data.get('exp_name', f"Exp_{data.get('task_name','train')}"),
            default_config=data
        )
    run = TrainingRun.objects.create(
        experiment=experiment,
        name=data.get('name', f"Run_{data.get('task_name','train')}"),
        job_type='training',
        config=data,
        status='queued'
    )
    start_job(run, run_training)
    serializer = TrainingRunSerializer(run)
    return Response(serializer.data, status=202)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_evaluate(request, run_id=None):
    """
    Start evaluation for a TrainingRun.
    If run_id is provided from the URL, use it; otherwise read from request body.
    """
    if run_id is None:
        run_id = request.data.get('run_id')
    run = get_object_or_404(TrainingRun, id=run_id)
    start_job(run, run_evaluation)
    return Response({"status": "evaluation started"})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_compress(request, run_id=None):
    """Start compression for a TrainingRun, with optional URL argument."""
    if run_id is None:
        run_id = request.data.get('run_id')
    run = get_object_or_404(TrainingRun, id=run_id)
    start_job(run, run_compression)
    return Response({"status": "compression started"})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def stop_job_endpoint(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    if run.status not in ('queued', 'running'):
        return Response({"error": f"Cannot stop run with status {run.status}"}, status=400)
    stop_job(str(run.id))
    run.status = 'stopped'
    run.save()
    push_update(run, {"type": "status", "status": "stopped"})
    return Response({"status": "stopped"})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def training_history(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    return Response(run.training_history)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def training_snapshots(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    snapshots = run.snapshots.all().order_by('step')
    serializer = TrainingSnapshotSerializer(snapshots, many=True)
    return Response(serializer.data)

@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_run_config(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    run.config.update(request.data)
    run.save(update_fields=['config'])
    return Response(TrainingRunSerializer(run).data)

# ----------------------------------------------------------------------
# Artifacts & Downloads
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def run_artifacts(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    artifacts_dir = run.artifacts_dir
    if not artifacts_dir or not os.path.isdir(artifacts_dir):
        return Response({"artifacts": []})
    artifacts = []
    for root, dirs, files in os.walk(artifacts_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), artifacts_dir)
            artifacts.append(rel)
    return Response({"artifacts": artifacts})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def run_artifact_download(request, run_id, artifact_path):
    run = get_object_or_404(TrainingRun, id=run_id)
    base = run.artifacts_dir
    if not base:
        raise Http404("No artifacts directory")
    full = os.path.join(base, artifact_path)
    if not os.path.exists(full) or not os.path.isfile(full):
        raise Http404("Artifact not found")
    return FileResponse(open(full, 'rb'), as_attachment=True)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def run_model_download(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    model_dir = os.path.join(run.artifacts_dir, 'model') if run.artifacts_dir else None
    if not model_dir or not os.path.isdir(model_dir):
        raise Http404("Model directory not found")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(tmp, 'w') as zf:
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, model_dir)
                zf.write(full, arcname)
    return FileResponse(open(tmp.name, 'rb'), as_attachment=True, filename=f"model_{run_id}.zip")

# ----------------------------------------------------------------------
# System & GPU information
# ----------------------------------------------------------------------
@api_view(['GET'])
def system_status(request):
    import psutil
    memory = psutil.virtual_memory()
    return Response({
        'cpu': psutil.cpu_percent(interval=0.5),
        'memory': memory.percent,
        'memory_used_gb': round(memory.used / (1024**3), 1),
        'memory_total_gb': round(memory.total / (1024**3), 1),
    })

@api_view(['GET'])
def gpu_info(request):
    if torch.cuda.is_available():
        total = torch.cuda.device_count()
        used = TrainingRun.objects.filter(status='running', job_type='training').count()
        used = min(used, total)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        return Response({
            'type': 'cuda',
            'total_gpus': total,
            'available': total - used,
            'used': used,
            'memory_total_mb': round(mem_total, 1),
            'memory_allocated_mb': round(mem_alloc, 1),
            'utilization': util,
        })
    elif torch.backends.mps.is_available():
        return Response({'type': 'mps', 'total_gpus': 1, 'available': 1, 'used': 0})
    else:
        return Response({'type': 'cpu', 'total_gpus': 0, 'available': 0, 'used': 0})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def stop_all_jobs_endpoint(request):
    stop_all_jobs()
    return Response({"status": "stop signals sent to all running jobs"})

# ----------------------------------------------------------------------
# Paper generation placeholders (simplified)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def generate_paper(request, run_id):
    # Placeholder – implement LaTeX/PDF generation
    return Response({"message": "Paper generation not implemented"})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def paper_preview_pdf(request, run_id):
    return Response({"error": "PDF preview not implemented"}, status=501)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def experiment_report(request, run_id):
    return render(request, 'web_dashboard/experiment_report.html', {'run_id': run_id})  # adjust template

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def paper_editor(request, run_id):
    return render(request, 'web_dashboard/paper_editor.html', {'run_id': run_id})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_paper_source(request, run_id):
    source = request.body.decode('utf-8')
    run = get_object_or_404(TrainingRun, id=run_id)
    run.config['latex_source'] = source
    run.save(update_fields=['config'])
    return Response({"status": "saved"})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def preview_paper(request, run_id):
    return Response({"html": "<div>Paper preview placeholder</div>"})


# experiments/views.py (add these functions)

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

@api_view(["POST"])
@permission_classes([AllowAny])
def register_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)
    user = User.objects.create_user(username=username, password=password)
    token = Token.objects.create(user=user)
    return Response({"token": token.key, "username": user.username}, status=201)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "username": user.username})
    return Response({"error": "Invalid credentials"}, status=401)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout_view(request):
    Token.objects.filter(user=request.user).delete()
    return Response({"status": "logged out"})


