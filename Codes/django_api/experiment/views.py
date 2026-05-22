import os
import json
import tempfile
import zipfile
from pathlib import Path
from functools import partial

import psutil
import torch
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, render
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token

from .models import Experiment, TrainingRun, TrainingSnapshot
from .serializers import (
    ExperimentSerializer, TrainingRunSerializer, TrainingSnapshotSerializer
)
from .jobs import start_job, stop_job, stop_all_jobs, push_update
from .runners import run_training, run_evaluation, run_compression

# ----------------------------------------------------------------------
# Optional deep‑diff library – fallback to simple dict diff if missing
# ----------------------------------------------------------------------
try:
    from deepdiff import DeepDiff as _DeepDiff
except ImportError:
    _DeepDiff = None

def _deep_diff(a, b, **kwargs):
    """Wrapper that returns a serialisable dict diff."""
    if _DeepDiff is not None:
        return _DeepDiff(a, b, **kwargs)
    # Fallback: simple key‑by‑key diff
    diff = {}
    all_keys = set(a.keys()) | set(b.keys())
    for k in sorted(all_keys):
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            diff[k] = {"old": va, "new": vb}
    return diff

def compare_runs_api(run_ids, baseline_id=None):
    """Stub – replace with real implementation later."""
    return {
        "message": "Comparison not implemented",
        "run_ids": run_ids,
        "baseline_id": baseline_id,
    }

def hyperparameter_importance(run_ids=None, target_metric="eval_accuracy"):
    return {"message": "Not implemented"}

def metric_correlation_matrix(run_ids=None):
    return {"message": "Not implemented"}

def training_stability_analysis(run_id):
    return {"message": "Not implemented"}

# ----------------------------------------------------------------------
# Config endpoints
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_attentions(request):
    return Response({
        "attentions": ["full", "goldfinch_ahn", "granite_hybrid", "griffin_ssd"]
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_models(request):
    return Response({
        "models": [
            "distilbert-base-uncased", "bert-base-uncased",
            "roberta-base", "albert-base-v2"
        ]
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_tasks(request):
    return Response({
        "tasks": ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "stsb"]
    })

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
# Experiment CRUD
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
# Training runs – list, status, start, stop
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def runs_list(request, exp_id=None):
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
    if run_id is None:
        run_id = request.data.get('run_id')
    run = get_object_or_404(TrainingRun, id=run_id)
    start_job(run, run_evaluation)
    return Response({"status": "evaluation started"})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_compress(request, run_id=None):
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def stop_all_jobs_endpoint(request):
    stop_all_jobs()
    return Response({"status": "stop signals sent to all running jobs"})

# ----------------------------------------------------------------------
# Run details – history, snapshots, config update
# ----------------------------------------------------------------------
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
# Artifacts & downloads
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def run_artifacts(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    artifacts_dir = run.artifacts_dir
    if not artifacts_dir or not os.path.isdir(artifacts_dir):
        return Response({"artifacts": []})
    artifacts = []
    for root, _, files in os.walk(artifacts_dir):
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
        for root, _, files in os.walk(model_dir):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, model_dir)
                zf.write(full, arcname)
    return FileResponse(open(tmp.name, 'rb'), as_attachment=True, filename=f"model_{run_id}.zip")

# ----------------------------------------------------------------------
# System & GPU info
# ----------------------------------------------------------------------
@api_view(['GET'])
def system_status(request):
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

# ----------------------------------------------------------------------
# Paper generation placeholders
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def generate_paper(request, run_id):
    return Response({"message": "Paper generation not implemented"})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def paper_preview_pdf(request, run_id):
    return Response({"error": "PDF preview not implemented"}, status=501)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def experiment_report(request, run_id):
    return render(request, 'web_dashboard/experiment_report.html', {'run_id': run_id})

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

# ----------------------------------------------------------------------
# Authentication views
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Analysis overview
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_analysis_overview(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    base = Path(settings.MEDIA_ROOT) / str(run_id)
    overview = {
        "run_id": str(run.id),
        "status": run.status,
        "metrics": run.final_metrics,
        "has_drift_report": (base / "drift_report.json").exists(),
        "has_fairness_report": (base / "fairness_report.json").exists(),
        "has_robustness_report": (base / "robustness_report.json").exists(),
        "artifacts_url": f"/api/runs/{run_id}/artifacts/",
        "drift_url": f"/api/runs/{run_id}/drift/",
        "fairness_url": f"/api/runs/{run_id}/fairness/",
        "robustness_url": f"/api/runs/{run_id}/robustness/",
    }
    return Response(overview)

# ----------------------------------------------------------------------
# Comparison & diff
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def compare_experiments(request):
    run_ids = request.data.get("run_ids", [])
    baseline_id = request.data.get("baseline_id")
    if len(run_ids) < 2:
        return Response({"error": "Need at least 2 run_ids"}, status=400)
    try:
        result = compare_runs_api(run_ids, baseline_id)
        return Response(result)
    except Exception as e:
        return Response({"error": str(e)}, status=400)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_diff(request, run_id_a, run_id_b):
    run_a = get_object_or_404(TrainingRun, id=run_id_a)
    run_b = get_object_or_404(TrainingRun, id=run_id_b)
    config_diff = _deep_diff(run_a.config or {}, run_b.config or {}, ignore_order=True, verbose_level=2)
    metrics_diff = _deep_diff(run_a.final_metrics or {}, run_b.final_metrics or {}, ignore_order=True)
    return Response({
        "run_a": {"id": str(run_a.id), "name": run_a.name},
        "run_b": {"id": str(run_b.id), "name": run_b.name},
        "config_diff": config_diff if isinstance(config_diff, dict) else config_diff.to_dict(),
        "metrics_diff": metrics_diff if isinstance(metrics_diff, dict) else metrics_diff.to_dict(),
    })

# ----------------------------------------------------------------------
# Analytics stubs
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hp_importance_view(request):
    run_ids = request.query_params.getlist("run_ids")
    metric = request.query_params.get("metric", "eval_accuracy")
    result = hyperparameter_importance(run_ids=run_ids if run_ids else None, target_metric=metric)
    return Response(result)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def metric_correlation_view(request):
    run_ids = request.query_params.getlist("run_ids")
    result = metric_correlation_matrix(run_ids=run_ids if run_ids else None)
    return Response(result)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stability_analysis_view(request, run_id):
    result = training_stability_analysis(run_id)
    return Response(result)

# ----------------------------------------------------------------------
# Template stubs
# ----------------------------------------------------------------------
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def template_list_view(request):
    return Response({"templates": []})

@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def template_detail_view(request, template_id):
    return Response({"id": template_id})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def use_template_view(request, template_id):
    return Response({"status": "template applied"})

# ----------------------------------------------------------------------
# Compare runs (detailed)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compare_runs(request):
    run_ids = request.query_params.getlist('run_ids')
    if len(run_ids) < 2:
        return Response({"error": "Need at least 2 run IDs"}, status=400)

    runs = TrainingRun.objects.filter(id__in=run_ids).prefetch_related(
        Prefetch('snapshots', queryset=TrainingSnapshot.objects.order_by('step'))
    )

    configs = [run.config for run in runs]
    all_keys = sorted(set().union(*configs))
    config_table = []
    for run in runs:
        row = {"run_id": str(run.id), "name": run.name or run.id.hex[:8]}
        for k in all_keys:
            row[k] = run.config.get(k, None)
        config_table.append(row)

    metric_keys = ['loss', 'accuracy', 'eval_loss', 'eval_accuracy']
    metrics = {k: {} for k in metric_keys}
    for run in runs:
        snapshots = run.snapshots.all()
        for mk in metric_keys:
            vals = []
            for snap in snapshots:
                val = getattr(snap, f'train_{mk}', None) or getattr(snap, f'eval_{mk}', None)
                if val is not None:
                    vals.append([snap.step, val])
            metrics[mk][str(run.id)] = vals

    return Response({
        "config_diff": {
            "keys": all_keys,
            "runs": {str(r.id): {k: r.config.get(k, None) for k in all_keys} for r in runs}
        },
        "metrics": metrics,
        "config_table": config_table,
    })




from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import TrainingRun
from .serializers import TrainingRunSerializer

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_runs(request):
    """List all training runs, with optional filters."""
    runs = TrainingRun.objects.all().order_by('-created_at')
    # Optional: filter by status, job_type, etc. from query params
    status_filter = request.query_params.get('status')
    if status_filter:
        runs = runs.filter(status=status_filter)
    serializer = TrainingRunSerializer(runs, many=True)
    return Response(serializer.data)


# experiment/views.py — add this

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def export_experiment_report_pdf(request, run_id):
    """Generate and download a PDF report for a training run."""
    run = get_object_or_404(TrainingRun, id=run_id)
    from notes.report_generator import generate_experiment_report
    from weasyprint import HTML

    note = generate_experiment_report(run_id=str(run.id), author=request.user, save=False)
    html_content = f"""
    <html><head><style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }}
        h1 {{ border-bottom: 2px solid #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style></head><body>{note.content}</body></html>
    """
    pdf = HTML(string=html_content).write_pdf()
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="experiment-report-{run_id}.pdf"'
    return response




@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reproduce_experiment(request, run_id):
    """Re‑run an experiment in the exact same Docker environment."""
    original_run = get_object_or_404(TrainingRun, id=run_id)
    env = get_object_or_404(ExperimentEnvironment, run=original_run)

    # Create a new run with the same config
    new_run = TrainingRun.objects.create(
        experiment=original_run.experiment,
        config=original_run.config,
        status='queued',
    )

    # Rebuild Docker image and launch
    from gpu_marketplace.tasks import provision_job_with_env
    provision_job_with_env.delay(str(new_run.id), env.docker_image, env.requirements_txt)

    return Response({'new_run_id': str(new_run.id), 'status': 'queued'}, status=201)