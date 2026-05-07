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



from .comparison import compare_runs_api

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def compare_experiments(request):
    """Compare multiple training runs side-by-side."""
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
    """Deep diff between two specific runs."""
    run_a = get_object_or_404(TrainingRun, id=run_id_a)
    run_b = get_object_or_404(TrainingRun, id=run_id_b)
    config_diff = DeepDiff(run_a.config or {}, run_b.config or {}, ignore_order=True, verbose_level=2)
    metrics_diff = DeepDiff(run_a.final_metrics or {}, run_b.final_metrics or {}, ignore_order=True)
    return Response({
        "run_a": {"id": str(run_a.id), "name": run_a.name},
        "run_b": {"id": str(run_b.id), "name": run_b.name},
        "config_diff": config_diff,
        "metrics_diff": metrics_diff,
        "config_diff_json": config_diff.to_dict() if hasattr(config_diff, 'to_dict') else str(config_diff),
        "metrics_diff_json": metrics_diff.to_dict() if hasattr(metrics_diff, 'to_dict') else str(metrics_diff),
    })



# ---- Import the analytics functions ----
from .analytics import (
    hyperparameter_importance,
    metric_correlation_matrix,
    training_stability_analysis,
)
from .templates_api import (
    template_list as _template_list,
    template_detail as _template_detail,
    use_template as _use_template,
)


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


# Template endpoints – just forward to the imported functions
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def template_list(request):
    return _template_list(request)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def template_detail(request, template_id):
    return _template_detail(request, template_id)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def use_template(request, template_id):
    return _use_template(request, template_id)