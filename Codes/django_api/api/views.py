import os, shutil, tempfile, json, logging
from django.http import FileResponse, Http404
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Experiment
from .serializers import ExperimentSerializer
from .jobs import start_job, stop_job as do_stop, cancel_job
from .runners import run_training, run_evaluation, run_compression
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY
import psutil, torch, time, math

logger = logging.getLogger(__name__)

# ── Config discovery ─────────────────────────────────────────────
@api_view(["GET"])
def list_attentions(request):
    return Response({"attentions": list(ATTENTION_REGISTRY.keys())})

@api_view(["GET"])
def list_models(request):
    return Response({
        "models": ["distilbert-base-uncased","bert-base-uncased","roberta-base","albert-base-v2"]
    })

@api_view(["GET"])
def list_tasks(request):
    return Response({
        "tasks": ["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"]
    })

@api_view(["GET"])
def list_compressors(request):
    return Response({"compressors": list(COMPRESSOR_REGISTRY.keys())})

@api_view(["GET"])
def list_methods(request):
    return Response({"methods": ["baseline","lora","rdrop","r3f","fisher_lora","fisher_gelora","llrd"]})

@api_view(["GET"])
def list_presets(request):
    return Response({"presets": ["fast","accurate","extreme_compress"]})

# ── Start jobs ────────────────────────────────────────────────────
@api_view(["POST"])
def start_train(request):
    data = request.data
    exp = Experiment.objects.create(config=data)
    start_job(exp, run_training)
    return Response(ExperimentSerializer(exp).data, status=status.HTTP_202_ACCEPTED)

@api_view(["POST"])
def start_evaluate(request):
    data = request.data
    exp = Experiment.objects.create(config=data)
    start_job(exp, run_evaluation)
    return Response(ExperimentSerializer(exp).data, status=status.HTTP_202_ACCEPTED)

@api_view(["POST"])
def start_compress(request):
    data = request.data
    exp = Experiment.objects.create(config=data)
    start_job(exp, run_compression)
    return Response(ExperimentSerializer(exp).data, status=status.HTTP_202_ACCEPTED)

# ── Job status / stop / config update ─────────────────────────────
@api_view(["GET"])
def job_status(request, job_id):
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
    return Response(ExperimentSerializer(exp).data)

@api_view(["POST"])
def stop_job(request, job_id):
    """Stop a running job (alias for cancel)."""
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
    if exp.status in ("queued", "running"):
        cancel_job(str(job_id))              # sets the event in jobs.py
        exp.status = "stopped"
        exp.save()
        return Response({"status": "stopped"})
    return Response({"error": "Job not cancelable"}, status=400)

@api_view(["PATCH"])
def update_job_config(request, job_id):
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
    new_config = request.data
    exp.config.update(new_config)
    exp.save(update_fields=["config"])
    return Response(ExperimentSerializer(exp).data)

# ── List experiments ─────────────────────────────────────────────
@api_view(["GET"])
def list_experiments(request):
    exps = Experiment.objects.all().order_by("-created_at")
    return Response(ExperimentSerializer(exps, many=True).data)

# ── Artifacts ─────────────────────────────────────────────────────
@api_view(["GET"])
def experiment_artifacts(request, job_id):
    exp_dir = settings.MEDIA_ROOT / str(job_id)
    if not exp_dir.exists():
        return Response({"artifacts": []})
    artifacts = [str(p.relative_to(exp_dir)) for p in exp_dir.rglob("*") if p.is_file()]
    return Response({"artifacts": artifacts})

@api_view(["GET"])
def experiment_artifact_download(request, job_id, artifact_path):
    exp_dir = settings.MEDIA_ROOT / str(job_id)
    file_path = exp_dir / artifact_path
    if not file_path.exists() or not file_path.is_file():
        raise Http404("Artifact not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=os.path.basename(artifact_path))

@api_view(["GET"])
def experiment_model_download(request, job_id):
    exp_dir = settings.MEDIA_ROOT / str(job_id) / "model"
    if not exp_dir.exists():
        raise Http404("Model directory not found")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    shutil.make_archive(tmp.name.replace(".zip", ""), "zip", exp_dir)
    return FileResponse(open(tmp.name, "rb"), as_attachment=True, filename="model.zip")

# ── System monitoring ────────────────────────────────────────────
@api_view(["GET"])
def gpu_info(request):
    info = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
        info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
        info["utilization"] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
    return Response(info)

@api_view(["GET"])
def system_status(request):
    return Response({
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
    })