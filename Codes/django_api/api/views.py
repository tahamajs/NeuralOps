import os, shutil, tempfile, json, logging
from django.http import FileResponse, Http404
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Experiment
from .serializers import ExperimentSerializer
from .jobs import start_job
from .runners import run_training, run_evaluation, run_compression
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

# ── Config discovery ─────────────────────────────────────────────
@api_view(["GET"])
def list_attentions(request):
    return Response({"attentions": list(ATTENTION_REGISTRY.keys())})

@api_view(["GET"])
def list_models(request):
    return Response({
        "models": [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
            "albert-base-v2",
        ]
    })

@api_view(["GET"])
def list_tasks(request):
    return Response({
        "tasks": ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]
    })

@api_view(["GET"])
def list_compressors(request):
    return Response({"compressors": list(COMPRESSOR_REGISTRY.keys())})

@api_view(["GET"])
def list_methods(request):
    return Response({"methods": ["baseline", "lora", "rdrop", "r3f", "fisher_lora", "fisher_gelora", "llrd"]})

@api_view(["GET"])
def list_presets(request):
    return Response({
        "presets": ["fast", "accurate", "extreme_compress"]
    })

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

# ── Job status ────────────────────────────────────────────────────
@api_view(["GET"])
def job_status(request, job_id):
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
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
    artifacts = []
    for p in exp_dir.rglob("*"):
        if p.is_file():
            artifacts.append(str(p.relative_to(exp_dir)))
    return Response({"artifacts": artifacts})

# ── Download a specific artifact ──────────────────────────────────
@api_view(["GET"])
def experiment_artifact_download(request, job_id, artifact_path):
    exp_dir = settings.MEDIA_ROOT / str(job_id)
    file_path = exp_dir / artifact_path
    if not file_path.exists() or not file_path.is_file():
        raise Http404("Artifact not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=os.path.basename(artifact_path))

# ── Model download (zip entire model directory) ────────────────────
@api_view(["GET"])
def experiment_model_download(request, job_id):
    exp_dir = settings.MEDIA_ROOT / str(job_id) / "model"
    if not exp_dir.exists():
        raise Http404("Model directory not found")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    shutil.make_archive(tmp.name.replace(".zip", ""), "zip", exp_dir)
    return FileResponse(open(tmp.name, "rb"), as_attachment=True, filename="model.zip")