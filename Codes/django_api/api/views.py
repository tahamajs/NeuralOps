from django.shortcuts import render

# Create your views here.
import os
import shutil
import tempfile
import logging
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
            "gpt2",
            "microsoft/deberta-v3-base",
        ]
    })

@api_view(["GET"])
def list_tasks(request):
    return Response({
        "tasks": ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]
    })

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
    files = []
    for p in exp_dir.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(exp_dir)))
    return Response({"artifacts": files})

# ── Model download ────────────────────────────────────────────────
@api_view(["GET"])
def experiment_model_download(request, job_id):
    """Zip the trained model directory and return it as a download."""
    exp_dir = settings.MEDIA_ROOT / str(job_id) / "model"
    if not exp_dir.exists():
        raise Http404("Model directory not found for this experiment.")

    # Create a zip archive in memory
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    shutil.make_archive(tmp.name.replace(".zip", ""), "zip", exp_dir)
    response = FileResponse(open(tmp.name, "rb"), as_attachment=True, filename="model.zip")
    # After response is sent, clean up
    # (Note: FileResponse does not delete the temp file automatically; we can't easily hook into closing.
    #  For production, consider using a streaming response and cleaning up after.)
    return response