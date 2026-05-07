import os
import json
import uuid
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

import requests
import psutil
import torch
import platform
import django
from django.conf import settings
from django.core.cache import cache
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.contrib.auth import authenticate, update_session_auth_hash
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.views.decorators.cache import cache_page
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

# Framework imports – fall back to empty registries if not present
try:
    from framework.attentions.core.registry import ATTENTION_REGISTRY
except ImportError:
    ATTENTION_REGISTRY = {}
try:
    from framework.compression_engine.core.registry import COMPRESSOR_REGISTRY
except ImportError:
    COMPRESSOR_REGISTRY = {}
try:
    from framework.core.registry import TRAINER_REGISTRY
except ImportError:
    TRAINER_REGISTRY = {}
try:
    from framework.trainer.methods.registry import METHOD_REGISTRY
except ImportError:
    METHOD_REGISTRY = {}
try:
    from framework.trainer.losses import LOSS_REGISTRY
except ImportError:
    LOSS_REGISTRY = {}
try:
    from framework.trainer.optimizers import OPTIMIZER_REGISTRY
except ImportError:
    OPTIMIZER_REGISTRY = {}
try:
    from framework.trainer.schedulers import SCHEDULER_REGISTRY
except ImportError:
    SCHEDULER_REGISTRY = {}
try:
    from framework.callbacks import CALLBACK_REGISTRY
except ImportError:
    CALLBACK_REGISTRY = {}
try:
    from framework.data.augmentation import AUGMENTATION_REGISTRY
except ImportError:
    AUGMENTATION_REGISTRY = {}

from .models import Experiment, TrainingRun
from .serializers import ExperimentSerializer, TrainingRunSerializer
from .runners import run_training, run_evaluation, run_compression
from .jobs import start_job, stop_job

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Rate limiting
# ----------------------------------------------------------------------
class BurstRateThrottle(UserRateThrottle):
    rate = '100/hour'


class AnonBurstRateThrottle(AnonRateThrottle):
    rate = '20/hour'


# ----------------------------------------------------------------------
# Authentication & User Management
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([AnonBurstRateThrottle])
def register(request):
    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")
    email = request.data.get("email", "").strip()

    if not username or not password:
        return Response({"error": "Username and password are required"}, status=400)
    if len(username) < 3:
        return Response({"error": "Username must be at least 3 characters"}, status=400)
    if len(username) > 150:
        return Response({"error": "Username too long"}, status=400)
    if len(password) < 6:
        return Response({"error": "Password must be at least 6 characters"}, status=400)
    if email:
        try:
            validate_email(email)
        except ValidationError:
            return Response({"error": "Invalid email address"}, status=400)

    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)
    if email and User.objects.filter(email=email).exists():
        return Response({"error": "Email already registered"}, status=400)

    user = User.objects.create_user(username=username, password=password, email=email)
    token = Token.objects.create(user=user)
    logger.info(f"New user registered: {username}")
    response = Response({
        "token": token.key,
        "username": user.username,
        "email": user.email,
        "user_id": user.id
    }, status=201)
    response.set_cookie(
        key='authToken', value=token.key,
        httponly=True, samesite='Lax', secure=False, max_age=60*60*24*7
    )
    return response


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([AnonBurstRateThrottle])
def login(request):
    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)

    user = authenticate(request, username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        logger.info(f"User logged in: {username}")
        response = Response({
            "token": token.key,
            "username": user.username,
            "email": user.email,
            "user_id": user.id,
            "is_staff": user.is_staff
        })
        response.set_cookie(
            key='authToken', value=token.key,
            httponly=True, samesite='Lax', secure=False, max_age=60*60*24*7
        )
        return response
    return Response({"error": "Invalid credentials"}, status=401)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout(request):
    if request.auth and hasattr(request.auth, 'delete'):
        request.auth.delete()
    else:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Token "):
            Token.objects.filter(key=auth_header[6:]).delete()
    response = Response({"status": "logged out"})
    response.delete_cookie('authToken')
    return response


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_info(request):
    user = request.user
    return Response({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "date_joined": user.date_joined,
        "last_login": user.last_login,
        "is_staff": user.is_staff,
    })


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def update_profile(request):
    user = request.user
    data = request.data
    if "first_name" in data:
        user.first_name = data["first_name"][:30]
    if "last_name" in data:
        user.last_name = data["last_name"][:30]
    if "email" in data and data["email"]:
        try:
            validate_email(data["email"])
            if User.objects.exclude(id=user.id).filter(email=data["email"]).exists():
                return Response({"error": "Email already in use"}, status=400)
            user.email = data["email"]
        except ValidationError:
            return Response({"error": "Invalid email address"}, status=400)
    user.save()
    return Response({
        "id": user.id, "username": user.username,
        "email": user.email, "first_name": user.first_name,
        "last_name": user.last_name,
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def change_password(request):
    old_password = request.data.get("old_password")
    new_password = request.data.get("new_password")
    if not old_password or not new_password:
        return Response({"error": "Old and new password required"}, status=400)
    if len(new_password) < 6:
        return Response({"error": "New password must be at least 6 characters"}, status=400)
    if not request.user.check_password(old_password):
        return Response({"error": "Old password is incorrect"}, status=400)
    request.user.set_password(new_password)
    request.user.save()
    update_session_auth_hash(request, request.user)
    request.user.auth_token.delete()
    new_token = Token.objects.create(user=request.user)
    logger.info(f"Password changed for {request.user.username}")
    return Response({"status": "password changed", "token": new_token.key})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def refresh_token(request):
    old_token = request.user.auth_token
    old_token.delete()
    new_token = Token.objects.create(user=request.user)
    return Response({"token": new_token.key})


# ----------------------------------------------------------------------
# Configuration listing endpoints
# ----------------------------------------------------------------------
def _list_registry(request, registry, key_name):
    return Response({key_name: list(registry.keys()), "count": len(registry)})


@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_attentions(request):
    return _list_registry(request, ATTENTION_REGISTRY, "attentions")


@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_compressors(request):
    return _list_registry(request, COMPRESSOR_REGISTRY, "compressors")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_trainers(request):
    return _list_registry(request, TRAINER_REGISTRY, "trainers")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_methods(request):
    return _list_registry(request, METHOD_REGISTRY, "methods")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_losses(request):
    return _list_registry(request, LOSS_REGISTRY, "losses")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_optimizers(request):
    return _list_registry(request, OPTIMIZER_REGISTRY, "optimizers")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_schedulers(request):
    return _list_registry(request, SCHEDULER_REGISTRY, "schedulers")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_callbacks(request):
    return _list_registry(request, CALLBACK_REGISTRY, "callbacks")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_augmentations(request):
    return _list_registry(request, AUGMENTATION_REGISTRY, "augmentations")


@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_models(request):
    models = [
        {"name": "distilbert-base-uncased", "variants": ["standard"]},
        {"name": "bert-base-uncased",       "variants": ["standard", "compressed", "hybrid_svd"]},
        {"name": "roberta-base",            "variants": ["standard"]},
        {"name": "Llama-2-7b-hf",          "variants": ["unsloth"]},
        {"name": "Mistral-7B-v0.1",        "variants": ["unsloth"]},
    ]
    return Response({"models": models, "count": len(models)})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_model_variants(request):
    variants = [
        {"key": "standard",   "description": "Standard HuggingFace"},
        {"key": "unsloth",    "description": "Unsloth accelerated"},
        {"key": "compressed", "description": "SVD compressed embeddings"},
        {"key": "hybrid_svd", "description": "Low‑rank attention layers"},
    ]
    return Response({"variants": variants})


@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_tasks(request):
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]
    return Response({"tasks": tasks, "count": len(tasks)})


@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_presets(request):
    presets = ["fast", "accurate", "extreme_compress"]
    return Response({"presets": presets, "count": len(presets)})


# ----------------------------------------------------------------------
# Configuration options (stubs)
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hpo_options(request):
    return Response({"methods": ["optuna", "hyperopt", "grid"], "default": "optuna"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def distributed_options(request):
    return Response({"strategies": ["ddp", "deepspeed", "fsdp"], "default": "ddp"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def tracking_options(request):
    return Response({"trackers": ["mlflow", "wandb", "tensorboard"], "default": "mlflow"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def deployment_options(request):
    return Response({"registries": ["huggingface_hub", "docker"], "default": "huggingface_hub"})


# ----------------------------------------------------------------------
# System & hardware monitoring
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([AllowAny])
def system_status(request):
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    cpu_freq = psutil.cpu_freq()
    return Response({
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.5),
            "count": psutil.cpu_count(),
            "freq_mhz": cpu_freq.current if cpu_freq else None,
        },
        "memory": {
            "percent": mem.percent,
            "used_gb": round(mem.used / (1024**3), 1),
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
        },
        "disk": {
            "percent": disk.percent,
            "used_gb": round(disk.used / (1024**3), 1),
            "total_gb": round(disk.total / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
        },
        "uptime_seconds": int(psutil.boot_time()),
        "uptime_human": str(datetime.now() - boot_time).split('.')[0],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    })


@api_view(["GET"])
@permission_classes([AllowAny])
def gpu_info(request):
    if torch.cuda.is_available():
        total = torch.cuda.device_count()
        result = {"type": "cuda", "total_gpus": total, "devices": []}
        for i in range(total):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**2)
            mem_alloc = torch.cuda.memory_allocated(i) / (1024**2)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**2)
            util = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util = util_info.gpu
            except Exception:
                pass
            result["devices"].append({
                "index": i,
                "name": props.name,
                "memory_total_mb": round(mem_total, 1),
                "memory_allocated_mb": round(mem_alloc, 1),
                "memory_reserved_mb": round(mem_reserved, 1),
                "utilization_percent": util,
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return Response(result)
    elif torch.backends.mps.is_available():
        return Response({
            "type": "mps",
            "total_gpus": 1,
            "device_name": "Apple Silicon (MPS)",
            "memory_allocated_mb": round(torch.mps.current_allocated_memory() / (1024**2), 1) if hasattr(torch.mps, 'current_allocated_memory') else 0,
        })
    else:
        return Response({"type": "cpu", "total_gpus": 0})


@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    return Response({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
    })


@api_view(["GET"])
@permission_classes([AllowAny])
def version(request):
    return Response({
        "api_version": "v1",
        "django_version": django.get_version(),
        "environment": "production",
    })


# ----------------------------------------------------------------------
# HuggingFace model search
# ----------------------------------------------------------------------
SUGGESTED_MODELS = [
    "distilbert-base-uncased", "bert-base-uncased", "roberta-base",
    "albert-base-v2", "google/electra-small-discriminator",
    "microsoft/deberta-v3-base", "Llama-2-7b-hf", "Mistral-7B-v0.1",
    "google/gemma-2b", "facebook/opt-125m",
]

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def search_huggingface_models(request):
    query = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", 20))
    if not query:
        return Response({"results": SUGGESTED_MODELS[:limit]})

    cache_key = f"hf_models_{query}_{limit}"
    cached = cache.get(cache_key)
    if cached:
        return Response({"results": cached})

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models = api.list_models(search=query, limit=limit)
        results = [model.modelId for model in models]
        cache.set(cache_key, results, 300)
        return Response({"results": results})
    except Exception as e:
        logger.warning("HF Hub search failed: %s", e)
        return Response({"results": [], "error": str(e)})


# ----------------------------------------------------------------------
# Experiments CRUD
# ----------------------------------------------------------------------
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def experiment_list(request):
    if request.method == "GET":
        experiments = Experiment.objects.all()
        serializer = ExperimentSerializer(experiments, many=True)
        return Response(serializer.data)
    elif request.method == "POST":
        serializer = ExperimentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def experiment_detail(request, pk):
    try:
        experiment = Experiment.objects.get(pk=pk)
    except Experiment.DoesNotExist:
        return Response(status=404)

    if request.method == "GET":
        serializer = ExperimentSerializer(experiment)
        return Response(serializer.data)
    elif request.method == "PATCH":
        serializer = ExperimentSerializer(experiment, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        experiment.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_train(request):
    config = request.data
    exp_id = config.get("experiment_id")
    if not exp_id:
        exp_name = config.get("name") or f"Training {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        exp = Experiment.objects.create(name=exp_name)
    else:
        exp = get_object_or_404(Experiment, pk=exp_id)

    run = TrainingRun.objects.create(
        experiment=exp,
        job_type="training",
        status="queued",
        config=config,
        name=config.get("name", exp.name)
    )
    start_job(run, run_training)
    return Response(TrainingRunSerializer(run).data, status=201)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_evaluate(request):
    config = request.data
    exp_id = config.get("experiment_id")
    if not exp_id:
        exp_name = config.get("name") or f"Eval {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        exp = Experiment.objects.create(name=exp_name)
    else:
        exp = get_object_or_404(Experiment, pk=exp_id)

    run = TrainingRun.objects.create(
        experiment=exp,
        job_type="evaluation",
        status="queued",
        config=config,
        name=config.get("name", exp.name)
    )
    start_job(run, run_evaluation)
    return Response(TrainingRunSerializer(run).data, status=201)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_compress(request):
    config = request.data
    exp_id = config.get("experiment_id")
    if not exp_id:
        exp_name = config.get("name") or f"Compression {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        exp = Experiment.objects.create(name=exp_name)
    else:
        exp = get_object_or_404(Experiment, pk=exp_id)

    run = TrainingRun.objects.create(
        experiment=exp,
        job_type="compression",
        status="queued",
        config=config,
        name=config.get("name", exp.name)
    )
    start_job(run, run_compression)
    return Response(TrainingRunSerializer(run).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_detail(request, pk):
    run = get_object_or_404(TrainingRun, pk=pk)
    return Response(TrainingRunSerializer(run).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def stop_run(request, pk):
    run = get_object_or_404(TrainingRun, pk=pk)
    if run.status in ["running", "queued"]:
        if stop_job(str(run.id)):
            run.status = "stopped"
            run.completed_at = timezone.now()
            run.save()
            return Response({"status": "job stop initiated"})
        else:
            run.status = "stopped"
            run.save()
            return Response({"status": "job marked as stopped"})
    return Response({"error": "Job is not running"}, status=400)


@api_view(["GET"])
def run_results(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    base = Path(settings.MEDIA_ROOT) / str(run_id)
    files = {}
    if base.exists():
        for root, dirs, filenames in os.walk(base):
            for f in filenames:
                rel = os.path.relpath(os.path.join(root, f), base)
                files[rel] = request.build_absolute_uri(f"/media/{run_id}/{rel}")
    return Response({
        "run_id": run_id,
        "status": run.status,
        "metrics": run.final_metrics,
        "history": run.training_history,
        "files": files,
    })


@api_view(["POST"])
def run_save(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    save_cfg = request.data.get("save_config", {})
    full_cfg = {**run.config.get("save_config", {}), **save_cfg}

    # Stub: if framework not available, return a simple acknowledgment
    try:
        from framework.utils.results_saver import ResultsSaver
        saver = ResultsSaver(run.id, Path(settings.MEDIA_ROOT), full_cfg)
        saver.save_config(run.config)
        saver.save_history(run.training_history)
        saver.save_final_metrics(run.final_metrics)
        saver.generate_plots(run.training_history)
        if full_cfg.get("save_as_version", True) and saver.should_save_model(run.final_metrics):
            from model_versioning.services import ModelVersionService
            ModelVersionService.create_version(
                run=run,
                model_state=torch.load(
                    Path(settings.MEDIA_ROOT) / str(run.id) / "artifacts" / "model" / "pytorch_model.bin"
                ),
                metrics=run.final_metrics,
                artifact_base=str(Path(settings.MEDIA_ROOT)),
            )
        saver.handle_storage_backend()
        return Response({"status": "saved", "config": full_cfg})
    except ImportError:
        return Response({"status": "save not fully implemented (framework missing)"}, status=501)


# ----------------------------------------------------------------------
# Mega Launcher
# ----------------------------------------------------------------------
try:
    from framework.experiment.mega_orchestrator import MegaOrchestrator
except ImportError:
    MegaOrchestrator = None


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def launch_experiment(request):
    config = request.data
    run_id = str(uuid.uuid4())

    if MegaOrchestrator is None:
        run = TrainingRun.objects.create(
            experiment=Experiment.objects.create(name=config.get("name", "launched")),
            job_type="training", status="queued", config=config
        )
        start_job(run, run_training)
        return Response({"run_id": str(run.id), "status": "launched (fallback)"}, status=202)

    stop_event = threading.Event()
    orchestrator = MegaOrchestrator(
        config=config,
        run_id=run_id,
        base_dir=Path(settings.MEDIA_ROOT),
        stop_event=stop_event,
    )
    def _run():
        try:
            orchestrator.run()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()
    return Response({
        "run_id": run_id,
        "status": "launched",
        "monitor_url": f"/dashboard/run/{run_id}/",
    }, status=202)


# ----------------------------------------------------------------------
# Load Balancer & GPU Allocation
# ----------------------------------------------------------------------
try:
    from framework.distributed.load_balancer.gpu_allocator import GPUAllocator
except ImportError:
    GPUAllocator = None
try:
    from framework.distributed.load_balancer.job_scheduler import DistributedJobScheduler
except ImportError:
    DistributedJobScheduler = None
try:
    from framework.distributed.load_balancer.cluster_manager import ClusterManager
except ImportError:
    ClusterManager = None


def _missing_framework_response(feature):
    return Response({"error": f"Framework module for {feature} not installed"}, status=501)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def gpu_allocator_status(request):
    if not GPUAllocator:
        return _missing_framework_response("GPU Allocator")
    return Response(GPUAllocator().status())


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def gpu_nodes_detail(request):
    if not GPUAllocator:
        return _missing_framework_response("GPU Allocator")
    allocator = GPUAllocator()
    with allocator._lock:
        nodes = {}
        for node_name, gpus in allocator._gpus.items():
            nodes[node_name] = [
                {
                    "index": g.index, "name": g.name,
                    "total_memory_mb": round(g.total_memory_mb, 1),
                    "free_memory_mb": round(g.free_memory_mb, 1),
                    "allocated_mb": round(g.allocated_mb, 1),
                    "utilization_pct": g.utilization_pct,
                    "is_allocated": g.is_allocated,
                    "allocated_to": g.allocated_to,
                } for g in gpus
            ]
    return Response(nodes)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scheduler_status(request):
    if not DistributedJobScheduler:
        return _missing_framework_response("Distributed Scheduler")
    return Response(DistributedJobScheduler().status())


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scheduler_queue(request):
    if not DistributedJobScheduler:
        return _missing_framework_response("Distributed Scheduler")
    scheduler = DistributedJobScheduler()
    with scheduler._lock:
        jobs = [
            {
                "job_id": j.job_id, "priority": j.priority,
                "submit_time": j.submit_time, "num_gpus": j.num_gpus,
                "estimated_duration": j.estimated_duration,
                "state": j.state.value, "user": j.user,
            } for j in scheduler._queue
        ]
    return Response(jobs)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scheduler_history(request):
    if not DistributedJobScheduler:
        return _missing_framework_response("Distributed Scheduler")
    scheduler = DistributedJobScheduler()
    with scheduler._lock:
        history = [
            {
                "job_id": j.job_id, "state": j.state.value,
                "num_gpus": j.num_gpus, "gpu_ids": str(j.gpu_ids),
                "start_time": j.start_time, "user": j.user,
            } for j in scheduler._history
        ]
    return Response(history)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def cluster_nodes(request):
    if not ClusterManager:
        return _missing_framework_response("Cluster Manager")
    cluster = ClusterManager()
    nodes = [
        {
            "hostname": n.hostname, "public_ip": n.public_ip,
            "private_ip": n.private_ip, "role": n.role.value,
            "gpu_count": n.gpu_count,
        } for n in cluster._nodes
    ]
    return Response(nodes)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def cluster_gpu_availability(request):
    if not ClusterManager:
        return _missing_framework_response("Cluster Manager")
    result = ClusterManager().check_gpu_availability()
    return Response(result)


# ----------------------------------------------------------------------
# Drift Detection
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def drift_report(request, run_id):
    report_path = Path(settings.MEDIA_ROOT) / str(run_id) / "drift_report.json"
    if not report_path.exists():
        return Response({"error": "No drift report found"}, status=404)
    with open(report_path) as f:
        report = json.load(f)
    return Response(report)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_drift_check(request, run_id):
    # Stub: would use framework.drift
    return Response({"status": "drift check not implemented (framework missing)"}, status=501)


# ----------------------------------------------------------------------
# Fairness & Bias
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def fairness_evaluation(request, run_id):
    data = request.data
    # Stub
    return Response({"status": "fairness evaluation not implemented (framework missing)"}, status=501)


# ----------------------------------------------------------------------
# Adversarial Robustness
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def robustness_evaluation(request, run_id):
    data = request.data
    # Stub
    return Response({"status": "robustness evaluation not implemented (framework missing)"}, status=501)


# ----------------------------------------------------------------------
# Prompt Registry
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def prompt_list(request):
    # Stub
    return Response([])


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def prompt_detail(request, name):
    return Response({"template": "Hello, {input}"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def prompt_history(request, name):
    return Response([])


# ----------------------------------------------------------------------
# HPO & Federated Learning Status
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hpo_status(request):
    return Response({"status": "idle", "trials_completed": 0})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def fl_status(request):
    return Response({"status": "no active federated run", "connected_clients": 0})


# ----------------------------------------------------------------------
# Model Card, Data Lineage, Model Lineage, Search, Activity Feed
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def generate_model_card_view(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    # Stub
    card = f"# Model Card for {run.id}\n\nNo framework."
    return Response({"model_card_md": card})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def push_model_card_view(request, run_id):
    return Response({"success": False, "message": "Framework missing"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def data_lineage_view(request, run_id):
    return Response({"lineage": "not implemented"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_lineage_view(request, version_id):
    return Response({"nodes": [], "edges": []})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def global_search(request):
    query = request.query_params.get("q", "")
    return Response({"results": [], "total": 0, "query": query})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def activity_feed(request):
    return Response([])