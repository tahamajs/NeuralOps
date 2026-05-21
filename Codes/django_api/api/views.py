import os, json, uuid, logging, threading
from pathlib import Path
from datetime import datetime, timedelta

import torch
import psutil
import platform
import django
import requests
import numpy as np
from django.conf import settings
from django.core.cache import cache
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.contrib.auth import authenticate, update_session_auth_hash
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.cache import cache_page
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

from experiment.models import Experiment, TrainingRun, TrainingSnapshot
from experiment.serializers import ExperimentSerializer, TrainingRunSerializer
from experiment.runners import run_training, run_evaluation, run_compression
from experiment.jobs import start_job, stop_job
from .models import ActivityLog, log_activity

# ── Framework registries (graceful fallback) ────────────────────────
_registry_fallbacks = lambda: {}
ATTENTION_REGISTRY = _registry_fallbacks()
COMPRESSOR_REGISTRY = _registry_fallbacks()
TRAINER_REGISTRY = _registry_fallbacks()
METHOD_REGISTRY = _registry_fallbacks()
LOSS_REGISTRY = _registry_fallbacks()
OPTIMIZER_REGISTRY = _registry_fallbacks()
SCHEDULER_REGISTRY = _registry_fallbacks()
CALLBACK_REGISTRY = _registry_fallbacks()
AUGMENTATION_REGISTRY = _registry_fallbacks()

try:
    from framework.core.registry import ATTENTION_REGISTRY as attn_reg
    ATTENTION_REGISTRY = attn_reg
except ImportError:
    pass
try:
    from framework.compression_engine.core.registry import COMPRESSOR_REGISTRY as comp_reg
    COMPRESSOR_REGISTRY = comp_reg
except ImportError:
    pass
try:
    from framework.core.registry import TRAINER_REGISTRY as train_reg
    TRAINER_REGISTRY = train_reg
except ImportError:
    pass
try:
    from framework.trainer.methods.registry import METHOD_REGISTRY as meth_reg
    METHOD_REGISTRY = meth_reg
except ImportError:
    pass
try:
    from framework.trainer.losses import LOSS_REGISTRY as loss_reg
    LOSS_REGISTRY = loss_reg
except ImportError:
    pass
try:
    from framework.trainer.optimizers import OPTIMIZER_REGISTRY as opt_reg
    OPTIMIZER_REGISTRY = opt_reg
except ImportError:
    pass
try:
    from framework.trainer.schedulers import SCHEDULER_REGISTRY as sched_reg
    SCHEDULER_REGISTRY = sched_reg
except ImportError:
    pass
try:
    from framework.callbacks import CALLBACK_REGISTRY as cb_reg
    CALLBACK_REGISTRY = cb_reg
except ImportError:
    pass
try:
    from framework.data.augmentation import AUGMENTATION_REGISTRY as aug_reg
    AUGMENTATION_REGISTRY = aug_reg
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Rate limiting ────────────────────────────────────────────────────
class BurstRateThrottle(UserRateThrottle):
    rate = '100/hour'

class AnonBurstRateThrottle(AnonRateThrottle):
    rate = '20/hour'

# ── Helpers ──────────────────────────────────────────────────────────
def _registry_list(registry, key_name):
    return Response({key_name: list(registry.keys()), "count": len(registry)})

# ═══════════════════════════════════════════════════════════════════
# Authentication & User Management
# ═══════════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([AnonBurstRateThrottle])
def register(request):
    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")
    email = request.data.get("email", "").strip()
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)
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
    log_activity(user, 'create', f'User {username} registered')
    resp = Response({"token": token.key, "username": user.username, "email": user.email, "user_id": user.id}, status=201)
    resp.set_cookie('authToken', token.key, httponly=True, samesite='Lax', secure=False, max_age=60*60*24*7)
    return resp

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
        log_activity(user, 'create', f'User {username} logged in')
        resp = Response({"token": token.key, "username": user.username, "email": user.email, "user_id": user.id, "is_staff": user.is_staff})
        resp.set_cookie('authToken', token.key, httponly=True, samesite='Lax', secure=False, max_age=60*60*24*7)
        return resp
    return Response({"error": "Invalid credentials"}, status=401)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout(request):
    Token.objects.filter(user=request.user).delete()
    resp = Response({"status": "logged out"})
    resp.delete_cookie('authToken')
    return resp

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_info(request):
    user = request.user
    return Response({
        "id": user.id, "username": user.username, "email": user.email,
        "first_name": user.first_name, "last_name": user.last_name,
        "date_joined": user.date_joined, "last_login": user.last_login, "is_staff": user.is_staff,
    })

@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def update_profile(request):
    user = request.user
    data = request.data
    if "first_name" in data: user.first_name = data["first_name"][:30]
    if "last_name" in data: user.last_name = data["last_name"][:30]
    if "email" in data and data["email"]:
        try:
            validate_email(data["email"])
            if User.objects.exclude(id=user.id).filter(email=data["email"]).exists():
                return Response({"error": "Email already in use"}, status=400)
            user.email = data["email"]
        except ValidationError:
            return Response({"error": "Invalid email address"}, status=400)
    user.save()
    log_activity(user, 'update', f'User {user.username} updated profile')
    return Response({"id": user.id, "username": user.username, "email": user.email, "first_name": user.first_name, "last_name": user.last_name})

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
    Token.objects.filter(user=request.user).delete()
    new_token = Token.objects.create(user=request.user)
    log_activity(request.user, 'update', f'User {request.user.username} changed password')
    return Response({"status": "password changed", "token": new_token.key})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def refresh_token(request):
    Token.objects.filter(user=request.user).delete()
    token = Token.objects.create(user=request.user)
    return Response({"token": token.key})

# ═══════════════════════════════════════════════════════════════════
# Configuration Listings
# ═══════════════════════════════════════════════════════════════════
@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_attentions(request): return _registry_list(request, ATTENTION_REGISTRY, "attentions")

@cache_page(60 * 5)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_compressors(request): return _registry_list(request, COMPRESSOR_REGISTRY, "compressors")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_trainers(request): return _registry_list(request, TRAINER_REGISTRY, "trainers")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_methods(request): return _registry_list(request, METHOD_REGISTRY, "methods")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_losses(request): return _registry_list(request, LOSS_REGISTRY, "losses")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_optimizers(request): return _registry_list(request, OPTIMIZER_REGISTRY, "optimizers")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_schedulers(request): return _registry_list(request, SCHEDULER_REGISTRY, "schedulers")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_callbacks(request): return _registry_list(request, CALLBACK_REGISTRY, "callbacks")

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_augmentations(request): return _registry_list(request, AUGMENTATION_REGISTRY, "augmentations")

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

# ═══════════════════════════════════════════════════════════════════
# System & GPU Monitoring
# ═══════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def system_status(request):
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    cpu_freq = psutil.cpu_freq()
    return Response({
        "cpu": {"percent": psutil.cpu_percent(interval=0.5), "count": psutil.cpu_count(), "freq_mhz": cpu_freq.current if cpu_freq else None},
        "memory": {"percent": mem.percent, "used_gb": round(mem.used / (1024**3), 1), "total_gb": round(mem.total / (1024**3), 1), "available_gb": round(mem.available / (1024**3), 1)},
        "disk": {"percent": disk.percent, "used_gb": round(disk.used / (1024**3), 1), "total_gb": round(disk.total / (1024**3), 1), "free_gb": round(disk.free / (1024**3), 1)},
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
                "index": i, "name": props.name,
                "memory_total_mb": round(mem_total, 1),
                "memory_allocated_mb": round(mem_alloc, 1),
                "memory_reserved_mb": round(mem_reserved, 1),
                "utilization_percent": util,
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return Response(result)
    elif torch.backends.mps.is_available():
        return Response({"type": "mps", "total_gpus": 1, "device_name": "Apple Silicon (MPS)"})
    else:
        return Response({"type": "cpu", "total_gpus": 0})

@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    return Response({"status": "healthy", "timestamp": datetime.now().isoformat(), "database": "connected"})

@api_view(["GET"])
@permission_classes([AllowAny])
def version(request):
    return Response({"api_version": "v1", "django_version": django.get_version(), "environment": "production"})

# ═══════════════════════════════════════════════════════════════════
# HuggingFace Model Search
# ═══════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════
# Experiments CRUD & Job Launching
# ═══════════════════════════════════════════════════════════════════
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
            log_activity(request.user, 'create', f'Experiment {serializer.data["name"]} created')
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
        return Response(ExperimentSerializer(experiment).data)
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
        exp = Experiment.objects.create(name=config.get("name") or f"Training {datetime.now()}")
    else:
        exp = get_object_or_404(Experiment, pk=exp_id)
    run = TrainingRun.objects.create(experiment=exp, job_type="training", status="queued", config=config, name=config.get("name", exp.name))
    start_job(run, run_training)
    log_activity(request.user, 'start', f'Training run {run.id} started')
    return Response(TrainingRunSerializer(run).data, status=201)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_evaluate(request):
    config = request.data
    exp_id = config.get("experiment_id")
    if not exp_id:
        exp = Experiment.objects.create(name=config.get("name") or f"Eval {datetime.now()}")
    else:
        exp = get_object_or_404(Experiment, pk=exp_id)
    run = TrainingRun.objects.create(experiment=exp, job_type="evaluation", status="queued", config=config, name=config.get("name", exp.name))
    start_job(run, run_evaluation)
    log_activity(request.user, 'start', f'Evaluation run {run.id} started')
    return Response(TrainingRunSerializer(run).data, status=201)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_compress(request):
    config = request.data
    exp_id = config.get("experiment_id")
    if not exp_id:
        exp = Experiment.objects.create(name=config.get("name") or f"Compression {datetime.now()}")
    else:
        exp = get_object_or_404(Experiment, pk=exp_id)
    run = TrainingRun.objects.create(experiment=exp, job_type="compression", status="queued", config=config, name=config.get("name", exp.name))
    start_job(run, run_compression)
    log_activity(request.user, 'start', f'Compression run {run.id} started')
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
            log_activity(request.user, 'stop', f'Training run {run.id} stopped')
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
        "run_id": run_id, "status": run.status,
        "metrics": run.final_metrics, "history": run.training_history,
        "files": files,
    })

@api_view(["POST"])
def run_save(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    save_cfg = request.data.get("save_config", {})
    full_cfg = {**run.config.get("save_config", {}), **save_cfg}
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
                model_state=torch.load(Path(settings.MEDIA_ROOT) / str(run.id) / "artifacts" / "model" / "pytorch_model.bin"),
                metrics=run.final_metrics,
                artifact_base=str(Path(settings.MEDIA_ROOT)),
            )
        saver.handle_storage_backend()
        return Response({"status": "saved", "config": full_cfg})
    except ImportError:
        return Response({"status": "save not fully implemented (framework missing)"}, status=501)

# ═══════════════════════════════════════════════════════════════════
# Mega Launcher (using MegaOrchestrator)
# ═══════════════════════════════════════════════════════════════════
try:
    from framework.experiment_utils.mega_orchestrator import MegaOrchestrator
except ImportError:
    MegaOrchestrator = None

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def launch_experiment(request):
    config = request.data
    run_id = str(uuid.uuid4())
    if MegaOrchestrator is None:
        exp = Experiment.objects.create(name=config.get("name", "launched"))
        run = TrainingRun.objects.create(experiment=exp, job_type="training", status="queued", config=config)
        start_job(run, run_training)
        return Response({"run_id": str(run.id), "status": "launched (fallback)"}, status=202)
    stop_event = threading.Event()
    orchestrator = MegaOrchestrator(config=config, run_id=run_id, base_dir=Path(settings.MEDIA_ROOT), stop_event=stop_event)
    def _run():
        try:
            orchestrator.run()
        except Exception as e:
            logger.exception("MegaOrchestrator failed")
    threading.Thread(target=_run, daemon=True).start()
    return Response({"run_id": run_id, "status": "launched", "monitor_url": f"/dashboard/run/{run_id}/"}, status=202)

# ═══════════════════════════════════════════════════════════════════
# Load Balancer & GPU Allocation
# ═══════════════════════════════════════════════════════════════════
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
    if not GPUAllocator: return _missing_framework_response("GPU Allocator")
    return Response(GPUAllocator().status())

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def gpu_nodes_detail(request):
    if not GPUAllocator: return _missing_framework_response("GPU Allocator")
    allocator = GPUAllocator()
    with allocator._lock:
        nodes = {}
        for node_name, gpus in allocator._gpus.items():
            nodes[node_name] = [{
                "index": g.index, "name": g.name,
                "total_memory_mb": round(g.total_memory_mb, 1),
                "free_memory_mb": round(g.free_memory_mb, 1),
                "allocated_mb": round(g.allocated_mb, 1),
                "utilization_pct": g.utilization_pct, "is_allocated": g.is_allocated,
                "allocated_to": g.allocated_to,
            } for g in gpus]
    return Response(nodes)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scheduler_status(request):
    if not DistributedJobScheduler: return _missing_framework_response("Distributed Scheduler")
    return Response(DistributedJobScheduler().status())

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scheduler_queue(request):
    if not DistributedJobScheduler: return _missing_framework_response("Distributed Scheduler")
    scheduler = DistributedJobScheduler()
    with scheduler._lock:
        jobs = [{
            "job_id": j.job_id, "priority": j.priority,
            "submit_time": j.submit_time, "num_gpus": j.num_gpus,
            "estimated_duration": j.estimated_duration, "state": j.state.value, "user": j.user,
        } for j in scheduler._queue]
    return Response(jobs)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scheduler_history(request):
    if not DistributedJobScheduler: return _missing_framework_response("Distributed Scheduler")
    scheduler = DistributedJobScheduler()
    with scheduler._lock:
        history = [{
            "job_id": j.job_id, "state": j.state.value, "num_gpus": j.num_gpus,
            "gpu_ids": str(j.gpu_ids), "start_time": j.start_time, "user": j.user,
        } for j in scheduler._history]
    return Response(history)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def cluster_nodes(request):
    if not ClusterManager: return _missing_framework_response("Cluster Manager")
    cluster = ClusterManager()
    nodes = [{
        "hostname": n.hostname, "public_ip": n.public_ip,
        "private_ip": n.private_ip, "role": n.role.value, "gpu_count": n.gpu_count,
    } for n in cluster._nodes]
    return Response(nodes)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def cluster_gpu_availability(request):
    if not ClusterManager: return _missing_framework_response("Cluster Manager")
    return Response(ClusterManager().check_gpu_availability())

# ═══════════════════════════════════════════════════════════════════
# Drift, Fairness, Robustness
# ═══════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def drift_report(request, run_id):
    report_path = Path(settings.MEDIA_ROOT) / str(run_id) / "drift_report.json"
    if not report_path.exists():
        return Response({"error": "No drift report found"}, status=404)
    with open(report_path) as f:
        return Response(json.load(f))

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def run_drift_check(request, run_id):
    try:
        from framework.drift.integration import auto_drift_check
        # Needs reference data – not available here; return stub
        return Response({"status": "drift check requires reference data"}, status=501)
    except ImportError:
        return Response({"status": "drift module not available"}, status=501)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def fairness_evaluation(request, run_id):
    try:
        from framework.evaluation.fairness import FairnessEvaluator
        # Stub – needs model, data
        return Response({"status": "fairness evaluation not yet implemented"}, status=501)
    except ImportError:
        return Response({"status": "fairness module not available"}, status=501)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def robustness_evaluation(request, run_id):
    try:
        from framework.evaluation.robustness import TextAdversarialTester
        return Response({"status": "robustness evaluation not yet implemented"}, status=501)
    except ImportError:
        return Response({"status": "robustness module not available"}, status=501)

# ═══════════════════════════════════════════════════════════════════
# Prompts, HPO/FL Status (stubs)
# ═══════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def prompt_list(request):
    return Response([])

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def prompt_detail(request, name):
    return Response({"template": "Hello, {input}"})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def prompt_history(request, name):
    return Response([])

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hpo_status(request):
    return Response({"status": "idle", "trials_completed": 0})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def fl_status(request):
    return Response({"status": "no active federated run", "connected_clients": 0})

# ═══════════════════════════════════════════════════════════════════
# Model Card & Lineage
# ═══════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def generate_model_card_view(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    try:
        from web_dashboard.model_card import generate_model_card
        card = generate_model_card(run)
        return Response({"model_card_md": card})
    except ImportError:
        return Response({"model_card_md": f"# Model Card for {run.id}\n\nNot available."})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def push_model_card_view(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        card_md = generate_model_card(run)
        repo_id = f"user/{run.id}-model-card"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_file(path_or_fileobj=card_md.encode(), path_in_repo="README.md", repo_id=repo_id)
        return Response({"success": True, "repo": repo_id})
    except Exception as e:
        return Response({"success": False, "message": str(e)})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def data_lineage_view(request, run_id):
    return Response({"lineage": "not implemented"})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_lineage_view(request, version_id):
    return Response({"nodes": [], "edges": []})

# ═══════════════════════════════════════════════════════════════════
# Global Search
# ═══════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def global_search(request):
    query = request.query_params.get('q', '').strip()
    entity_type = request.query_params.get('type', 'all')
    limit = int(request.query_params.get('limit', 20))
    if not query:
        return Response({"results": [], "total": 0})

    results = []
    total = 0
    if entity_type in ('all', 'experiments'):
        runs = TrainingRun.objects.filter(
            Q(name__icontains=query) | Q(config__model_name__icontains=query) | Q(config__task_name__icontains=query)
        ).order_by('-created_at')[:limit]
        for r in runs:
            results.append({
                'type': 'training_run', 'id': str(r.id), 'title': r.name or str(r.id)[:8],
                'subtitle': f"{r.config.get('model_name','')} → {r.config.get('task_name','')}",
                'status': r.status, 'created_at': r.created_at.isoformat(),
            })
        total += runs.count()

    if entity_type in ('all', 'notes'):
        from notes.models import ResearcherNote
        notes = ResearcherNote.objects.filter(deleted_at__isnull=True).filter(
            Q(title__icontains=query) | Q(content__icontains=query)
        ).order_by('-created_at')[:limit]
        if not request.user.is_staff:
            notes = notes.filter(Q(is_public=True) | Q(author=request.user))
        for n in notes:
            results.append({
                'type': 'note', 'id': str(n.id), 'title': n.title or 'Untitled',
                'subtitle': n.content[:150], 'author': n.author.username,
                'created_at': n.created_at.isoformat(),
            })
        total += notes.count()

    if entity_type in ('all', 'models'):
        from model_versioning.models import ModelVersion
        models_qs = ModelVersion.objects.filter(
            Q(version__icontains=query) | Q(run__name__icontains=query)
        ).order_by('-created_at')[:limit]
        for m in models_qs:
            results.append({
                'type': 'model_version', 'id': str(m.id), 'title': m.version,
                'subtitle': f"Run: {m.run.name or str(m.run.id)[:8]}",
                'metrics': m.metrics, 'created_at': m.created_at.isoformat(),
            })
        total += models_qs.count()

    if entity_type in ('all', 'jobs'):
        from jobscheduler.models import Job
        jobs = Job.objects.filter(
            Q(name__icontains=query) | Q(model_name__icontains=query) | Q(task_name__icontains=query)
        ).order_by('-created_at')[:limit]
        for j in jobs:
            results.append({
                'type': 'job', 'id': str(j.id), 'title': j.name or str(j.id)[:8],
                'subtitle': f"{j.model_name} / {j.task_name}", 'status': j.status,
                'created_at': j.created_at.isoformat(),
            })
        total += jobs.count()

    results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    results = results[:limit]
    return Response({"query": query, "total": total, "results": results})

# ═══════════════════════════════════════════════════════════════════
# Activity Feed
# ═══════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def activity_feed(request):
    limit = int(request.query_params.get('limit', 50))
    logs = ActivityLog.objects.select_related('user').order_by('-created_at')[:limit]
    data = [{
        'id': str(log.id), 'username': log.user.username if log.user else 'system',
        'action_type': log.action_type, 'description': log.description,
        'entity_type': str(log.content_type) if log.content_type else None,
        'object_id': log.object_id, 'metadata': log.metadata,
        'created_at': log.created_at.isoformat(),
    } for log in logs]
    return Response(data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def run_hf_checkpoints(request, run_id):
    """Return a list of checkpoints uploaded to HuggingFace for this run."""
    run = get_object_or_404(TrainingRun, id=run_id)
    # The HF repo ID is derived from run.id
    username = getattr(settings, 'HUGGINGFACE_USERNAME', 'neuralops')
    repo_id = f"{username}/neuralops-run-{run.id}"
    try:
        api = HfApi()
        # List files under checkpoints/
        files = api.list_repo_files(repo_id, repo_type="model")
        checkpoints = []
        for f in files:
            if f.startswith("checkpoints/step_"):
                step = int(f.split("_")[-1].split("/")[0])
                checkpoints.append({
                    "step": step,
                    "url": f"https://huggingface.co/{repo_id}/resolve/main/{f}"
                })
        return Response({"checkpoints": checkpoints})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    



# ── System Status (CPU, memory, disk) ───────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def system_status(request):
    import psutil
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return Response({
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.5),
            "count": psutil.cpu_count(),
        },
        "memory": {
            "percent": mem.percent,
            "used_gb": round(mem.used / (1024**3), 1),
            "total_gb": round(mem.total / (1024**3), 1),
        },
        "disk": {
            "percent": disk.percent,
            "free_gb": round(disk.free / (1024**3), 1),
            "total_gb": round(disk.total / (1024**3), 1),
        },
    })

# ── GPU Info ────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def gpu_info(request):
    try:
        import torch
        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "memory_total_mb": round(props.total_memory / (1024**2), 1),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(i) / (1024**2), 1),
                    "memory_reserved_mb": round(torch.cuda.memory_reserved(i) / (1024**2), 1),
                })
            return Response({"type": "cuda", "total_gpus": len(devices), "devices": devices})
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return Response({"type": "mps", "total_gpus": 1})
        else:
            return Response({"type": "cpu", "total_gpus": 0})
    except ImportError:
        return Response({"type": "cpu", "total_gpus": 0})