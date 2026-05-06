# api/views.py
import psutil
import torch
import logging
import platform
import django
from datetime import datetime
from django.conf import settings
from django.contrib.auth import authenticate, update_session_auth_hash
from django.contrib.auth.models import User
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.views.decorators.cache import cache_page
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Rate limiting (optional)
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
        return Response({"error": "Username and password are required"}, status=status.HTTP_400_BAD_REQUEST)
    if len(username) < 3:
        return Response({"error": "Username must be at least 3 characters"}, status=status.HTTP_400_BAD_REQUEST)
    if len(username) > 150:
        return Response({"error": "Username too long"}, status=status.HTTP_400_BAD_REQUEST)
    if len(password) < 6:
        return Response({"error": "Password must be at least 6 characters"}, status=status.HTTP_400_BAD_REQUEST)
    if email:
        try:
            validate_email(email)
        except ValidationError:
            return Response({"error": "Invalid email address"}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=status.HTTP_400_BAD_REQUEST)
    if email and User.objects.filter(email=email).exists():
        return Response({"error": "Email already registered"}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(username=username, password=password, email=email)
    token = Token.objects.create(user=user)
    logger.info(f"New user registered: {username}")
    response = Response({
        "token": token.key,
        "username": user.username,
        "email": user.email,
        "user_id": user.id
    }, status=status.HTTP_201_CREATED)
    # Set cookie for registration too
    response.set_cookie(
        key='authToken',
        value=token.key,
        httponly=True,
        samesite='Lax',
        secure=False,
        max_age=60*60*24*7
    )
    return response

@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([AnonBurstRateThrottle])
def login(request):
    """Authenticate user and return token."""
    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=status.HTTP_400_BAD_REQUEST)

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
        # Set cookie for CookieTokenAuthentication fallback
        response.set_cookie(
            key='authToken',
            value=token.key,
            httponly=True,
            samesite='Lax',
            secure=False, # Set to True in production with HTTPS
            max_age=60*60*24*7 # 7 days
        )
        return response
    return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout(request):
    """Logout user by deleting the token."""
    request.user.auth_token.delete()
    logger.info(f"User logged out: {request.user.username}")
    response = Response({"status": "logged out"})
    response.delete_cookie('authToken')
    return response

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_info(request):
    """Get current user profile information."""
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
    """Update user profile (first_name, last_name, email)."""
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
                return Response({"error": "Email already in use"}, status=status.HTTP_400_BAD_REQUEST)
            user.email = data["email"]
        except ValidationError:
            return Response({"error": "Invalid email address"}, status=status.HTTP_400_BAD_REQUEST)
    user.save()
    return Response({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
    })

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def change_password(request):
    """Change user password."""
    old_password = request.data.get("old_password")
    new_password = request.data.get("new_password")
    if not old_password or not new_password:
        return Response({"error": "Old password and new password are required"}, status=status.HTTP_400_BAD_REQUEST)
    if len(new_password) < 6:
        return Response({"error": "New password must be at least 6 characters"}, status=status.HTTP_400_BAD_REQUEST)
    if not request.user.check_password(old_password):
        return Response({"error": "Old password is incorrect"}, status=status.HTTP_400_BAD_REQUEST)
    request.user.set_password(new_password)
    request.user.save()
    update_session_auth_hash(request, request.user)
    request.user.auth_token.delete()
    new_token = Token.objects.create(user=request.user)
    logger.info(f"Password changed for user: {request.user.username}")
    return Response({"status": "password changed", "token": new_token.key})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def refresh_token(request):
    """Delete existing token and create a new one."""
    old_token = request.user.auth_token
    old_token.delete()
    new_token = Token.objects.create(user=request.user)
    return Response({"token": new_token.key})

# ----------------------------------------------------------------------
# Configuration endpoints
# ----------------------------------------------------------------------
@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_attentions(request):
    return Response({"attentions": ATTENTION_REGISTRY.keys(), "count": len(ATTENTION_REGISTRY)})

@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_compressors(request):
    return Response({"compressors": COMPRESSOR_REGISTRY.keys(), "count": len(COMPRESSOR_REGISTRY)})

@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_models(request):
    models = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"]
    return Response({"models": models, "count": len(models)})

@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_tasks(request):
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]
    return Response({"tasks": tasks, "count": len(tasks)})

@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_methods(request):
    methods = ["baseline", "lora", "rdrop", "r3f", "fisher_lora", "fisher_gelora", "llrd"]
    return Response({"methods": methods, "count": len(methods)})

@cache_page(60 * 5)  # Cache for 5 minutes
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_presets(request):
    presets = ["fast", "accurate", "extreme_compress"]
    return Response({"presets": presets, "count": len(presets)})

# ----------------------------------------------------------------------
# System & hardware monitoring
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([AllowAny])
def system_status(request):
    """Get CPU, memory, disk, uptime, platform info."""
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
    """Get detailed GPU information (CUDA/MPS)."""
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
    """Simple health check endpoint."""
    return Response({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
    })

@api_view(["GET"])
@permission_classes([AllowAny])
def version(request):
    """API version and backend info."""
    return Response({
        "api_version": "v1",
        "django_version": django.get_version(),
        "environment": "production",
    })


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
from .models import Experiment, TrainingRun
from .serializers import ExperimentSerializer, TrainingRunSerializer
from .runners import run_training, run_evaluation

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
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def experiment_detail(request, pk):
    try:
        experiment = Experiment.objects.get(pk=pk)
    except Experiment.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        serializer = ExperimentSerializer(experiment)
        return Response(serializer.data)
    elif request.method == "PATCH":
        serializer = ExperimentSerializer(experiment, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == "DELETE":
        experiment.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_train(request):
    """Start a training job."""
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
    
    from .jobs import start_job
    start_job(run, run_training)
    
    return Response(TrainingRunSerializer(run).data, status=status.HTTP_201_CREATED)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_evaluate(request):
    """Start an evaluation job."""
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
    
    from .jobs import start_job
    start_job(run, run_evaluation)
    
    return Response(TrainingRunSerializer(run).data, status=status.HTTP_201_CREATED)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_compress(request):
    """Start a compression job."""
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
    
    from .jobs import start_job
    from .runners import run_compression
    start_job(run, run_compression)
    
    return Response(TrainingRunSerializer(run).data, status=status.HTTP_201_CREATED)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_detail(request, pk):
    run = get_object_or_404(TrainingRun, pk=pk)
    return Response(TrainingRunSerializer(run).data)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def stop_run(request, pk):
    """Stop a running job."""
    from .jobs import stop_job
    run = get_object_or_404(TrainingRun, pk=pk)
    if run.status in ["running", "queued"]:
        if stop_job(str(run.id)):
            run.status = "stopped"
            run.completed_at = datetime.now()
            run.save()
            return Response({"status": "job stop initiated"})
        else:
            # Maybe it wasn't in _stop_events but status says running
            run.status = "stopped"
            run.save()
            return Response({"status": "job marked as stopped"})
    return Response({"error": "Job is not running"}, status=status.HTTP_400_BAD_REQUEST)

