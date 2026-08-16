import psutil
import torch
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, render
from experiment.models import TrainingRun
from jobscheduler.models import Job
from model_versioning.models import ModelVersion
from notes.models import ResearcherNote
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from sweeps.models import Sweep

from .models import DashboardWidget
from .serializers import (
    ComputeJobSerializer,
    ComputeOrderSerializer,
    DashboardWidgetSerializer,
    EscrowBalanceSerializer,
    GPUListingSerializer,
    InvoiceSerializer,
)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════
DEFAULT_WIDGETS = [
    {
        "widget_type": "recent_runs",
        "title": "Recent Runs",
        "position_x": 0,
        "position_y": 0,
        "width": 2,
        "height": 1,
    },
    {
        "widget_type": "gpu_monitor",
        "title": "GPU Monitor",
        "position_x": 2,
        "position_y": 0,
        "width": 1,
        "height": 1,
    },
    {
        "widget_type": "run_status",
        "title": "Run Status",
        "position_x": 0,
        "position_y": 1,
        "width": 1,
        "height": 1,
    },
    {
        "widget_type": "notes_feed",
        "title": "Notes",
        "position_x": 1,
        "position_y": 1,
        "width": 2,
        "height": 1,
    },
    {
        "widget_type": "scheduler_queue",
        "title": "Scheduler Queue",
        "position_x": 3,
        "position_y": 0,
        "width": 1,
        "height": 1,
    },
    {
        "widget_type": "system_health",
        "title": "System Health",
        "position_x": 3,
        "position_y": 1,
        "width": 1,
        "height": 1,
    },
]


# ═════════════════════════════════════════════════════════════════
# HTML Page Views (for SPA fallback / direct access)
# ═════════════════════════════════════════════════════════════════
def index(request):
    runs = TrainingRun.objects.select_related("experiment").order_by("-created_at")[:20]
    return render(request, "web_dashboard/index.html", {"runs": runs})


def create_job_page(request):
    return render(request, "web_dashboard/dashboard.html")


@login_required
def dashboard(request):
    return render(request, "web_dashboard/dashboard.html")


@login_required
def job_detail(request, job_id):
    run = get_object_or_404(TrainingRun, id=job_id)
    return render(request, "web_dashboard/experiment_monitor.html", {"run": run})


@login_required
def statistics_page(request):
    total = TrainingRun.objects.count()
    completed = TrainingRun.objects.filter(status="completed").count()
    success_rate = round(completed / total * 100, 1) if total else 0
    if success_rate == int(success_rate):
        success_rate_val = int(success_rate)
    else:
        success_rate_val = success_rate
    return render(
        request,
        "web_dashboard/statistics.html",
        {"success_rate": success_rate_val, "total_experiments": total}
    )


def login_page(request):
    return render(request, "web_dashboard/login.html")


def register_page(request):
    return render(request, "web_dashboard/register.html")


@login_required
def job_queue_page(request):
    return render(request, "web_dashboard/job_queue_manager.html")


@login_required
def experiments_page(request):
    return render(request, "web_dashboard/dashboard.html")


@login_required
def analysis_page(request):
    return render(request, "web_dashboard/research_analysis.html")


@login_required
def latex_editor_page(request):
    return render(request, "web_dashboard/latex_editor.html")


@login_required
def notes_page(request):
    return render(request, "web_dashboard/notes_manager.html")


@login_required
def models_page(request):
    return render(request, "web_dashboard/model_versions.html")


@login_required
def scheduler_page(request):
    return render(request, "web_dashboard/scheduler_panel.html")


@login_required
def metrics_page(request):
    return render(request, "web_dashboard/metrics_dashboard.html")


# ═════════════════════════════════════════════════════════════════
# Authentication API (JWT via DRF Token)
# ═════════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([AllowAny])
def api_register(request):
    username = request.data.get("username")
    password = request.data.get("password")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)
    if len(username) > 150:
        return Response({"error": "Username too long (max 150 characters)"}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)
    user = User.objects.create_user(username=username, password=password)
    token = Token.objects.create(user=user)
    auth_login(request, user, backend="django.contrib.auth.backends.ModelBackend")
    resp = Response({"token": token.key, "username": user.username}, status=201)
    resp.set_cookie(
        "authToken",
        token.key,
        httponly=False,
        samesite="Lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return resp


@api_view(["POST"])
@permission_classes([AllowAny])
def api_login(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        auth_login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        resp = Response({"token": token.key, "username": user.username})
        resp.set_cookie(
            "authToken",
            token.key,
            httponly=False,
            samesite="Lax",
            max_age=60 * 60 * 24 * 30,
            path="/",
        )
        return resp
    return Response({"error": "Invalid credentials"}, status=401)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def api_logout(request):
    Token.objects.filter(user=request.user).delete()
    resp = Response({"status": "logged out"})
    resp.delete_cookie("authToken")
    auth_logout(request)
    return resp


# ═════════════════════════════════════════════════════════════════
# Dashboard Widget Management
# ═════════════════════════════════════════════════════════════════
@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def dashboard_widgets(request):
    if request.method == "GET":
        widgets = DashboardWidget.objects.filter(user=request.user).order_by(
            "position_y", "position_x"
        )
        if not widgets.exists():
            for w in DEFAULT_WIDGETS:
                DashboardWidget.objects.create(user=request.user, **w)
            widgets = DashboardWidget.objects.filter(user=request.user).order_by(
                "position_y", "position_x"
            )
        serializer = DashboardWidgetSerializer(widgets, many=True)
        return Response(serializer.data)

    elif request.method == "PUT":
        widgets_data = request.data.get("widgets", [])
        existing_ids = set(
            str(w.id) for w in DashboardWidget.objects.filter(user=request.user)
        )
        updated_ids = set()

        for wdata in widgets_data:
            wid = wdata.get("id")
            if wid and wid in existing_ids:
                widget = DashboardWidget.objects.get(id=wid, user=request.user)
                for field in [
                    "position_x",
                    "position_y",
                    "width",
                    "height",
                    "is_visible",
                    "config",
                    "title",
                ]:
                    if field in wdata:
                        setattr(widget, field, wdata[field])
                widget.save()
                updated_ids.add(wid)
            elif not wid:
                new_widget = DashboardWidget.objects.create(
                    user=request.user,
                    **{
                        k: v
                        for k, v in wdata.items()
                        if k
                        in [
                            "widget_type",
                            "title",
                            "config",
                            "position_x",
                            "position_y",
                            "width",
                            "height",
                            "is_visible",
                        ]
                    },
                )
                updated_ids.add(str(new_widget.id))

        to_delete = existing_ids - updated_ids
        DashboardWidget.objects.filter(id__in=to_delete).delete()
        return Response({"status": "saved"})


# ═════════════════════════════════════════════════════════════════
# Widget Live Data Resolvers (15+ types, fully implemented)
# ═════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def widget_data(request, widget_id):
    widget = get_object_or_404(DashboardWidget, id=widget_id, user=request.user)
    data = resolve_widget_data(widget)
    return Response(data)


def resolve_widget_data(widget):
    """Returns live data based on widget type, utilizing all platform apps."""
    wtype = widget.widget_type
    cfg = widget.config or {}

    # ── Recent Runs ───────────────────────────────────
    if wtype == "recent_runs":
        runs = TrainingRun.objects.select_related("experiment").order_by("-created_at")[
            : cfg.get("limit", 10)
        ]
        return {
            "runs": [
                {
                    "id": str(r.id),
                    "name": r.name,
                    "status": r.status,
                    "experiment": r.experiment.name if r.experiment else "",
                    "created_at": r.created_at.isoformat(),
                    "progress": (
                        round(r.current_step / r.total_steps * 100, 1)
                        if r.total_steps
                        else 0
                    ),
                }
                for r in runs
            ]
        }

    # ── GPU Monitor ────────────────────────────────────
    elif wtype == "gpu_monitor":
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.device_count(),
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            }
        return {"total": 0, "allocated_mb": 0, "reserved_mb": 0}

    # ── Run Status Overview ────────────────────────────
    elif wtype == "run_status":
        runs = TrainingRun.objects.all()
        return {
            "total": runs.count(),
            "running": runs.filter(status="running").count(),
            "completed": runs.filter(status="completed").count(),
            "failed": runs.filter(status="failed").count(),
            "queued": runs.filter(status="queued").count(),
        }

    # ── Notes Feed ─────────────────────────────────────
    elif wtype == "notes_feed":
        notes = ResearcherNote.objects.filter(is_public=True).order_by("-created_at")[
            : cfg.get("limit", 5)
        ]
        return {
            "notes": [
                {
                    "id": str(n.id),
                    "title": n.title,
                    "author": n.author.username,
                    "created_at": n.created_at.isoformat(),
                    "content": n.content[:200],
                }
                for n in notes
            ]
        }

    # ── Loss Chart (per run snapshots) ─────────────────
    elif wtype == "loss_chart":
        run_ids = cfg.get("run_ids", [])
        if not run_ids:
            runs = TrainingRun.objects.filter(
                status__in=["running", "completed"]
            ).order_by("-created_at")[:1]
            run_ids = [str(r.id) for r in runs]
        history = {}
        for rid in run_ids:
            try:
                run = TrainingRun.objects.get(id=rid)
                snapshots = run.snapshots.order_by("step")
                history[rid] = {
                    "steps": [s.step for s in snapshots],
                    "losses": [s.train_loss for s in snapshots],
                    "eval_losses": [s.eval_loss for s in snapshots],
                }
            except TrainingRun.DoesNotExist:
                pass
        return {"history": history}

    # ── Accuracy Chart ─────────────────────────────────
    elif wtype == "accuracy_chart":
        run_ids = cfg.get("run_ids", [])
        if not run_ids:
            runs = TrainingRun.objects.filter(
                status__in=["running", "completed"]
            ).order_by("-created_at")[:1]
            run_ids = [str(r.id) for r in runs]
        history = {}
        for rid in run_ids:
            try:
                run = TrainingRun.objects.get(id=rid)
                snapshots = run.snapshots.order_by("step")
                history[rid] = {
                    "steps": [s.step for s in snapshots if s.eval_accuracy is not None],
                    "accuracies": [
                        s.eval_accuracy
                        for s in snapshots
                        if s.eval_accuracy is not None
                    ],
                }
            except TrainingRun.DoesNotExist:
                pass
        return {"history": history}

    # ── Metric Comparison (selected runs) ───────────────
    elif wtype == "metric_comparison":
        run_ids = cfg.get("run_ids", [])
        metric = cfg.get("metric", "eval_accuracy")
        runs = (
            TrainingRun.objects.filter(id__in=run_ids)
            if run_ids
            else TrainingRun.objects.filter(status="completed")[:5]
        )
        data = []
        for r in runs:
            val = r.final_metrics.get(metric) if r.final_metrics else None
            data.append({"id": str(r.id), "name": r.name, "metric": val})
        return {"metric": metric, "runs": data}

    # ── Model Registry ─────────────────────────────────
    elif wtype == "model_registry":
        models_list = ModelVersion.objects.filter(is_published_to_hub=True).order_by(
            "-created_at"
        )[: cfg.get("limit", 10)]
        return {
            "models": [
                {
                    "id": str(m.id),
                    "version": m.version,
                    "metrics": m.metrics,
                    "created_at": m.created_at.isoformat(),
                    "hub_url": m.hub_url,
                }
                for m in models_list
            ]
        }

    # ── Scheduler Queue ─────────────────────────────────
    elif wtype == "scheduler_queue":
        from jobscheduler.scheduler import scheduler as gpu_sched

        status = gpu_sched.get_status() if hasattr(gpu_sched, "get_status") else {}
        jobs = Job.objects.filter(status="queued").order_by("priority", "created_at")[
            :10
        ]
        return {
            "total_gpus": status.get("total_gpus", 0),
            "available_gpus": status.get("available_gpus", 0),
            "queued_jobs": status.get("queued_jobs", 0),
            "running_jobs": status.get("running_jobs", 0),
            "jobs": [
                {"id": str(j.id), "name": j.name, "priority": j.priority} for j in jobs
            ],
        }

    # ── Custom Plot (stored Plotly JSON) ────────────────
    elif wtype == "custom_plot":
        return {"plot": cfg.get("plot_json", {})}

    # ── Terminal (just returns WebSocket endpoint) ──────
    elif wtype == "terminal":
        return {"ws_endpoint": "/ws/terminal/"}

    # ── Sweep Monitor ───────────────────────────────────
    elif wtype == "sweep_monitor":
        sweep_id = cfg.get("sweep_id")
        if sweep_id:
            try:
                sweep = Sweep.objects.get(id=sweep_id)
            except Sweep.DoesNotExist:
                return {"sweep": None, "trials": []}
            trials = sweep.trials.order_by("trial_number")
            return {
                "sweep": {
                    "id": str(sweep.id),
                    "name": sweep.name,
                    "status": sweep.status,
                    "best_value": (
                        sweep.best_trial.objective_value if sweep.best_trial else None
                    ),
                },
                "trials": [
                    {
                        "number": t.trial_number,
                        "status": t.status,
                        "value": t.objective_value,
                        "params": t.params,
                    }
                    for t in trials
                ],
            }
        else:
            # Show most recently active sweep
            sweep = (
                Sweep.objects.filter(status="active").order_by("-created_at").first()
            )
            if sweep:
                return resolve_widget_data(
                    DashboardWidget(
                        widget_type="sweep_monitor", config={"sweep_id": str(sweep.id)}
                    )
                )
            return {"sweep": None, "trials": []}

    # ── Experiment Comparison (multi‑run metrics) ─────────
    elif wtype == "experiment_comparison":
        run_ids = cfg.get("run_ids", [])
        if len(run_ids) >= 2:
            runs = TrainingRun.objects.filter(id__in=run_ids)
            metrics = {}
            for r in runs:
                if r.final_metrics:
                    for k, v in r.final_metrics.items():
                        if k not in metrics:
                            metrics[k] = {}
                        metrics[k][str(r.id)] = v
            return {"metrics": metrics, "run_ids": run_ids}
        return {"metrics": {}, "run_ids": run_ids}

    # ── System Health ───────────────────────────────────
    elif wtype == "system_health":
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu": cpu,
            "memory_percent": mem.percent,
            "memory_used_gb": round(mem.used / 1024**3, 1),
            "memory_total_gb": round(mem.total / 1024**3, 1),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1024**3, 1),
        }

    # ── Live Log ────────────────────────────────────────
    elif wtype == "live_log":
        run_ids = cfg.get("run_ids", [])
        if run_ids:
            try:
                run = TrainingRun.objects.get(id=run_ids[0])
                return {"logs": run.logs[-5000:] if run.logs else ""}
            except TrainingRun.DoesNotExist:
                return {"logs": ""}
        return {"logs": ""}

    # ── Recently Viewed / Pinned ─────────────────────────
    elif wtype == "recently_viewed":
        # Store viewed run IDs in user preferences or config
        # For now, return the most recent 5 completed runs
        runs = TrainingRun.objects.filter(status="completed").order_by("-created_at")[
            :5
        ]
        return {"runs": [{"id": str(r.id), "name": r.name} for r in runs]}

    # ── Quick Actions ────────────────────────────────────
    elif wtype == "quick_actions":
        return {
            "actions": [
                {"label": "New Training", "url": "/create/"},
                {"label": "View Sweeps", "url": "/sweeps/"},
                {"label": "Model Registry", "url": "/models/"},
                {"label": "Terminal", "url": "/terminal/"},
            ]
        }

    # ── System Status (last error, etc.) ────────────────
    elif wtype == "system_status":
        last_failed = (
            TrainingRun.objects.filter(status="failed").order_by("-created_at").first()
        )
        return {
            "last_failed": (
                {
                    "id": str(last_failed.id),
                    "name": last_failed.name,
                    "created_at": last_failed.created_at.isoformat(),
                }
                if last_failed
                else None
            ),
            "active_users": User.objects.filter(is_active=True).count(),
        }

    # Fallback
    return {"message": f"No resolver for widget type: {wtype}"}


# ═════════════════════════════════════════════════════════════════
# Dashboard Statistics & Launch Configuration
# ═════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    total = TrainingRun.objects.count()
    running = TrainingRun.objects.filter(status="running").count()
    completed = TrainingRun.objects.filter(status="completed").count()
    failed = TrainingRun.objects.filter(status="failed").count()
    queued = TrainingRun.objects.filter(status="queued").count()
    sweeps_active = Sweep.objects.filter(status="active").count()
    jobs_queued = Job.objects.filter(status="queued").count()
    models_published = ModelVersion.objects.filter(is_published_to_hub=True).count()
    notes_count = ResearcherNote.objects.count()

    return Response(
        {
            "total_runs": total,
            "running": running,
            "completed": completed,
            "failed": failed,
            "queued": queued,
            "success_rate": round(completed / total * 100, 1) if total else 0,
            "sweeps_active": sweeps_active,
            "jobs_queued": jobs_queued,
            "models_published": models_published,
            "notes_count": notes_count,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def launch_config_options(request):
    """Return all available options for job creation wizard."""
    try:
        from framework.callbacks import CALLBACK_REGISTRY
        from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        from framework.data.augmentation import AUGMENTATION_REGISTRY
        from framework.trainer.losses import LOSS_REGISTRY
        from framework.trainer.methods.registry import METHOD_REGISTRY
        from framework.trainer.optimizers import OPTIMIZER_REGISTRY
        from framework.trainer.schedulers import SCHEDULER_REGISTRY
    except ImportError:
        return Response({"error": "Framework not available"}, status=500)

    return Response(
        {
            "attentions": sorted(ATTENTION_REGISTRY.keys()),
            "compressors": sorted(COMPRESSOR_REGISTRY.keys()),
            "methods": sorted(METHOD_REGISTRY.keys()),
            "losses": sorted(LOSS_REGISTRY.keys()),
            "optimizers": sorted(OPTIMIZER_REGISTRY.keys()),
            "schedulers": sorted(SCHEDULER_REGISTRY.keys()),
            "callbacks": sorted(CALLBACK_REGISTRY.keys()),
            "augmentations": sorted(AUGMENTATION_REGISTRY.keys()),
            "tasks": [
                "cola",
                "sst2",
                "mrpc",
                "qqp",
                "mnli",
                "qnli",
                "rte",
                "wnli",
                "stsb",
            ],
            "models": [
                "distilbert-base-uncased",
                "bert-base-uncased",
                "roberta-base",
                "albert-base-v2",
                "gpt2",
                "llama-2-7b",
                "mistral-7b",
            ],
            "distributed_strategies": ["none", "ddp", "deepspeed", "fsdp"],
            "model_variants": ["standard", "unsloth", "compressed", "hybrid_svd"],
            "tracking_backends": ["local", "tensorboard", "wandb", "mlflow"],
            "storage_backends": ["local", "s3", "gcp", "azure", "hf_hub"],
        }
    )


# ═════════════════════════════════════════════════════════════════
# User Dashboard Settings (theme, refresh, etc.)
# ═════════════════════════════════════════════════════════════════
@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def dashboard_settings(request):
    if request.method == "GET":
        prefs = (
            request.user.dashboard_preferences
            if hasattr(request.user, "dashboard_preferences")
            else {}
        )
        return Response(prefs)
    elif request.method == "PUT":
        request.user.dashboard_preferences = request.data
        request.user.save(update_fields=["dashboard_preferences"])
        return Response({"status": "saved"})


# ═════════════════════════════════════════════════════════════════
# Widget Configuration (for customizable widgets)
# ═════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def widget_config(request, widget_id):
    widget = get_object_or_404(DashboardWidget, id=widget_id, user=request.user)
    # Return safe configuration for editing
    return Response(
        {
            "config": widget.config,
            "widget_type": widget.widget_type,
            "title": widget.title,
        }
    )


# ═════════════════════════════════════════════════════════════════
# Model Card Generation (from run)
# ═════════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def generate_model_card(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    from .model_card import generate_model_card as card_gen

    card = card_gen(run)
    return Response({"model_card": card})


# ======================== CONSUMER MARKETPLACE VIEWS ========================


# -------------------------------------------------------------------------
# Marketplace Listings (browse & filter)
# -------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def marketplace_listings(request):
    """
    Browse available GPU listings with full filtering.
    This is the main "Marketplace" page for consumers.
    """
    from gpu_marketplace.models import GPUListing

    qs = GPUListing.objects.filter(status="online").select_related("provider")

    # ---- Filters ----
    gpu_model = request.query_params.get("gpu_model")
    if gpu_model:
        qs = qs.filter(gpu_model__icontains=gpu_model)

    min_vram = request.query_params.get("min_vram")
    if min_vram:
        qs = qs.filter(vram_gb__gte=int(min_vram))

    max_price = request.query_params.get("max_price")
    if max_price:
        qs = qs.filter(current_price_per_hour__lte=float(max_price))

    min_reliability = request.query_params.get("min_reliability")
    if min_reliability:
        qs = qs.filter(reliability_score__gte=float(min_reliability))

    listing_type = request.query_params.get("listing_type")
    if listing_type:
        qs = qs.filter(listing_type=listing_type)

    location = request.query_params.get("location")
    if location:
        qs = qs.filter(location__icontains=location)

    # ---- Sorting ----
    sort = request.query_params.get("sort", "reliability")
    sort_map = {
        "price_asc": "current_price_per_hour",
        "price_desc": "-current_price_per_hour",
        "reliability": "-reliability_score",
        "dlperf": "-dlperf_score",
        "newest": "-created_at",
    }
    qs = qs.order_by(sort_map.get(sort, "-reliability_score"))

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = GPUListingSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# -------------------------------------------------------------------------
# Orders – list & place
# -------------------------------------------------------------------------
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def consumer_orders(request):
    """
    GET: list the authenticated user's buy orders.
    POST: place a new buy order.
    """
    from gpu_marketplace.models import ComputeOrder

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def api_jobs(request):
    """List all training runs for dashboard jobs view."""
    runs = TrainingRun.objects.select_related("experiment").order_by("-created_at")
    data = []
    for r in runs:
        data.append({
            "id": str(r.id),
            "name": r.name,
            "status": r.status,
            "experiment_name": r.experiment.name if r.experiment else "Default",
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "started_at": r.started_at.isoformat() if getattr(r, "started_at", None) else None,
            "completed_at": r.completed_at.isoformat() if getattr(r, "completed_at", None) else None,
            "config": r.config or {},
            "final_metrics": r.final_metrics or {},
            "current_step": r.current_step,
            "total_steps": r.total_steps,
            "eta_seconds": getattr(r, "eta_seconds", 0) or 0,
            "gpu_percent": getattr(r, "gpu_percent", 0.0) or 0.0,
            "gpu_memory_mb": getattr(r, "gpu_memory_mb", 0.0) or 0.0,
            "cpu_percent": getattr(r, "cpu_percent", 0.0) or 0.0,
            "ram_mb": getattr(r, "ram_mb", 0.0) or 0.0,
        })
    return Response(data)


# -------------------------------------------------------------------------
# Jobs – list & detail
# -------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_jobs(request):
    """
    List the authenticated user's compute jobs (as a consumer).
    """
    from gpu_marketplace.models import ComputeJob

    qs = ComputeJob.objects.filter(consumer=request.user).order_by("-created_at")
    serializer = ComputeJobSerializer(qs, many=True)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_job_detail(request, job_id):
    """
    Get detailed info about a single job (including logs, events).
    """
    from gpu_marketplace.models import ComputeJob, JobEvent

    job = get_object_or_404(ComputeJob, id=job_id, consumer=request.user)
    data = ComputeJobSerializer(job).data
    events = JobEvent.objects.filter(job=job).order_by("created_at")
    data["events"] = [
        {"type": e.event_type, "description": e.description, "created_at": e.created_at}
        for e in events
    ]
    return Response(data)


# -------------------------------------------------------------------------
# Billing / Usage for the consumer
# -------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_balance(request):
    """
    Get the consumer's escrow balance.
    """
    from gpu_escrow.models import EscrowAccount

    account, _ = EscrowAccount.objects.get_or_create(user=request.user)
    return Response(
        {
            "balance_usd": account.balance_usd,
            "locked_balance_usd": account.locked_balance_usd,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_invoices(request):
    """
    List the consumer's invoices.
    """
    from gpu_finance.models import Invoice

    qs = Invoice.objects.filter(user=request.user).order_by("-created_at")
    serializer = InvoiceSerializer(qs, many=True)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_usage(request):
    """
    Get the consumer's usage records (GPU hours consumed, cost).
    """
    from gpu_finance.models import UsageRecord

    records = UsageRecord.objects.filter(user=request.user).order_by("-recorded_at")
    total_hours = records.aggregate(s=models.Sum("hours_consumed"))["s"] or 0
    total_cost = records.aggregate(s=models.Sum("cost_usd"))["s"] or 0
    data = {
        "records": [
            {
                "job_id": str(r.job_id),
                "gpu_model": r.gpu_model,
                "hours": r.hours_consumed,
                "cost": r.cost_usd,
            }
            for r in records[:50]
        ],
        "total_hours": total_hours,
        "total_cost": total_cost,
    }
    return Response(data)


# -------------------------------------------------------------------------
# Benchmark Leaderboard (global GPU comparison)
# -------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_leaderboard(request):
    """
    GPU performance leaderboard – useful for choosing a GPU.
    """
    from gpu_benchmark.models import ListingBenchmarkScore

    qs = (
        ListingBenchmarkScore.objects.select_related("listing__provider")
        .filter(composite_score__isnull=False)
        .order_by("-composite_score")[:50]
    )

    data = [
        {
            "rank": i + 1,
            "gpu_model": score.listing.gpu_model,
            "composite_score": score.composite_score,
            "training_score": score.training_score,
            "inference_score": score.inference_score,
            "price_per_hour": str(score.listing.current_price_per_hour),
            "provider": score.listing.provider.username,
        }
        for i, score in enumerate(qs)
    ]
    return Response(data)


# -------------------------------------------------------------------------
# Consumer Reputation / Disputes
# -------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def consumer_reputation(request):
    """
    Consumer's own reputation score and badges.
    """
    from gpu_reputation.models import ReputationScore, UserReputationBadge

    score, _ = ReputationScore.objects.get_or_create(user=request.user)
    badges = UserReputationBadge.objects.filter(user=request.user).select_related(
        "badge"
    )
    return Response(
        {
            "consumer_score": score.consumer_score,
            "provider_score": score.provider_score,
            "total_jobs_completed": score.total_jobs_completed,
            "badges": [
                {"name": b.badge.name, "tier": b.badge.tier, "icon": b.badge.icon}
                for b in badges
            ],
        }
    )
