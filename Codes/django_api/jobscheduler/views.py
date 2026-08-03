import os

import psutil
import torch
from deepdiff import DeepDiff
from django.db import models as db_models
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from model_versioning.models import ModelVersion
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Job, VersionedConfig
from .scheduler import scheduler
from .serializers import CustomMetricEntrySerializer, JobSerializer
from .version_utils import (
    capture_data_version,
    capture_git_info,
    create_config_snapshot,
)

# Optional distributed scheduler import
try:
    from framework.distributed.load_balancer.job_scheduler import (
        DistributedJobScheduler,
    )
except ImportError:
    DistributedJobScheduler = None


# ----------------------------------------------------------------------
# 1. Job CRUD & Scheduling
# ----------------------------------------------------------------------
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def job_list(request):
    if request.method == "GET":
        jobs = Job.objects.all().order_by("priority", "-created_at")
        return Response(JobSerializer(jobs, many=True).data)

    elif request.method == "POST":
        data = request.data
        required = ["model_name", "task_name"]
        for field in required:
            if not data.get(field):
                return Response({"error": f"Missing field: {field}"}, status=400)

        # If load balancing is enabled, submit to distributed scheduler
        if (
            data.get("load_balancer", {}).get("enabled")
            and DistributedJobScheduler is not None
        ):
            dist_scheduler = DistributedJobScheduler()
            job_id = dist_scheduler.submit(
                config=data,
                num_gpus=data.get("load_balancer", {}).get("num_gpus", 1),
                priority=data.get("priority", 5),
                user=request.user.username,
            )
            return Response(
                {"job_id": job_id, "status": "submitted_to_scheduler"}, status=202
            )

        # Original single‑job creation
        job = Job.objects.create(
            name=data.get("name", ""),
            model_name=data["model_name"],
            task_name=data["task_name"],
            attention_type=data.get("attention_type", "full"),
            method=data.get("method", "baseline"),
            epochs=data.get("epochs", 3),
            batch_size=data.get("batch_size", 16),
            learning_rate=data.get("learning_rate", 2e-5),
            max_seq_length=data.get("max_seq_length", 128),
            compression_rank=data.get("compression_rank", 64),
            gpu_requirements=data.get("gpu_requirements", 1),
            memory_requirements=data.get("memory_requirements", 8),
            priority=data.get("priority", 5),
            estimated_duration=data.get("estimated_duration", None),
            notes=data.get("notes", ""),
        )
        capture_git_info(job)
        capture_data_version(job)
        create_config_snapshot(job, "Initial configuration", request.user)
        scheduler.add_job(
            str(job.id), job.priority, job.estimated_duration, job.memory_requirements
        )

        serializer = JobSerializer(job)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def job_detail(request, job_id):
    job = get_object_or_404(Job, id=job_id)

    if request.method == "GET":
        return Response(JobSerializer(job).data)

    elif request.method == "PATCH":
        if "priority" in request.data:
            try:
                new_prio = int(request.data["priority"])
            except (ValueError, TypeError):
                return Response({"error": "Priority must be an integer"}, status=400)
            new_prio = max(1, min(10, new_prio))
            job.priority = new_prio
            if job.status == "queued":
                scheduler.update_priority(str(job.id), new_prio)

        if "notes" in request.data:
            job.notes = request.data["notes"]
        if "estimated_duration" in request.data:
            job.estimated_duration = request.data["estimated_duration"]
        if "memory_requirements" in request.data:
            job.memory_requirements = request.data["memory_requirements"]

        job.save()
        important_fields = [
            "model_name",
            "task_name",
            "attention_type",
            "method",
            "epochs",
            "batch_size",
            "learning_rate",
            "max_seq_length",
            "compression_rank",
            "gpu_requirements",
            "memory_requirements",
        ]
        if any(field in request.data for field in important_fields):
            create_config_snapshot(
                job, f"Updated by {request.user.username}", request.user
            )

        return Response(JobSerializer(job).data)

    elif request.method == "DELETE":
        if job.status in ["running", "queued"]:
            return Response(
                {"error": "Cannot delete a queued/running job. Stop it first."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        job.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ----------------------------------------------------------------------
# 2. Job Control Actions (pause, resume, stop, retry)
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_pause(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    scheduler.pause_job(str(job.id))
    return Response({"status": "paused"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_resume(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    scheduler.resume_job(str(job.id))
    return Response({"status": "resumed"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_stop(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    scheduler.stop_job(str(job.id))
    return Response({"status": "stopped"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_retry(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    if job.status not in ["failed", "stopped"]:
        return Response(
            {"error": f"Cannot retry job with status {job.status}"}, status=400
        )
    job.status = "queued"
    job.save()
    scheduler.add_job(
        str(job.id), job.priority, job.estimated_duration, job.memory_requirements
    )
    return Response({"status": "retried"})


# ----------------------------------------------------------------------
# 3. Training History (for charts)
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def job_history(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    history = job.training_history or {}
    return Response(
        {
            "steps": history.get("steps", []),
            "losses": history.get("losses", []),
            "eval_steps": history.get("eval_steps", []),
            "eval_losses": history.get("eval_losses", []),
            "eval_metrics": history.get("eval_metrics", []),
            "eval_accuracies": [
                m.get("eval_accuracy") for m in history.get("eval_metrics", [])
            ],
        }
    )


# ----------------------------------------------------------------------
# 4. Researcher Notes
# ----------------------------------------------------------------------
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def job_notes(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    if request.method == "GET":
        return Response({"notes": job.notes})
    else:
        job.notes = request.data.get("notes", "")
        job.save()
        return Response({"status": "saved"})


# ----------------------------------------------------------------------
# 5. Configuration Versioning
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_config_history(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    versions = job.config_versions.all().order_by("-version_number")
    return Response(
        {
            "versions": [
                {
                    "version": v.version_number,
                    "created_at": v.created_at,
                    "description": v.change_description,
                    "snapshot": v.config_snapshot,
                }
                for v in versions
            ]
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def compare_configs(request, job_id, version_a, version_b):
    job = get_object_or_404(Job, id=job_id)
    v1 = get_object_or_404(VersionedConfig, run=job, version_number=version_a)
    v2 = get_object_or_404(VersionedConfig, run=job, version_number=version_b)
    diff = DeepDiff(v1.config_snapshot, v2.config_snapshot, ignore_order=True)
    return Response(
        {
            "version_a": version_a,
            "version_b": version_b,
            "diff": diff,
            "values_changed": diff.get("values_changed", {}),
            "type_changes": diff.get("type_changes", {}),
            "dictionary_item_added": diff.get("dictionary_item_added", []),
            "dictionary_item_removed": diff.get("dictionary_item_removed", []),
        }
    )


# ----------------------------------------------------------------------
# 6. Custom Metrics Logging & Retrieval
# ----------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def log_custom_metric(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    serializer = CustomMetricEntrySerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(run=job)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_custom_metrics(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    metrics = job.custom_metrics.all().order_by("step")
    grouped = {}
    for m in metrics:
        if m.name not in grouped:
            grouped[m.name] = {"steps": [], "values": [], "type": m.type}
        grouped[m.name]["steps"].append(m.step)
        grouped[m.name]["values"].append(m.value)
    return Response(grouped)


# ----------------------------------------------------------------------
# 7. Environment & Code Versioning
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_environment(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    return Response(
        {
            "python_version": job.python_version,
            "pip_freeze": job.pip_freeze,
            "git": {
                "commit": job.git_commit_hash,
                "branch": job.git_branch,
                "dirty": job.git_dirty,
                "repository": job.git_repository,
            },
            "data_version": {
                "git_commit": job.data_git_commit_id,
                "dvc_hash": job.dvc_data_hash,
            },
        }
    )


# ----------------------------------------------------------------------
# 8. System & GPU Information (public)
# ----------------------------------------------------------------------
@api_view(["GET"])
def gpu_info(request):
    total = torch.cuda.device_count() if torch.cuda.is_available() else 1
    used = sum(job.gpu_requirements for job in Job.objects.filter(status="running"))
    used = min(used, total)
    memory_total_mb = memory_allocated_mb = utilization = 0
    if torch.cuda.is_available():
        memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        memory_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        utilization = (
            torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else 0
        )
    return Response(
        {
            "total_gpus": total,
            "available": total - used,
            "used": used,
            "memory_total_mb": round(memory_total_mb, 1),
            "memory_allocated_mb": round(memory_allocated_mb, 1),
            "utilization": utilization,
            "type": (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            ),
        }
    )


@api_view(["GET"])
def system_status(request):
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    return Response(
        {
            "cpu": cpu_percent,
            "memory": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 1),
            "memory_total_gb": round(memory.total / (1024**3), 1),
        }
    )


@api_view(["GET"])
def scheduler_status(request):
    return Response(
        {
            "total_gpus": scheduler.total_gpus,
            "available_gpus": scheduler.available_gpus,
            "queued_jobs": scheduler.task_queue.qsize(),
            "running_jobs": len(scheduler.running_jobs),
            "preempted_jobs": Job.objects.filter(status="preempted").count(),
        }
    )


@api_view(["GET"])
def scheduler_metrics(request):
    jobs = (
        Job.objects.filter(status="completed")
        .order_by("-completed_at")[:100]
        .values(
            "estimated_duration",
            "actual_duration",
            "gpu_requirements",
            "memory_requirements",
            "priority",
            "task_name",
        )
    )
    return Response(list(jobs))


# ----------------------------------------------------------------------
# 9. Additional Utility Endpoints
# ----------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def job_stats(request):
    total = Job.objects.count()
    by_status = {}
    for status_code, _ in Job.STATUS_CHOICES:
        by_status[status_code] = Job.objects.filter(status=status_code).count()
    avg_duration = Job.objects.filter(
        status="completed", actual_duration__isnull=False
    ).aggregate(avg=db_models.Avg("actual_duration"))["avg"]
    return Response(
        {
            "total": total,
            "by_status": by_status,
            "average_completion_seconds": (
                round(avg_duration, 2) if avg_duration else None
            ),
        }
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def bulk_delete_jobs(request):
    job_ids = request.data.get("job_ids", [])
    if not job_ids:
        return Response({"error": "No job IDs provided"}, status=400)
    jobs = Job.objects.filter(id__in=job_ids)
    deletable = jobs.exclude(status__in=["running", "queued"])
    count = deletable.count()
    deletable.delete()
    return Response({"deleted": count, "skipped": len(job_ids) - count})


# ---------- Scheduling Update ----------
@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def update_job_scheduling(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    if "priority" in request.data:
        try:
            new_prio = int(request.data["priority"])
        except (ValueError, TypeError):
            return Response({"error": "Priority must be an integer"}, status=400)
        new_prio = max(1, min(10, new_prio))
        job.priority = new_prio
        if job.status == "queued":
            scheduler.update_priority(str(job.id), new_prio)
    if "estimated_duration" in request.data:
        job.estimated_duration = request.data["estimated_duration"]
    if "memory_requirements" in request.data:
        job.memory_requirements = request.data["memory_requirements"]
    job.save()
    return Response({"status": "updated"})


# ---------- Model Versioning (if ModelVersion model exists) ----------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_model_versions(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    versions = job.model_versions.all().order_by("-version_number")
    return Response(
        [
            {
                "id": str(v.id),
                "version": v.version,
                "version_number": v.version_number,
                "artifact_size_mb": v.artifact_size_mb,
                "metrics": v.metrics,
                "created_at": v.created_at.isoformat(),
                "is_published_to_hub": v.is_published_to_hub,
            }
            for v in versions
        ]
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def publish_model_to_hub(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.run.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    from .hf_publisher import HuggingFacePublisher

    publisher = HuggingFacePublisher(version)
    hub_url = publisher.publish()
    return Response({"hub_url": hub_url, "status": "published"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def download_model_version(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    file_path = os.path.join(version.artifact_path, "model.pth")
    if not os.path.exists(file_path):
        return Response({"error": "Model file not found"}, status=404)
    response = FileResponse(open(file_path, "rb"), as_attachment=True)
    response["Content-Disposition"] = (
        f'attachment; filename="model_v{version.version_number}.pth"'
    )
    return response


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_scheduled_jobs(request):
    if DistributedJobScheduler is None:
        return Response({"error": "Distributed scheduler not available"}, status=501)
    sched = DistributedJobScheduler()
    with sched._lock:
        pending = [
            {
                "job_id": j.job_id,
                "priority": j.priority,
                "num_gpus": j.num_gpus,
                "state": j.state.value,
            }
            for j in sched._queue
            if j.user == request.user.username
        ]
        running = [
            {"job_id": j.job_id, "gpu_ids": str(j.gpu_ids)}
            for j in sched._running.values()
            if j.user == request.user.username
        ]
    return Response({"pending": pending, "running": running})


# Add these views to the existing views.py:

from .models import JobGroup


# ── Group Management ────────────────────────────────────────────────
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def group_list(request):
    if request.method == "GET":
        groups = JobGroup.objects.all()
        return Response(
            [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "guarantee_gpus": g.guarantee_gpus,
                    "max_gpus": g.max_gpus,
                    "description": g.description,
                }
                for g in groups
            ]
        )

    elif request.method == "POST":
        data = request.data
        group, created = JobGroup.objects.update_or_create(
            name=data.get("name", "default"),
            defaults={
                "weight": data.get("weight", 1.0),
                "guarantee_gpus": data.get("guarantee_gpus", 0),
                "max_gpus": data.get("max_gpus", -1),
                "description": data.get("description", ""),
            },
        )
        return Response(
            {
                "name": group.name,
                "weight": group.weight,
                "created": created,
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def group_detail(request, group_name):
    group = get_object_or_404(JobGroup, name=group_name)
    if request.method == "GET":
        return Response(
            {
                "name": group.name,
                "weight": group.weight,
                "guarantee_gpus": group.guarantee_gpus,
                "max_gpus": group.max_gpus,
                "description": group.description,
            }
        )
    elif request.method == "DELETE":
        group.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ── Scheduler Advanced Info ─────────────────────────────────────────
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def fair_share_status(request):
    return Response(
        {
            "groups": scheduler._fair_share_snapshot(),
            "enabled": scheduler.cfg.fair_share_enabled,
            "decay_half_life": scheduler.cfg.fair_share_decay_half_life,
        }
    )


@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def scheduler_config_view(request):
    if request.method == "GET":
        cfg = scheduler.cfg
        return Response(
            {
                "aging_enabled": cfg.aging_enabled,
                "aging_grace_period": cfg.aging_grace_period,
                "aging_interval": cfg.aging_interval,
                "aging_max_effective": cfg.aging_max_effective,
                "backfill_enabled": cfg.backfill_enabled,
                "backfill_interval": cfg.backfill_interval,
                "backfill_max_job_duration": cfg.backfill_max_job_duration,
                "fair_share_enabled": cfg.fair_share_enabled,
                "preempt_checkpoint_grace": cfg.preempt_checkpoint_grace,
                "preempt_min_priority_gap": cfg.preempt_min_priority_gap,
                "job_timeout": cfg.job_timeout,
            }
        )
    elif request.method == "PATCH":
        # Update mutable config fields
        for field in (
            "aging_enabled",
            "aging_grace_period",
            "aging_interval",
            "aging_max_effective",
            "backfill_enabled",
            "backfill_interval",
            "backfill_max_job_duration",
            "fair_share_enabled",
            "preempt_checkpoint_grace",
            "preempt_min_priority_gap",
            "job_timeout",
        ):
            if field in request.data:
                setattr(scheduler.cfg, field, request.data[field])
        return Response({"status": "updated"})
