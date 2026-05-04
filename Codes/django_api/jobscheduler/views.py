# jobscheduler/views.py
import torch
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import Job, VersionedConfig, CustomMetricEntry
from .serializers import JobSerializer, CustomMetricEntrySerializer
from .scheduler import scheduler
from .version_utils import create_config_snapshot, capture_git_info, capture_environment, capture_data_version

from deepdiff import DeepDiff

# ----------------------------------------------------------------------
# 1. Job CRUD & Scheduling
# ----------------------------------------------------------------------
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def job_list(request):
    if request.method == 'GET':
        jobs = Job.objects.all().order_by('priority', '-created_at')
        return Response(JobSerializer(jobs, many=True).data)

    elif request.method == 'POST':
        data = request.data
        # Validate required fields
        required = ['model_name', 'task_name']
        for field in required:
            if not data.get(field):
                return Response({"error": f"Missing field: {field}"}, status=status.HTTP_400_BAD_REQUEST)

        # Create job with full configuration
        job = Job.objects.create(
            name=data.get('name', ''),
            model_name=data['model_name'],
            task_name=data['task_name'],
            attention_type=data.get('attention_type', 'full'),
            method=data.get('method', 'baseline'),
            epochs=data.get('epochs', 3),
            batch_size=data.get('batch_size', 16),
            learning_rate=data.get('learning_rate', 2e-5),
            max_seq_length=data.get('max_seq_length', 128),
            compression_rank=data.get('compression_rank', 64),
            gpu_requirements=data.get('gpu_requirements', 1),
            memory_requirements=data.get('memory_requirements', 8),
            priority=data.get('priority', 5),
            estimated_duration=data.get('estimated_duration', None),
            notes=data.get('notes', ''),
        )
        # Capture versioning metadata (Git, DVC) and initial config snapshot
        capture_git_info(job)
        capture_data_version(job)   # optional (requires DVC)
        create_config_snapshot(job, "Initial configuration", request.user)

        # Add to scheduler
        scheduler.add_job(str(job.id), job.priority, job.estimated_duration, job.memory_requirements)

        serializer = JobSerializer(job)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def job_detail(request, job_id):
    job = get_object_or_404(Job, id=job_id)

    if request.method == 'GET':
        serializer = JobSerializer(job)
        return Response(serializer.data)

    elif request.method == 'PATCH':
        # Update fields (priority, notes, scheduling params, etc.)
        if 'priority' in request.data:
            new_prio = max(1, min(10, request.data['priority']))
            job.priority = new_prio
            if job.status == 'queued':
                scheduler.update_priority(str(job.id), new_prio)
        if 'notes' in request.data:
            job.notes = request.data['notes']
        if 'estimated_duration' in request.data:
            job.estimated_duration = request.data['estimated_duration']
        if 'memory_requirements' in request.data:
            job.memory_requirements = request.data['memory_requirements']

        job.save()
        # Create a config snapshot if important fields changed
        important_fields = ['model_name', 'task_name', 'attention_type', 'method',
                            'epochs', 'batch_size', 'learning_rate', 'max_seq_length',
                            'compression_rank', 'gpu_requirements', 'memory_requirements']
        if any(field in request.data for field in important_fields):
            create_config_snapshot(job, f"Updated by {request.user.username}", request.user)

        serializer = JobSerializer(job)
        return Response(serializer.data)

    elif request.method == 'DELETE':
        if job.status in ['running', 'queued']:
            return Response({"error": "Cannot delete a queued/running job. Stop it first."},
                            status=status.HTTP_400_BAD_REQUEST)
        job.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_job_scheduling(request, job_id):
    """Convenience endpoint to update scheduling parameters (priority, duration, memory)."""
    job = get_object_or_404(Job, id=job_id)
    if 'priority' in request.data:
        new_prio = max(1, min(10, request.data['priority']))
        job.priority = new_prio
        if job.status == 'queued':
            scheduler.update_priority(str(job.id), new_prio)
    if 'estimated_duration' in request.data:
        job.estimated_duration = request.data['estimated_duration']
    if 'memory_requirements' in request.data:
        job.memory_requirements = request.data['memory_requirements']
    job.save()
    return Response({"status": "updated"})


# ----------------------------------------------------------------------
# 2. Job Control Actions (pause, resume, stop, retry)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_pause(request, job_id):
    scheduler.pause_job(job_id)
    return Response({'status': 'paused'})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_resume(request, job_id):
    scheduler.resume_job(job_id)
    return Response({'status': 'resumed'})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_stop(request, job_id):
    scheduler.stop_job(job_id)
    return Response({'status': 'stopped'})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_retry(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    if job.status not in ['failed', 'stopped']:
        return Response({"error": f"Cannot retry job with status {job.status}"}, status=400)
    job.status = 'queued'
    job.save()
    scheduler.add_job(str(job.id), job.priority, job.estimated_duration, job.memory_requirements)
    return Response({'status': 'retried'})


# ----------------------------------------------------------------------
# 3. Training History (for charts)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_history(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    history = job.training_history or {}
    return Response({
        'steps': history.get('steps', []),
        'losses': history.get('losses', []),
        'eval_steps': history.get('eval_steps', []),
        'eval_losses': history.get('eval_losses', []),
        'eval_metrics': history.get('eval_metrics', []),
        'eval_accuracies': [m.get('eval_accuracy') for m in history.get('eval_metrics', [])],
    })


# ----------------------------------------------------------------------
# 4. Researcher Notes
# ----------------------------------------------------------------------
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def job_notes(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    if request.method == 'GET':
        return Response({'notes': job.notes})
    else:
        job.notes = request.data.get('notes', '')
        job.save()
        return Response({'status': 'saved'})


# ----------------------------------------------------------------------
# 5. Configuration Versioning
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_config_history(request, run_id):
    """Return version history of a run's configuration."""
    run = get_object_or_404(Job, id=run_id)
    versions = run.config_versions.all().order_by('-version_number')
    return Response({
        'versions': [
            {
                'version': v.version_number,
                'created_at': v.created_at,
                'description': v.change_description,
                'snapshot': v.config_snapshot,
            }
            for v in versions
        ]
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compare_configs(request, run_id, version_a, version_b):
    """Compare two configuration versions and return a detailed diff."""
    run = get_object_or_404(Job, id=run_id)
    v1 = get_object_or_404(VersionedConfig, run=run, version_number=version_a)
    v2 = get_object_or_404(VersionedConfig, run=run, version_number=version_b)
    diff = DeepDiff(v1.config_snapshot, v2.config_snapshot, ignore_order=True)
    return Response({
        'version_a': version_a,
        'version_b': version_b,
        'diff': diff,
        'values_changed': diff.get('values_changed', {}),
        'type_changes': diff.get('type_changes', {}),
        'dictionary_item_added': diff.get('dictionary_item_added', []),
        'dictionary_item_removed': diff.get('dictionary_item_removed', []),
    })


# ----------------------------------------------------------------------
# 6. Custom Metrics Logging & Retrieval
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def log_custom_metric(request, run_id):
    """Log a custom metric (called by the training script)."""
    run = get_object_or_404(Job, id=run_id)
    serializer = CustomMetricEntrySerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(run=run)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_custom_metrics(request, run_id):
    """Retrieve all custom metrics for a run, grouped by name and type."""
    run = get_object_or_404(Job, id=run_id)
    metrics = run.custom_metrics.all().order_by('step')
    grouped = {}
    for m in metrics:
        if m.name not in grouped:
            grouped[m.name] = {
                'steps': [],
                'values': [],
                'type': m.type,
            }
        grouped[m.name]['steps'].append(m.step)
        grouped[m.name]['values'].append(m.value)
    return Response(grouped)


# ----------------------------------------------------------------------
# 7. Environment & Code Versioning
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_environment(request, run_id):
    """Return captured environment (Python version, pip freeze, Git, DVC) for a run."""
    run = get_object_or_404(Job, id=run_id)
    return Response({
        'python_version': run.python_version,
        'pip_freeze': run.pip_freeze,
        'git': {
            'commit': run.git_commit_hash,
            'branch': run.git_branch,
            'dirty': run.git_dirty,
            'repository': run.git_repository,
        },
        'data_version': {
            'git_commit': run.data_git_commit_id,
            'dvc_hash': run.dvc_data_hash,
        }
    })


# ----------------------------------------------------------------------
# 8. System & GPU Information
# ----------------------------------------------------------------------
@api_view(['GET'])
def gpu_info(request):
    """Return GPU statistics (total, used, memory)."""
    total = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # Count GPUs used by running jobs (simplified: each job uses gpu_requirements)
    used = 0
    for job in Job.objects.filter(status='running'):
        used += job.gpu_requirements
    used = min(used, total)
    memory_total_mb = 0
    memory_allocated_mb = 0
    utilization = 0
    if torch.cuda.is_available():
        memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
    return Response({
        'total_gpus': total,
        'available': total - used,
        'used': used,
        'memory_total_mb': round(memory_total_mb, 1),
        'memory_allocated_mb': round(memory_allocated_mb, 1),
        'utilization': utilization,
        'type': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    })

@api_view(['GET'])
def system_status(request):
    """Return basic system metrics (CPU, memory)."""
    import psutil
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    return Response({
        'cpu': cpu_percent,
        'memory': memory.percent,
        'memory_used_gb': round(memory.used / (1024**3), 1),
        'memory_total_gb': round(memory.total / (1024**3), 1),
    })

@api_view(['GET'])
def scheduler_status(request):
    """Return current scheduler state (queue size, running jobs, etc.)."""
    return Response({
        'total_gpus': scheduler.total_gpus,
        'available_gpus': scheduler.available_gpus,
        'queued_jobs': scheduler.task_queue.qsize(),
        'running_jobs': len(scheduler.running_jobs),
        'preempted_jobs': Job.objects.filter(status='preempted').count(),
    })

@api_view(['GET'])
def scheduler_metrics(request):
    """Return historical job data for AI retraining (last 100 completed jobs)."""
    jobs = Job.objects.filter(status='completed').order_by('-completed_at')[:100].values(
        'estimated_duration', 'actual_duration', 'gpu_requirements',
        'memory_requirements', 'priority', 'task_name'
    )
    return Response(list(jobs))




@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_stats(request):
    """Return summary statistics of all jobs."""
    total = Job.objects.count()
    by_status = {
        status: Job.objects.filter(status=status).count()
        for status, _ in Job.STATUS_CHOICES
    }
    avg_duration = Job.objects.filter(status='completed', actual_duration__isnull=False).aggregate(
        avg=models.Avg('actual_duration'))['avg__actual_duration']
    return Response({
        'total': total,
        'by_status': by_status,
        'average_completion_seconds': avg_duration,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def bulk_delete_jobs(request):
    """Delete multiple jobs by their IDs (only if not running/queued)."""
    job_ids = request.data.get('job_ids', [])
    if not job_ids:
        return Response({"error": "No job IDs provided"}, status=400)
    jobs = Job.objects.filter(id__in=job_ids)
    deletable = jobs.exclude(status__in=['running', 'queued'])
    count = deletable.count()
    deletable.delete()
    return Response({"deleted": count, "skipped": len(job_ids) - count})