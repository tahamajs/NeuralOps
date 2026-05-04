from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Job
from .serializers import JobSerializer
from .scheduler import scheduler
import torch

@api_view(['GET', 'POST'])
def job_list(request):
    if request.method == 'GET':
        jobs = Job.objects.all().order_by('priority', '-created_at')
        return Response(JobSerializer(jobs, many=True).data)
    elif request.method == 'POST':
        data = request.data
        job = Job.objects.create(
            name=data.get('name', ''),
            model_name=data.get('model_name', 'distilbert-base-uncased'),
            task_name=data.get('task_name', 'mrpc'),
            gpu_requirements=data.get('gpu_requirements', 1),
            priority=data.get('priority', 5),
            notes=data.get('notes', ''),
        )
        scheduler.add_job(str(job.id), job.priority)
        return Response(JobSerializer(job).data, status=201)

@api_view(['GET', 'PATCH'])
def job_detail(request, job_id):
    job = Job.objects.get(id=job_id)
    if request.method == 'GET':
        return Response(JobSerializer(job).data)
    elif request.method == 'PATCH':
        if 'priority' in request.data:
            job.priority = request.data['priority']
            job.save()
        return Response(JobSerializer(job).data)

@api_view(['POST'])
def job_pause(request, job_id):
    scheduler.pause_job(job_id)
    return Response({'status': 'paused'})

@api_view(['POST'])
def job_resume(request, job_id):
    scheduler.resume_job(job_id)
    return Response({'status': 'resumed'})

@api_view(['POST'])
def job_stop(request, job_id):
    scheduler.stop_job(job_id)
    return Response({'status': 'stopped'})

@api_view(['POST'])
def job_retry(request, job_id):
    job = Job.objects.get(id=job_id)
    job.status = 'queued'
    job.save()
    scheduler.add_job(job_id, job.priority)
    return Response({'status': 'retried'})

@api_view(['GET'])
def job_history(request, job_id):
    job = Job.objects.get(id=job_id)
    history = job.training_history
    return Response({
        'steps': history.get('steps', []),
        'losses': history.get('losses', []),
        'eval_steps': history.get('eval_steps', []),
        'eval_accuracies': [m.get('eval_accuracy') for m in history.get('eval_metrics', [])]
    })

@api_view(['GET', 'POST'])
def job_notes(request, job_id):
    job = Job.objects.get(id=job_id)
    if request.method == 'GET':
        return Response({'notes': job.notes})
    else:
        job.notes = request.data.get('notes', '')
        job.save()
        return Response({'status': 'saved'})

@api_view(['GET'])
def gpu_info(request):
    total = torch.cuda.device_count() if torch.cuda.is_available() else 1
    used = Job.objects.filter(status='running').count()
    return Response({
        'total_gpus': total,
        'available': total - used,
        'used': used
    })


# jobscheduler/views.py

@api_view(['GET'])
def get_config_history(request, run_id):
    """Get the version history of a run's configuration"""
    run = get_object_or_404(Job, id=run_id)
    versions = run.config_versions.all().order_by('-version_number')
    
    return Response({
        'versions': [
            {
                'version': v.version_number,
                'created_at': v.created_at,
                'description': v.change_description,
                'snapshot': v.config_snapshot
            }
            for v in versions
        ]
    })

@api_view(['GET'])
def compare_configs(request, run_id, version_a, version_b):
    """Compare two configuration versions and return the diff"""
    run = get_object_or_404(Job, id=run_id)
    v1 = run.config_versions.get(version_number=version_a)
    v2 = run.config_versions.get(version_number=version_b)
    
    from deepdiff import DeepDiff
    diff = DeepDiff(v1.config_snapshot, v2.config_snapshot, ignore_order=True)
    
    return Response({
        'version_a': version_a,
        'version_b': version_b,
        'diff': diff
    })




# jobscheduler/views.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import Job, CustomMetricEntry
from .serializers import CustomMetricEntrySerializer

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def log_custom_metric(request, run_id):
    """Endpoint for logging custom metrics during training"""
    run = get_object_or_404(Job, id=run_id)
    metric_type = request.data.get('type')
    name = request.data.get('name')
    value = request.data.get('value')
    step = request.data.get('step')
    epoch = request.data.get('epoch')

    CustomMetricEntry.objects.create(
        run=run,
        name=name,
        type=metric_type,
        value=value,
        step=step,
        epoch=epoch,
        media_url=request.data.get('media_url')
    )
    return Response({'status': 'logged'}, status=status.HTTP_201_CREATED)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_custom_metrics(request, run_id):
    """Retrieve all custom metrics for a run, grouped by name"""
    run = get_object_or_404(Job, id=run_id)
    metrics_by_name = {}
    for metric in run.custom_metrics.all().order_by('step'):
        if metric.name not in metrics_by_name:
            metrics_by_name[metric.name] = {
                'steps': [], 'values': [], 'type': metric.type
            }
        metrics_by_name[metric.name]['steps'].append(metric.step)
        metrics_by_name[metric.name]['values'].append(metric.value)
    return Response(metrics_by_name)




# jobscheduler/views.py
@api_view(['GET'])
def get_config_history(request, run_id):
    """Return version history of a run's configuration"""
    run = get_object_or_404(Job, id=run_id)
    versions = run.config_versions.all().order_by('-version_number')
    return Response({
        'versions': [
            {
                'version': v.version_number,
                'created_at': v.created_at,
                'description': v.change_description,
                'snapshot': v.config_snapshot
            }
            for v in versions
        ]
    })

@api_view(['GET'])
def compare_configs(request, run_id, version_a, version_b):
    """Compare two configuration versions using DeepDiff"""
    from deepdiff import DeepDiff
    run = get_object_or_404(Job, id=run_id)
    v1 = run.config_versions.get(version_number=version_a)
    v2 = run.config_versions.get(version_number=version_b)
    diff = DeepDiff(v1.config_snapshot, v2.config_snapshot, ignore_order=True)
    return Response({
        'version_a': version_a,
        'version_b': version_b,
        'diff': diff,
        'values_changed': diff.get('values_changed', {}),
        'type_changes': diff.get('type_changes', {}),
        'dictionary_item_added': diff.get('dictionary_item_added', []),
        'dictionary_item_removed': diff.get('dictionary_item_removed', [])
    })