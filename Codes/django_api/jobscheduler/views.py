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