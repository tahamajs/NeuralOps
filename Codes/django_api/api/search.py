"""
Unified full-text search across experiments, notes, models, and jobs.
"""
from django.db.models import Q, Count
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from experiment.models import Experiment, TrainingRun
from notes.models import ResearcherNote
from jobscheduler.models import Job, ModelVersion


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def global_search(request):
    """Search across all entities with a single query."""
    query = request.query_params.get('q', '').strip()
    entity_type = request.query_params.get('type', 'all')  # all, experiments, notes, models, jobs
    limit = int(request.query_params.get('limit', 20))

    if not query:
        return Response({"results": [], "total": 0})

    results = []
    total = 0

    if entity_type in ('all', 'experiments'):
        runs = TrainingRun.objects.filter(
            Q(name__icontains=query) |
            Q(config__model_name__icontains=query) |
            Q(config__task_name__icontains=query)
        ).order_by('-created_at')[:limit]
        for r in runs:
            results.append({
                'type': 'training_run',
                'id': str(r.id),
                'title': r.name or str(r.id)[:8],
                'subtitle': f"{r.config.get('model_name', '')} → {r.config.get('task_name', '')}",
                'status': r.status,
                'created_at': r.created_at.isoformat(),
            })
        total += runs.count()

    if entity_type in ('all', 'notes'):
        notes = ResearcherNote.objects.filter(
            deleted_at__isnull=True
        ).filter(
            Q(title__icontains=query) | Q(content__icontains=query) | Q(tags__icontains=query)
        ).order_by('-created_at')[:limit]
        if not request.user.is_staff:
            notes = notes.filter(Q(is_public=True) | Q(author=request.user))
        for n in notes:
            results.append({
                'type': 'note',
                'id': str(n.id),
                'title': n.title or 'Untitled',
                'subtitle': n.content[:150],
                'author': n.author.username,
                'created_at': n.created_at.isoformat(),
            })
        total += notes.count()

    if entity_type in ('all', 'models'):
        models_qs = ModelVersion.objects.filter(
            Q(version__icontains=query) | Q(run__name__icontains=query)
        ).order_by('-created_at')[:limit]
        for m in models_qs:
            results.append({
                'type': 'model_version',
                'id': str(m.id),
                'title': m.version,
                'subtitle': f"Run: {m.run.name or str(m.run.id)[:8]}",
                'metrics': m.metrics,
                'created_at': m.created_at.isoformat(),
            })
        total += models_qs.count()

    if entity_type in ('all', 'jobs'):
        jobs = Job.objects.filter(
            Q(name__icontains=query) | Q(model_name__icontains=query) | Q(task_name__icontains=query)
        ).order_by('-created_at')[:limit]
        for j in jobs:
            results.append({
                'type': 'job',
                'id': str(j.id),
                'title': j.name or str(j.id)[:8],
                'subtitle': f"{j.model_name} / {j.task_name}",
                'status': j.status,
                'created_at': j.created_at.isoformat(),
            })
        total += jobs.count()

    # Sort combined results by created_at
    results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    results = results[:limit]

    return Response({
        "query": query,
        "total": total,
        "results": results,
    })