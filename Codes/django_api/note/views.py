from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import ResearcherNote
from .serializers import ResearcherNoteSerializer

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def note_list(request):
    if request.method == 'GET':
        # Filtering
        run_id = request.query_params.get('run')
        tag = request.query_params.get('tag')
        author = request.query_params.get('author')
        notes = ResearcherNote.objects.all()
        if run_id:
            notes = notes.filter(run_id=run_id)
        if tag:
            notes = notes.filter(tags__contains=[tag])
        if author:
            notes = notes.filter(author__username=author)
        serializer = ResearcherNoteSerializer(notes, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ResearcherNoteSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def note_detail(request, note_id):
    note = get_object_or_404(ResearcherNote, id=note_id)
    # Only author can edit/delete
    if request.method in ['PUT', 'PATCH', 'DELETE'] and note.author != request.user:
        return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

    if request.method == 'GET':
        serializer = ResearcherNoteSerializer(note)
        return Response(serializer.data)

    elif request.method in ['PUT', 'PATCH']:
        serializer = ResearcherNoteSerializer(note, data=request.data, partial=(request.method=='PATCH'))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        note.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def note_reaction(request, note_id):
    note = get_object_or_404(ResearcherNote, id=note_id)
    reaction_type = request.data.get('reaction')
    if not reaction_type:
        return Response({"error": "reaction type required"}, status=400)
    reactions = note.reactions or {}
    users = reactions.get(reaction_type, [])
    if request.user.username in users:
        users.remove(request.user.username)
    else:
        users.append(request.user.username)
    reactions[reaction_type] = users
    note.reactions = reactions
    note.save()
    return Response({"status": "updated", "reactions": reactions})



@api_view(['POST'])
def log_custom_metric(request, run_id):
    """Endpoint for logging custom metrics during training"""
    run = get_object_or_404(Job, id=run_id)
    
    metric_type = request.data.get('type')
    name = request.data.get('name')
    value = request.data.get('value')
    step = request.data.get('step')
    epoch = request.data.get('epoch')
    
    # For images and figures, handle base64 data
    media_url = None
    if metric_type in ['image', 'figure']:
        # Optionally save to media storage and store URL
        pass
    
    CustomMetricEntry.objects.create(
        run=run,
        name=name,
        type=metric_type,
        value=value,
        step=step,
        epoch=epoch,
        media_url=media_url
    )
    
    
    return Response({'status': 'logged'})

@api_view(['GET'])
def get_custom_metrics(request, run_id):
    """Retrieve all custom metrics for a run"""
    run = get_object_or_404(Job, id=run_id)
    
    # Group metrics by name for easy plotting
    metrics_by_name = {}
    for metric in run.custom_metrics.all().order_by('step'):
        if metric.name not in metrics_by_name:
            metrics_by_name[metric.name] = {
                'steps': [],
                'values': [],
                'type': metric.type
            }
        metrics_by_name[metric.name]['steps'].append(metric.step)
        metrics_by_name[metric.name]['values'].append(metric.value)
    
    return Response(metrics_by_name)