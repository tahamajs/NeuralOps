"""
Experiment configuration templates – save, load, share, and reuse experiment configs.
"""
import uuid
from django.db import models
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import ExperimentTemplate


class ExperimentTemplate(models.Model):
    """Reusable experiment configuration template."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    author = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='templates')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    config = models.JSONField(default=dict)
    category = models.CharField(max_length=100, blank=True, default="general")
    is_public = models.BooleanField(default=False)
    usage_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-usage_count', '-created_at']
        indexes = [
            models.Index(fields=['category']),
            models.Index(fields=['is_public']),
            models.Index(fields=['author']),
        ]

    def __str__(self):
        return self.name


# ---- Views ----
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def template_list(request):
    if request.method == 'GET':
        category = request.query_params.get('category', '')
        public_only = request.query_params.get('public', 'true').lower() == 'true'
        qs = ExperimentTemplate.objects.all()
        if public_only:
            qs = qs.filter(Q(is_public=True) | Q(author=request.user))
        else:
            qs = qs.filter(author=request.user)
        if category:
            qs = qs.filter(category=category)
        qs = qs.order_by('-usage_count', '-created_at')
        data = [{
            'id': str(t.id), 'name': t.name, 'description': t.description,
            'category': t.category, 'is_public': t.is_public,
            'usage_count': t.usage_count, 'config': t.config,
            'author': t.author.username, 'created_at': t.created_at.isoformat(),
        } for t in qs]
        return Response(data)

    elif request.method == 'POST':
        name = request.data.get('name', '').strip()
        if not name:
            return Response({"error": "name required"}, status=400)
        template = ExperimentTemplate.objects.create(
            author=request.user,
            name=name,
            description=request.data.get('description', ''),
            config=request.data.get('config', {}),
            category=request.data.get('category', 'general'),
            is_public=request.data.get('is_public', False),
        )
        return Response({'id': str(template.id), 'status': 'created'}, status=201)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def template_detail(request, template_id):
    template = get_object_or_404(ExperimentTemplate, id=template_id)
    if request.method == 'GET':
        return Response({
            'id': str(template.id), 'name': template.name,
            'description': template.description, 'category': template.category,
            'is_public': template.is_public, 'config': template.config,
            'author': template.author.username, 'usage_count': template.usage_count,
            'created_at': template.created_at.isoformat(),
        })
    elif request.method == 'PATCH':
        if template.author != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        for field in ['name', 'description', 'config', 'category', 'is_public']:
            if field in request.data:
                setattr(template, field, request.data[field])
        template.save()
        return Response({'status': 'updated'})
    elif request.method == 'DELETE':
        if template.author != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        template.delete()
        return Response(status=204)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def use_template(request, template_id):
    """Increment usage count when a template is used."""
    template = get_object_or_404(ExperimentTemplate, id=template_id)
    template.usage_count += 1
    template.save(update_fields=['usage_count'])
    return Response({'config': template.config, 'usage_count': template.usage_count})