# notes/views.py
import csv
import json
import logging
from io import StringIO
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.db.models import Q
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from .models import ResearcherNote, NoteImage, WebhookSubscription
from .serializers import (ResearcherNoteSerializer, NoteImageSerializer,
                          WebhookSubscriptionSerializer)
from .permissions import IsOwnerOrReadOnly, IsResearcherOrReadOnly
from .webhook_utils import trigger_webhook_event
from .filters import NoteFilter

logger = logging.getLogger(__name__)

class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

# ----------------------------------------------------------------------
# 1. Note List & Create (with advanced filtering, versioning)
# ----------------------------------------------------------------------
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def note_list(request, version='v1'):
    """List notes with advanced filtering, pagination, and export support."""
    if request.method == 'GET':
        # Use Django FilterSet for advanced filtering
        from .filters import NoteFilter
        notes = ResearcherNote.objects.filter(deleted_at__isnull=True)
        filter_set = NoteFilter(request.query_params, queryset=notes)
        notes = filter_set.qs
        
        # Additional permission filtering
        if not request.user.is_staff:
            notes = notes.filter(Q(is_public=True) | Q(author=request.user))
        
        # Export as CSV if requested
        if request.query_params.get('export') == 'csv':
            return export_notes_csv(notes)
        if request.query_params.get('export') == 'json':
            serializer = ResearcherNoteSerializer(notes, many=True)
            return Response(serializer.data)
        
        # Pagination
        paginator = StandardPagination()
        page = paginator.paginate_queryset(notes, request)
        serializer = ResearcherNoteSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)
    
    elif request.method == 'POST':
        serializer = ResearcherNoteSerializer(data=request.data)
        if serializer.is_valid():
            note = serializer.save(author=request.user)
            # Log metric: note created
            logger.info(f"Note created", extra={'user': request.user.username, 'note_id': str(note.id), 'event': 'note.created'})
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ----------------------------------------------------------------------
# 2. Note Detail, Update, Delete
# ----------------------------------------------------------------------
@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated, IsOwnerOrReadOnly])
def note_detail(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    
    if request.method == 'GET':
        serializer = ResearcherNoteSerializer(note)
        return Response(serializer.data)
    
    elif request.method in ['PUT', 'PATCH']:
        serializer = ResearcherNoteSerializer(note, data=request.data, partial=(request.method=='PATCH'))
        if serializer.is_valid():
            serializer.save()
            if 'content' in request.data:
                note.increment_version()
                logger.info(f"Note updated", extra={'user': request.user.username, 'note_id': str(note.id), 'event': 'note.updated'})
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        note.soft_delete()
        logger.info(f"Note deleted", extra={'user': request.user.username, 'note_id': str(note.id), 'event': 'note.deleted'})
        return Response(status=status.HTTP_204_NO_CONTENT)

# ----------------------------------------------------------------------
# 3. Reactions
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def note_reaction(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
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
    logger.info(f"Reaction added to note", extra={'user': request.user.username, 'note_id': str(note.id), 'reaction': reaction_type})
    return Response({"status": "updated", "reactions": reactions})

# ----------------------------------------------------------------------
# 4. Image Upload
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_note_image(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    serializer = NoteImageSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(note=note)
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)

# ----------------------------------------------------------------------
# 5. Batch Operations
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated, IsResearcherOrReadOnly])
def bulk_notes(request, version='v1'):
    """Bulk create, update, or delete notes."""
    action = request.data.get('action')
    if not action:
        return Response({"error": "action required"}, status=400)
    
    if action == 'delete':
        note_ids = request.data.get('ids', [])
        notes = ResearcherNote.objects.filter(id__in=note_ids, author=request.user, deleted_at__isnull=True)
        count = notes.count()
        for note in notes:
            note.soft_delete()
        return Response({"deleted": count})
    
    elif action == 'create':
        items = request.data.get('items', [])
        created = []
        for item in items:
            serializer = ResearcherNoteSerializer(data=item)
            if serializer.is_valid():
                note = serializer.save(author=request.user)
                created.append(str(note.id))
            else:
                return Response(serializer.errors, status=400)
        return Response({"created": created})
    
    return Response({"error": f"Unknown action {action}"}, status=400)

# ----------------------------------------------------------------------
# 6. Export (CSV/JSON) – reused in note_list
# ----------------------------------------------------------------------
def export_notes_csv(queryset):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Title', 'Author', 'Run ID', 'Tags', 'Created At'])
    for note in queryset:
        writer.writerow([note.id, note.title, note.author.username, note.run_id, ','.join(note.tags), note.created_at])
    response = HttpResponse(output.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="notes_export.csv"'
    return response

# ----------------------------------------------------------------------
# 7. Webhook Management
# ----------------------------------------------------------------------
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def webhook_list(request, version='v1'):
    if request.method == 'GET':
        subs = WebhookSubscription.objects.filter(user=request.user)
        serializer = WebhookSubscriptionSerializer(subs, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = WebhookSubscriptionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def webhook_delete(request, webhook_id, version='v1'):
    sub = get_object_or_404(WebhookSubscription, id=webhook_id, user=request.user)
    sub.delete()
    return Response(status=204)

# ----------------------------------------------------------------------
# 8. Versioned URL entry point (wrapper)
# ----------------------------------------------------------------------
def versioned_view(view_func):
    def wrapper(request, *args, **kwargs):
        version = request.resolver_match.kwargs.get('version', 'v1')
        return view_func(request, version=version, *args, **kwargs)
    return wrapper


# ============================================================================
# 8. Note Versions (history)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_versions(request, note_id, version='v1'):
    """List all versions of a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not request.user.is_staff and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    versions = note.versions.all().order_by('-version_number')
    data = [
        {
            'version': v.version_number,
            'content_preview': v.content_snapshot[:200],
            'changed_by': v.changed_by.username if v.changed_by else None,
            'change_reason': v.change_reason,
            'created_at': v.created_at
        }
        for v in versions
    ]
    return Response(data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_version_detail(request, note_id, version_number, version='v1'):
    """Get a specific version snapshot."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not request.user.is_staff and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    version_obj = get_object_or_404(note.versions, version_number=version_number)
    return Response({
        'version': version_obj.version_number,
        'title': version_obj.title_snapshot or note.title,
        'content': version_obj.content_snapshot,
        'changed_by': version_obj.changed_by.username if version_obj.changed_by else None,
        'change_reason': version_obj.change_reason,
        'created_at': version_obj.created_at
    })

# ============================================================================
# 9. Mentions
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_mentions(request, note_id, version='v1'):
    """List all mentions in a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not request.user.is_staff and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    mentions = note.mentions.select_related('mentioned_user').all()
    data = [
        {
            'user': m.mentioned_user.username,
            'is_read': m.is_read,
            'created_at': m.created_at
        }
        for m in mentions
    ]
    return Response(data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_mention_read(request, mention_id, version='v1'):
    """Mark a specific mention as read."""
    mention = get_object_or_404(NoteMention, id=mention_id, mentioned_user=request.user)
    mention.is_read = True
    mention.save()
    return Response({'status': 'marked_read'})

# ============================================================================
# 10. Replies (threading)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_replies(request, note_id, version='v1'):
    """List all replies to a note (threaded comments)."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True, parent_note__isnull=True)
    replies = note.replies.filter(deleted_at__isnull=True).order_by('created_at')
    serializer = ResearcherNoteSerializer(replies, many=True)
    return Response(serializer.data)

# ============================================================================
# 11. Restore soft-deleted note
# ============================================================================
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def restore_note(request, note_id, version='v1'):
    """Restore a soft-deleted note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=False)
    if note.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    note.deleted_at = None
    note.save(update_fields=['deleted_at'])
    logger.info(f"Note restored", extra={'user': request.user.username, 'note_id': str(note.id), 'event': 'note.restored'})
    return Response({"status": "restored"})

# ============================================================================
# 12. Export note as PDF (requires reportlab or weasyprint)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def export_note_pdf(request, note_id, version='v1'):
    """Export a single note as a PDF file."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not note.is_public and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    # Simplified PDF generation using weasyprint (install: pip install weasyprint)
    from weasyprint import HTML
    html_content = f"""
    <html>
    <head><title>{note.title or 'Note'}</title></head>
    <body>
        <h1>{note.title or 'Untitled'}</h1>
        <p>Author: {note.author.username}</p>
        <p>Created: {note.created_at}</p>
        <hr>
        <div>{note.content}</div>
    </body>
    </html>
    """
    pdf_file = HTML(string=html_content).write_pdf()
    response = HttpResponse(pdf_file, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="note_{note_id}.pdf"'
    return response

# ============================================================================
# 13. Test Webhook (send test payload)
# ============================================================================
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def test_webhook(request, webhook_id, version='v1'):
    """Send a test payload to a registered webhook URL."""
    sub = get_object_or_404(WebhookSubscription, id=webhook_id, user=request.user)
    test_payload = {
        'event': 'test',
        'timestamp': timezone.now().isoformat(),
        'data': {'message': 'This is a test webhook payload'}
    }
    success = trigger_webhook_event.manual_send(sub.url, test_payload, sub.secret)
    return Response({"success": success, "url": sub.url})

# ============================================================================
# 14. Statistics
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def notes_statistics(request, version='v1'):
    """Aggregated statistics about notes."""
    from django.db.models import Count
    total_notes = ResearcherNote.objects.filter(deleted_at__isnull=True).count()
    public_notes = ResearcherNote.objects.filter(deleted_at__isnull=True, is_public=True).count()
    notes_by_author = ResearcherNote.objects.filter(deleted_at__isnull=True).values('author__username').annotate(count=Count('id')).order_by('-count')[:10]
    notes_by_run = ResearcherNote.objects.filter(deleted_at__isnull=True, run__isnull=False).values('run__id').annotate(count=Count('id')).order_by('-count')[:10]
    return Response({
        'total_notes': total_notes,
        'public_notes': public_notes,
        'private_notes': total_notes - public_notes,
        'top_authors': list(notes_by_author),
        'top_runs': list(notes_by_run)
    })

# ============================================================================
# 15. Recent Notes (last N days)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recent_notes(request, version='v1'):
    """Get notes created or updated in the last N days (default 7)."""
    days = int(request.query_params.get('days', 7))
    from datetime import timedelta
    cutoff = timezone.now() - timedelta(days=days)
    notes = ResearcherNote.objects.filter(
        deleted_at__isnull=True,
        updated_at__gte=cutoff
    ).order_by('-updated_at')
    if not request.user.is_staff:
        notes = notes.filter(Q(is_public=True) | Q(author=request.user))
    paginator = StandardPagination()
    page = paginator.paginate_queryset(notes, request)
    serializer = ResearcherNoteSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)

# ============================================================================
# 16. Bulk Tag Operations
# ============================================================================
@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def bulk_tags(request, version='v1'):
    """Add or remove tags from multiple notes."""
    action = request.data.get('action')
    note_ids = request.data.get('ids', [])
    tags = request.data.get('tags', [])
    if not action or not note_ids or not tags:
        return Response({"error": "action, ids, and tags required"}, status=400)
    
    notes = ResearcherNote.objects.filter(id__in=note_ids, author=request.user, deleted_at__isnull=True)
    if action == 'add':
        for note in notes:
            new_tags = list(set(note.tags + tags))
            note.tags = new_tags
            note.save(update_fields=['tags'])
    elif action == 'remove':
        for note in notes:
            new_tags = [t for t in note.tags if t not in tags]
            note.tags = new_tags
            note.save(update_fields=['tags'])
    else:
        return Response({"error": "action must be 'add' or 'remove'"}, status=400)
    return Response({"updated": notes.count()})