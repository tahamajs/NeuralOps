# notes/views.py
import csv
import json
import logging
from io import StringIO
from datetime import timedelta

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.db.models import Q, Count
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import ResearcherNote, NoteImage, WebhookSubscription, NoteMention
from .serializers import (ResearcherNoteSerializer, NoteImageSerializer,
                          WebhookSubscriptionSerializer)
from .permissions import IsOwnerOrReadOnly, IsResearcherOrReadOnly
from .webhook_utils import trigger_webhook_event
from .filters import NoteFilter
from django.http import Http404

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

# ─── به ابتدای فایل اضافه کنید ───
from .report_generator import generate_experiment_report
from .models import NoteTemplate, NoteComment, Notification
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db.models import F, Value, CharField
from django.db.models.functions import Concat


# ═══════════════════════════════════════════════════════════════════
# 18. Auto-Generate Experiment Report
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_run_report(request, run_id, version='v1'):
    """Auto-generate a rich Markdown report from a training run."""
    note = generate_experiment_report(run_id=run_id, author=request.user, save=True)
    if note is None:
        return Response({"error": "Run not found"}, status=404)
    serializer = ResearcherNoteSerializer(note)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


# ═══════════════════════════════════════════════════════════════════
# 19. Note Templates
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def note_templates(request, version='v1'):
    """List all templates or create a new one."""
    if request.method == 'GET':
        note_type = request.query_params.get('note_type')
        templates = NoteTemplate.objects.all()
        if note_type:
            templates = templates.filter(note_type=note_type)
        
        data = [
            {
                'id': str(t.id),
                'name': t.name,
                'description': t.description,
                'note_type': t.note_type,
                'variables_schema': t.variables_schema,
                'is_default': t.is_default,
            }
            for t in templates
        ]
        return Response(data)
    
    elif request.method == 'POST':
        template = NoteTemplate.objects.create(
            name=request.data.get('name', 'Untitled Template'),
            description=request.data.get('description', ''),
            note_type=request.data.get('note_type', 'general'),
            content_template=request.data.get('content_template', ''),
            variables_schema=request.data.get('variables_schema', {}),
            created_by=request.user,
        )
        return Response({'id': str(template.id), 'status': 'created'}, status=201)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def render_template(request, template_id, version='v1'):
    """Render a template with provided variables and create a note."""
    template = get_object_or_404(NoteTemplate, id=template_id)
    variables = request.data.get('variables', {})
    content = template.render(variables)
    
    note = ResearcherNote.objects.create(
        author=request.user,
        title=request.data.get('title', template.name),
        content=content,
        note_type=template.note_type,
        tags=request.data.get('tags', []),
        run_id=request.data.get('run_id'),
    )
    serializer = ResearcherNoteSerializer(note)
    return Response(serializer.data, status=201)


# ═══════════════════════════════════════════════════════════════════
# 20. Full-Text Search
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def fulltext_search(request, version='v1'):
    """Full-text search across all notes using PostgreSQL."""
    query = request.query_params.get('q', '').strip()
    if not query:
        return Response({"results": [], "total": 0})
    
    search_query = SearchQuery(query, config='english')
    search_vector = SearchVector('title', weight='A', config='english') + \
                    SearchVector('content', weight='B', config='english')
    
    notes = ResearcherNote.objects.filter(
        deleted_at__isnull=True
    ).annotate(
        rank=SearchRank(search_vector, search_query)
    ).filter(rank__gte=0.05).order_by('-rank')
    
    if not request.user.is_staff:
        notes = notes.filter(models.Q(is_public=True) | models.Q(author=request.user))
    
    paginator = StandardPagination()
    page = paginator.paginate_queryset(notes, request)
    
    data = [
        {
            'id': str(n.id),
            'title': n.title,
            'content_preview': n.content[:300],
            'author': n.author.username,
            'note_type': n.note_type,
            'tags': n.tags,
            'rank': round(n.rank, 4),
            'created_at': n.created_at,
        }
        for n in page
    ]
    return paginator.get_paginated_response(data)


# ═══════════════════════════════════════════════════════════════════
# 21. Favorite / Bookmark
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_favorite(request, note_id, version='v1'):
    """Toggle favorite/bookmark status of a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not note.is_public and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    note.toggle_favorite()
    return Response({"is_favorite": note.is_favorite})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def favorite_notes(request, version='v1'):
    """List all favorite/bookmarked notes for the user."""
    notes = ResearcherNote.objects.filter(
        author=request.user, is_favorite=True, deleted_at__isnull=True
    ).order_by('-updated_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(notes, request)
    serializer = ResearcherNoteSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ═══════════════════════════════════════════════════════════════════
# 22. Comments on Notes
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def note_comments(request, note_id, version='v1'):
    """Get or add comments on a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    
    if request.method == 'GET':
        comments = note.comments.filter(parent_comment__isnull=True).order_by('created_at')
        data = [
            {
                'id': str(c.id),
                'author': c.author.username,
                'content': c.content,
                'is_resolved': c.is_resolved,
                'created_at': c.created_at,
                'replies': [
                    {
                        'id': str(r.id),
                        'author': r.author.username,
                        'content': r.content,
                        'created_at': r.created_at,
                    }
                    for r in c.replies.all()
                ],
            }
            for c in comments
        ]
        return Response(data)
    
    elif request.method == 'POST':
        comment = NoteComment.objects.create(
            note=note,
            author=request.user,
            content=request.data.get('content', ''),
            parent_comment_id=request.data.get('parent_comment_id'),
        )
        return Response({'id': str(comment.id), 'status': 'created'}, status=201)


# ═══════════════════════════════════════════════════════════════════
# 23. Dashboard / Analytics
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def notes_dashboard(request, version='v1'):
    """Analytics dashboard for notes."""
    base_qs = ResearcherNote.objects.filter(deleted_at__isnull=True)
    user_qs = base_qs.filter(author=request.user)
    
    return Response({
        'total_notes': base_qs.count(),
        'my_notes': user_qs.count(),
        'public_notes': base_qs.filter(is_public=True).count(),
        'favorites': user_qs.filter(is_favorite=True).count(),
        'pinned': user_qs.filter(is_pinned=True).count(),
        'by_type': {
            t[0]: user_qs.filter(note_type=t[0]).count()
            for t in ResearcherNote.NOTE_TYPES
        },
        'by_tag': _tag_distribution(user_qs),
        'recent_activity': list(
            user_qs.order_by('-updated_at')[:5].values('id', 'title', 'note_type', 'updated_at')
        ),
        'linked_runs': user_qs.filter(run__isnull=False).count(),
        'total_comments': NoteComment.objects.filter(note__author=request.user).count(),
        'total_mentions': NoteMention.objects.filter(mentioned_user=request.user, is_read=False).count(),
    })


def _tag_distribution(queryset):
    """Compute tag frequency across notes."""
    tag_counts = {}
    for note in queryset:
        for tag in (note.tags or []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])


# ═══════════════════════════════════════════════════════════════════
# 24. Export as Markdown
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def export_note_markdown(request, note_id, version='v1'):
    """Export note as raw Markdown file."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not note.is_public and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    
    md_content = f"""---
title: "{note.title}"
author: "{note.author.username}"
date: "{note.created_at.isoformat()}"
tags: {json.dumps(note.tags)}
note_type: "{note.note_type}"
version: {note.version}
---

{note.content}
"""
    response = HttpResponse(md_content, content_type='text/markdown')
    response['Content-Disposition'] = f'attachment; filename="note_{note.slug or note.id}.md"'
    return response


# ═══════════════════════════════════════════════════════════════════
# 25. Notifications
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_notifications(request, version='v1'):
    """List user's notifications."""
    notifs = Notification.objects.filter(recipient=request.user).order_by('-created_at')
    unread_only = request.query_params.get('unread_only', 'false').lower() == 'true'
    if unread_only:
        notifs = notifs.filter(is_read=False)
    
    paginator = StandardPagination()
    page = paginator.paginate_queryset(notifs, request)
    data = [
        {
            'id': str(n.id),
            'type': n.type,
            'content': n.content,
            'sender': n.sender.username if n.sender else None,
            'note_id': str(n.note_id) if n.note else None,
            'is_read': n.is_read,
            'created_at': n.created_at,
        }
        for n in page
    ]
    return paginator.get_paginated_response(data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_all_notifications_read(request, version='v1'):
    """Mark all notifications as read."""
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    return Response({"status": "all marked read"})
# ----------------------------------------------------------------------
# 1. Note List & Create
# ----------------------------------------------------------------------
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def note_list(request, version='v1'):
    if request.method == 'GET':
        notes = ResearcherNote.objects.filter(deleted_at__isnull=True)
        filter_set = NoteFilter(request.query_params, queryset=notes)
        notes = filter_set.qs

        if not request.user.is_staff:
            notes = notes.filter(Q(is_public=True) | Q(author=request.user))

        if request.query_params.get('export') == 'csv':
            return export_notes_csv(notes)
        if request.query_params.get('export') == 'json':
            serializer = ResearcherNoteSerializer(notes, many=True)
            return Response(serializer.data)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(notes, request)
        serializer = ResearcherNoteSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == 'POST':
        serializer = ResearcherNoteSerializer(data=request.data)
        if serializer.is_valid():
            note = serializer.save(author=request.user)
            logger.info("Note created", extra={
                'user': request.user.username,
                'note_id': str(note.id),
                'event': 'note.created'
            })
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ----------------------------------------------------------------------
# 2. Note Detail, Update, Delete
# ----------------------------------------------------------------------
@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def note_detail(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)

    # ----- Privacy & Permission Guard -----
    if not note.is_public and note.author != request.user and not request.user.is_staff:
        # For GET requests, hide the existence of the note entirely
        if request.method == 'GET':
            raise Http404
        # For any modifying request, return Forbidden (403)
        return Response({"error": "Not allowed"}, status=403)

    # For GET, everyone who passed the guard can see the note
    if request.method == 'GET':
        serializer = ResearcherNoteSerializer(note)
        return Response(serializer.data)

    # For modifying requests, only author or staff are allowed
    if note.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)

    if request.method == 'PUT' or request.method == 'PATCH':
        partial = (request.method == 'PATCH')
        serializer = ResearcherNoteSerializer(note, data=request.data, partial=partial)
        if serializer.is_valid():
            serializer.save()
            if 'content' in request.data:
                logger.info("Note updated", extra={
                    'user': request.user.username,
                    'note_id': str(note.id),
                    'event': 'note.updated'
                })
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        note.soft_delete()
        logger.info("Note deleted", extra={
            'user': request.user.username,
            'note_id': str(note.id),
            'event': 'note.deleted'
        })
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
    logger.info("Reaction added", extra={
        'user': request.user.username,
        'note_id': str(note.id),
        'reaction': reaction_type
    })
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
# 5. Bulk Operations
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def bulk_notes(request, version='v1'):
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
# 6. Export CSV
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
# 8. Versioned URL wrapper
# ----------------------------------------------------------------------
def versioned_view(view_func):
    def wrapper(request, *args, **kwargs):
        version = request.resolver_match.kwargs.get('version', 'v1')
        return view_func(request, version=version, *args, **kwargs)
    return wrapper


# ----------------------------------------------------------------------
# 9. Note Versions (history)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_versions(request, note_id, version='v1'):
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


# ----------------------------------------------------------------------
# 10. Mentions
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_mentions(request, note_id, version='v1'):
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
    mention = get_object_or_404(NoteMention, id=mention_id, mentioned_user=request.user)
    mention.is_read = True
    mention.save()
    return Response({'status': 'marked_read'})


# ----------------------------------------------------------------------
# 11. Replies
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_replies(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True, parent_note__isnull=True)
    replies = note.replies.filter(deleted_at__isnull=True).order_by('created_at')
    serializer = ResearcherNoteSerializer(replies, many=True)
    return Response(serializer.data)


# ----------------------------------------------------------------------
# 12. Restore
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def restore_note(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=False)
    if note.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    note.deleted_at = None
    note.save(update_fields=['deleted_at'])
    logger.info("Note restored", extra={
        'user': request.user.username,
        'note_id': str(note.id),
        'event': 'note.restored'
    })
    return Response({"status": "restored"})


# ----------------------------------------------------------------------
# 13. Export as PDF
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def export_note_pdf(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not note.is_public and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
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


# ----------------------------------------------------------------------
# 14. Test Webhook
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def test_webhook(request, webhook_id, version='v1'):
    from .webhook_utils import manual_send
    sub = get_object_or_404(WebhookSubscription, id=webhook_id, user=request.user)
    test_payload = {
        'event': 'test',
        'timestamp': timezone.now().isoformat(),
        'data': {'message': 'This is a test webhook payload'}
    }
    success = manual_send(sub.url, test_payload, sub.secret)
    return Response({"success": success, "url": sub.url})


# ----------------------------------------------------------------------
# 15. Statistics
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def notes_statistics(request, version='v1'):
    total_notes = ResearcherNote.objects.filter(deleted_at__isnull=True).count()
    public_notes = ResearcherNote.objects.filter(deleted_at__isnull=True, is_public=True).count()
    notes_by_author = ResearcherNote.objects.filter(deleted_at__isnull=True) \
        .values('author__username').annotate(count=Count('id')).order_by('-count')[:10]
    notes_by_run = ResearcherNote.objects.filter(deleted_at__isnull=True, run__isnull=False) \
        .values('run__id').annotate(count=Count('id')).order_by('-count')[:10]
    return Response({
        'total_notes': total_notes,
        'public_notes': public_notes,
        'private_notes': total_notes - public_notes,
        'top_authors': list(notes_by_author),
        'top_runs': list(notes_by_run)
    })


# ----------------------------------------------------------------------
# 16. Recent Notes
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recent_notes(request, version='v1'):
    days = int(request.query_params.get('days', 7))
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


# ----------------------------------------------------------------------
# 17. Bulk Tag Operations
# ----------------------------------------------------------------------
@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def bulk_tags(request, version='v1'):
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





@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_notes(request, run_id):
    """Get all notes linked to a specific training run."""
    notes = ResearcherNote.objects.filter(
        run__id=run_id, deleted_at__isnull=True
    ).order_by('-created_at')
    if not request.user.is_staff:
        notes = notes.filter(Q(is_public=True) | Q(author=request.user))
    paginator = StandardPagination()
    page = paginator.paginate_queryset(notes, request)
    serializer = ResearcherNoteSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)






@api_view(['POST'])
@permission_classes([IsAuthenticated])
def share_note(request, note_id):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Only the author can share"}, status=403)
    username = request.data.get('username')
    if not username:
        return Response({"error": "username required"}, status=400)
    try:
        user_to_share = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({"error": "User not found"}, status=404)
    note.shared_with.add(user_to_share)
    # After note.shared_with.add(user_to_share)
    from experiment.activity import log_activity
    log_activity(
        user=request.user,
        action_type='share',
        description=f'Shared note "{note.title or "Untitled"}" with {user_to_share.username}',
        content_object=note,
        metadata={'note_id': str(note.id), 'shared_with': user_to_share.username}
    )
    return Response({"status": "shared", "shared_with": [u.username for u in note.shared_with.all()]})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def unshare_note(request, note_id):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Only the author can unshare"}, status=403)
    username = request.data.get('username')
    if not username:
        return Response({"error": "username required"}, status=400)
    note.shared_with.remove(User.objects.get(username=username))
    return Response({"status": "unshared"})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def shared_with_me(request):
    notes = ResearcherNote.objects.filter(shared_with=request.user, deleted_at__isnull=True)
    serializer = ResearcherNoteSerializer(notes, many=True)
    return Response(serializer.data)

@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def toggle_pin(request, note_id):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    note.pinned = not note.pinned
    note.save(update_fields=['pinned'])
    return Response({"pinned": note.pinned})


# notes/views.py – add at the bottom

from .hf_upload import upload_note_to_hub

# ═══════════════════════════════════════════════════════════════════
# 26. HuggingFace Upload
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_note_to_huggingface(request, note_id, version='v1'):
    """Upload a note to HuggingFace Hub as a Markdown dataset/space."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if note.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    try:
        hub_url = upload_note_to_hub(note)
        note.metadata['hf_url'] = hub_url
        note.save(update_fields=['metadata'])
        return Response({"status": "uploaded", "url": hub_url})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
# 27. Workspace CRUD
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def workspace_list(request, version='v1'):
    if request.method == 'GET':
        workspaces = Workspace.objects.filter(
            models.Q(created_by=request.user) | models.Q(members=request.user)
        ).distinct()
        data = [{'id': str(w.id), 'name': w.name, 'description': w.description,
                 'member_count': w.members.count(), 'is_archived': w.is_archived}
                for w in workspaces]
        return Response(data)
    elif request.method == 'POST':
        workspace = Workspace.objects.create(
            name=request.data.get('name', 'Untitled'),
            description=request.data.get('description', ''),
            created_by=request.user
        )
        return Response({'id': str(workspace.id), 'status': 'created'}, status=201)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def workspace_add_note(request, workspace_id, note_id, version='v1'):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    note = get_object_or_404(ResearcherNote, id=note_id)
    workspace.notes.add(note)
    return Response({"status": "added"})

# (You can add more workspace endpoints similarly)


# ═══════════════════════════════════════════════════════════════════
# 28. Task CRUD
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def task_list(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if request.method == 'GET':
        tasks = note.tasks.all()
        data = [{'id': str(t.id), 'title': t.title, 'status': t.status,
                 'due_date': t.due_date, 'assigned_to': t.assigned_to.username if t.assigned_to else None}
                for t in tasks]
        return Response(data)
    elif request.method == 'POST':
        task = Task.objects.create(
            note=note,
            title=request.data.get('title', ''),
            description=request.data.get('description', ''),
            assigned_to_id=request.data.get('assigned_to'),
            status=request.data.get('status', 'pending'),
            due_date=request.data.get('due_date'),
        )
        return Response({'id': str(task.id)}, status=201)

@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def task_update(request, note_id, task_id, version='v1'):
    task = get_object_or_404(Task, id=task_id, note_id=note_id)
    if request.data.get('status'):
        task.status = request.data['status']
    task.save()
    return Response({'status': 'updated'})


# ═══════════════════════════════════════════════════════════════════
# 29. Model Registry
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def model_registry_list(request, version='v1'):
    if request.method == 'GET':
        models_qs = ModelRegistry.objects.filter(created_by=request.user)
        data = [{'id': str(m.id), 'name': m.name, 'version': m.version,
                 'stage': m.stage, 'created_at': m.created_at}
                for m in models_qs]
        return Response(data)
    elif request.method == 'POST':
        registry = ModelRegistry.objects.create(
            name=request.data['name'],
            version=request.data['version'],
            description=request.data.get('description', ''),
            stage=request.data.get('stage', 'development'),
            created_by=request.user
        )
        return Response({'id': str(registry.id)}, status=201)

# ═══════════════════════════════════════════════════════════════════
# 30. Note Links
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_note_link(request, version='v1'):
    source_id = request.data.get('source_id')
    target_id = request.data.get('target_id')
    link_type = request.data.get('link_type', 'related')
    source = get_object_or_404(ResearcherNote, id=source_id, author=request.user)
    target = get_object_or_404(ResearcherNote, id=target_id)
    link, created = NoteLink.objects.get_or_create(
        source=source, target=target, defaults={'link_type': link_type}
    )
    return Response({'status': 'created' if created else 'already_exists'})

# notes/views.py — append at the bottom of the file

from .hf_upload import upload_note_to_hub, upload_note_as_space, sync_note_to_hf
from .models import ShareableLink, Artifact, Protocol, NoteLink, NoteComment
import secrets


# ═══════════════════════════════════════════════════════════════════
# 26. HuggingFace Upload — as Dataset or Space (free hosting)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_note_to_huggingface(request, note_id, version='v1'):
    """Upload a note to HuggingFace as a Dataset (publicly viewable)."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if note.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    try:
        make_public = request.data.get('make_public', True)
        repo_type = request.data.get('repo_type', 'dataset')
        result = upload_note_to_hub(note, repo_type=repo_type, make_public=make_public)
        # Store URL in metadata
        note.metadata = note.metadata or {}
        note.metadata['hf_url'] = result['url']
        note.metadata['hf_repo_id'] = result['repo_id']
        note.save(update_fields=['metadata'])
        return Response(result)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_note_as_hf_space(request, note_id, version='v1'):
    """Upload a note to HuggingFace as a static Space (free web hosting)."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if note.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    try:
        result = upload_note_as_space(note)
        note.metadata = note.metadata or {}
        note.metadata['hf_space_url'] = result['url']
        note.save(update_fields=['metadata'])
        return Response(result)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def sync_note_to_hf_view(request, note_id, version='v1'):
    """Sync updated note content to HuggingFace."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    result = sync_note_to_hf(note)
    return Response(result)


# ═══════════════════════════════════════════════════════════════════
# 27. Shareable Links — generate, view, revoke
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_share_link(request, note_id, version='v1'):
    """Create a public share link for a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)

    expires_in_days = request.data.get('expires_in_days')
    expires_at = timezone.now() + timedelta(days=expires_in_days) if expires_in_days else None

    link = ShareableLink.objects.create(
        note=note,
        created_by=request.user,
        password=request.data.get('password', ''),
        expires_at=expires_at,
        max_views=request.data.get('max_views'),
    )
    return Response({
        "id": str(link.id),
        "token": link.token,
        "url": link.public_url,
        "expires_at": link.expires_at,
        "is_active": link.is_active,
    }, status=201)


@api_view(['GET'])
def view_shared_note(request, token, version='v1'):
    """
    Public endpoint — view a note via share link.
    No authentication required, but password may be needed.
    """
    link = get_object_or_404(ShareableLink, token=token, is_active=True)

    # Check expiration
    if link.is_expired:
        return Response({"error": "This share link has expired"}, status=410)

    # Check password
    if link.password:
        provided_password = request.query_params.get('password', '')
        if provided_password != link.password:
            return Response({"error": "Password required", "password_required": True}, status=401)

    note = link.note
    if note.deleted_at:
        return Response({"error": "Note has been deleted"}, status=404)

    # Record view
    link.record_view()

    serializer = ResearcherNoteSerializer(note)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_share_links(request, note_id, version='v1'):
    """List all share links for a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    links = note.share_links.all()
    data = [{
        "id": str(l.id),
        "token": l.token,
        "url": l.public_url,
        "is_active": l.is_active,
        "view_count": l.view_count,
        "expires_at": l.expires_at,
        "created_at": l.created_at,
    } for l in links]
    return Response(data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def revoke_share_link(request, note_id, link_id, version='v1'):
    """Revoke a share link."""
    link = get_object_or_404(ShareableLink, id=link_id, note_id=note_id)
    if link.created_by != request.user:
        return Response({"error": "Not allowed"}, status=403)
    link.is_active = False
    link.save(update_fields=['is_active'])
    return Response({"status": "revoked"})


# ═══════════════════════════════════════════════════════════════════
# 28. Share with Users (direct)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def share_note(request, note_id, version='v1'):
    """Share a note with specific users by username."""
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Only the author can share"}, status=403)
    username = request.data.get('username')
    if not username:
        return Response({"error": "username required"}, status=400)
    try:
        user_to_share = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({"error": "User not found"}, status=404)
    note.shared_with.add(user_to_share)
    Notification.objects.create(
        recipient=user_to_share, sender=request.user, type='share',
        note=note, content=f"{request.user.username} shared a note with you: {note.title or 'Untitled'}"
    )
    return Response({"status": "shared", "shared_with": [u.username for u in note.shared_with.all()]})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def unshare_note(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Only the author can unshare"}, status=403)
    username = request.data.get('username')
    if not username:
        return Response({"error": "username required"}, status=400)
    note.shared_with.remove(User.objects.get(username=username))
    return Response({"status": "unshared"})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def shared_with_me(request, version='v1'):
    notes = ResearcherNote.objects.filter(shared_with=request.user, deleted_at__isnull=True)
    serializer = ResearcherNoteSerializer(notes, many=True)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════════
# 29. Artifact CRUD
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def artifact_list(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if request.method == 'GET':
        artifacts = note.artifacts.all()
        data = [{
            'id': str(a.id), 'name': a.name, 'file_type': a.file_type,
            'file_size_mb': a.file_size_mb, 'description': a.description,
            'file_url': request.build_absolute_uri(a.file.url) if a.file else None,
            'created_at': a.created_at,
        } for a in artifacts]
        return Response(data)
    elif request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({"error": "file required"}, status=400)
        size_mb = round(uploaded_file.size / (1024 * 1024), 2)
        artifact = Artifact.objects.create(
            note=note,
            name=request.data.get('name', uploaded_file.name),
            file=uploaded_file,
            file_type=request.data.get('file_type', ''),
            file_size_mb=size_mb,
            description=request.data.get('description', ''),
        )
        return Response({'id': str(artifact.id), 'file_url': request.build_absolute_uri(artifact.file.url)}, status=201)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def artifact_delete(request, note_id, artifact_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id)
    artifact = get_object_or_404(Artifact, id=artifact_id, note=note)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    artifact.file.delete(save=False)
    artifact.delete()
    return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def artifact_download(request, note_id, artifact_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    artifact = get_object_or_404(Artifact, id=artifact_id, note=note)
    if not note.is_public and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    response = FileResponse(artifact.file.open('rb'), as_attachment=True)
    response['Content-Disposition'] = f'attachment; filename="{artifact.name}"'
    return response


# ═══════════════════════════════════════════════════════════════════
# 30. Protocol CRUD
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def protocol_list(request, version='v1'):
    if request.method == 'GET':
        protocols = Protocol.objects.filter(
            models.Q(created_by=request.user) | models.Q(is_public=True)
        ).order_by('-created_at')
        data = [{
            'id': str(p.id), 'name': p.name, 'description': p.description,
            'steps_count': len(p.steps), 'is_public': p.is_public,
            'created_by': p.created_by.username, 'created_at': p.created_at,
        } for p in protocols]
        return Response(data)
    elif request.method == 'POST':
        protocol = Protocol.objects.create(
            name=request.data.get('name', 'Untitled Protocol'),
            description=request.data.get('description', ''),
            steps=request.data.get('steps', []),
            created_by=request.user,
            is_public=request.data.get('is_public', False),
        )
        return Response({'id': str(protocol.id), 'status': 'created'}, status=201)


@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def protocol_detail(request, protocol_id, version='v1'):
    protocol = get_object_or_404(Protocol, id=protocol_id)
    if request.method == 'GET':
        return Response({
            'id': str(protocol.id), 'name': protocol.name,
            'description': protocol.description, 'steps': protocol.steps,
            'is_public': protocol.is_public,
            'created_by': protocol.created_by.username,
            'created_at': protocol.created_at, 'updated_at': protocol.updated_at,
        })
    if protocol.created_by != request.user:
        return Response({"error": "Not allowed"}, status=403)
    if request.method in ('PUT', 'PATCH'):
        protocol.name = request.data.get('name', protocol.name)
        protocol.description = request.data.get('description', protocol.description)
        protocol.steps = request.data.get('steps', protocol.steps)
        protocol.is_public = request.data.get('is_public', protocol.is_public)
        protocol.save()
        return Response({'status': 'updated'})
    elif request.method == 'DELETE':
        protocol.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════════
# 31. Note Links (Knowledge Graph)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_note_link(request, version='v1'):
    source_id = request.data.get('source_id')
    target_id = request.data.get('target_id')
    link_type = request.data.get('link_type', 'related')
    source = get_object_or_404(ResearcherNote, id=source_id, author=request.user, deleted_at__isnull=True)
    target = get_object_or_404(ResearcherNote, id=target_id, deleted_at__isnull=True)
    link, created = NoteLink.objects.get_or_create(
        source=source, target=target, defaults={'link_type': link_type}
    )
    return Response({'status': 'created' if created else 'already_exists', 'id': str(link.id)})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_links_graph(request, note_id, version='v1'):
    """Return the knowledge graph centered on a note."""
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    outgoing = note.outgoing_links.select_related('target')
    incoming = note.incoming_links.select_related('source')

    nodes = {str(note.id): {'id': str(note.id), 'title': note.title or 'Untitled', 'type': 'current'}}
    edges = []

    for link in outgoing:
        target = link.target
        if target.deleted_at is None:
            nodes[str(target.id)] = {'id': str(target.id), 'title': target.title or 'Untitled', 'type': 'outgoing'}
            edges.append({'from': str(note.id), 'to': str(target.id), 'type': link.link_type})

    for link in incoming:
        source = link.source
        if source.deleted_at is None:
            nodes[str(source.id)] = {'id': str(source.id), 'title': source.title or 'Untitled', 'type': 'incoming'}
            edges.append({'from': str(source.id), 'to': str(note.id), 'type': link.link_type})

    return Response({'nodes': list(nodes.values()), 'edges': edges})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_note_link(request, link_id, version='v1'):
    link = get_object_or_404(NoteLink, id=link_id)
    if link.source.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    link.delete()
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════════
# 32. Toggle Pin
# ═══════════════════════════════════════════════════════════════════
@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def toggle_pin(request, note_id, version='v1'):
    note = get_object_or_404(ResearcherNote, id=note_id)
    if note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)
    note.pinned = not note.pinned
    note.save(update_fields=['pinned'])
    return Response({"pinned": note.pinned})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_note_from_paper(request, version='v1'):
    """
    Save a paper and create a note with its details.
    Body: { paper_id, source, note_title (optional) }
    """
    paper_id = request.data.get('paper_id')
    source = request.data.get('source', 'semantic_scholar')
    if not paper_id:
        return Response({"error": "paper_id required"}, status=400)

    # Fetch paper details
    from agents.tools import get_paper_details, save_paper
    result = get_paper_details(paper_id=paper_id, source=source)
    if result.get('status') != 'success':
        return Response(result, status=404)

    paper = result['paper']
    # Save to user's library
    save_result = save_paper(
        title=paper.get('title',''), source=source, source_id=paper_id,
        authors=paper.get('authors',[]), abstract=paper.get('abstract',''),
        url=paper.get('url',''), year=paper.get('year'),
        citation_count=paper.get('citation_count',0),
        pdf_url=paper.get('pdf_url',''),
        user_id=request.user.id
    )

    # Create a rich note
    note = ResearcherNote.objects.create(
        author=request.user,
        note_type='literature_review',
        title=request.data.get('note_title') or paper.get('title', 'Untitled'),
        content=f"""## 📄 {paper.get('title','Untitled')}

**Authors:** {', '.join(paper.get('authors',[]))}
**Year:** {paper.get('year','N/A')}
**Citations:** {paper.get('citation_count',0)}
**Source:** [{paper.get('source','')}]({paper.get('url','')})

### Abstract
{paper.get('abstract','')}

### Notes
*(Add your notes here)*
""",
        tags=['paper-review', source],
        metadata={'paper': paper},
    )
    serializer = ResearcherNoteSerializer(note)
    return Response(serializer.data, status=201)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def note_bibliography(request, note_id, version='v1'):
    """
    Return a formatted bibliography (BibTeX or APA) for all papers linked to a note.
    """
    note = get_object_or_404(ResearcherNote, id=note_id, deleted_at__isnull=True)
    if not note.is_public and note.author != request.user:
        return Response({"error": "Not allowed"}, status=403)

    papers = note.metadata.get('papers', [])
    if not papers:
        return Response({"references": [], "bibtex": ""})

    format_type = request.query_params.get('format', 'apa')

    references = []
    bibtex_entries = []
    for paper in papers:
        title = paper.get('title', '')
        authors = paper.get('authors', [])
        year = paper.get('year', '')
        source = paper.get('source', '')
        source_id = paper.get('source_id', '')
        first_author = authors[0].split()[-1] if authors else 'Unknown'
        ref_key = f"{first_author.lower()}{year}"

        # APA style
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{authors[0]} et al."
        else:
            author_str = 'Unknown'

        apa_ref = f"{author_str} ({year}). {title}."
        references.append(apa_ref)

        # BibTeX
        bibtex = f"""@article{{{ref_key},
  title = {{{title}}},
  author = {{{' and '.join(authors)}}},
  year = {{{year}}},
  url = {{https://api.semanticscholar.org/CorpusID:{source_id}}}
}}"""
        bibtex_entries.append(bibtex)

    bibtex_all = '\n\n'.join(bibtex_entries)

    return Response({
        "references": references,
        "bibtex": bibtex_all,
        "count": len(references),
    })



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def search_and_format_paper(request, version='v1'):
    """Search for a paper and return a formatted Markdown snippet for insertion into a note."""
    query = request.query_params.get('q', '')
    source = request.query_params.get('source', 'semantic_scholar')
    if not query:
        return Response({"results": []})

    from agents.tools import search_papers
    result = search_papers(query=query, limit=5, source=source)
    papers = result.get('results', [])

    formatted = []
    for paper in papers:
        md = f"""### [{paper.get('title','Untitled')}]({paper.get('url','')})
- **Authors:** {', '.join(paper.get('authors',[]))}
- **Year:** {paper.get('year','N/A')} | **Citations:** {paper.get('citation_count',0)}
- **Abstract:** {paper.get('abstract','')[:300]}...

```bibtex
@article{{{paper.get('authors',[['Unknown']])[0].split()[-1].lower()}{paper.get('year','')},
  title = {{{paper.get('title','')}}},
  author = {{{' and '.join(paper.get('authors',[]))}}},
  year = {{{paper.get('year','')}}}
}}
```"""
        formatted.append({
            "source_id": paper.get('source_id'),
            "source": paper.get('source'),
            "markdown": md,
        })
    return Response({"results": formatted})