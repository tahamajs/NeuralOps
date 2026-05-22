# reviews/views.py
"""Peer Review REST API — full CRUD for requests, reviews, comments, templates, badges."""
import logging
from django.shortcuts import get_object_or_404
from django.db.models import Q, Count, Avg
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)

class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 50


# ═══════════════════════════════════════════════════════════════
# REVIEW REQUESTS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def review_request_list(request):
    if request.method == 'GET':
        qs = ReviewRequest.objects.annotate(completed_count=Count('reviews'))
        # Filters
        status_filter = request.query_params.get('status')
        if status_filter: qs = qs.filter(status=status_filter)
        type_filter = request.query_params.get('submission_type')
        if type_filter: qs = qs.filter(submission_type=type_filter)
        tag_filter = request.query_params.get('tag')
        if tag_filter: qs = qs.filter(tags__contains=[tag_filter])
        search = request.query_params.get('q')
        if search: qs = qs.filter(Q(title__icontains=search) | Q(abstract__icontains=search))

        # Visibility: public + user's own
        if not request.user.is_staff:
            qs = qs.filter(Q(visibility='public') | Q(author=request.user))

        qs = qs.order_by('-created_at')
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(ReviewRequestListSerializer(page, many=True).data)

    elif request.method == 'POST':
        serializer = ReviewRequestDetailSerializer(data=request.data)
        if serializer.is_valid():
            rr = serializer.save(author=request.user, status='submitted')
            # Auto-assign reviewers if request is public
            _auto_assign_reviewers(rr)
            return Response(ReviewRequestDetailSerializer(rr).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def review_request_detail(request, request_id):
    rr = get_object_or_404(ReviewRequest, id=request_id)
    if request.method == 'GET':
        rr.view_count += 1
        rr.save(update_fields=['view_count'])
        return Response(ReviewRequestDetailSerializer(rr).data)
    elif request.method == 'PATCH':
        if rr.author != request.user:
            return Response({"error": "Only the author can edit"}, status=403)
        serializer = ReviewRequestDetailSerializer(rr, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == 'DELETE':
        if rr.author != request.user and not request.user.is_staff:
            return Response({"error": "Only the author can delete"}, status=403)
        rr.delete()
        return Response(status=204)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_for_review(request, request_id):
    """Submit a draft for review."""
    rr = get_object_or_404(ReviewRequest, id=request_id, author=request.user)
    if rr.status != 'draft':
        return Response({"error": "Only drafts can be submitted"}, status=400)
    rr.status = 'submitted'
    rr.save(update_fields=['status'])
    _auto_assign_reviewers(rr)
    return Response(ReviewRequestDetailSerializer(rr).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def open_for_review(request, request_id):
    """Open the submission for review (author action)."""
    rr = get_object_or_404(ReviewRequest, id=request_id, author=request.user)
    if rr.status not in ('submitted', 'in_review'):
        return Response({"error": "Cannot open in current status"}, status=400)
    rr.status = 'in_review'
    rr.save(update_fields=['status'])
    # Notify assigned reviewers
    for reviewer in rr.reviewers.all():
        from experiment.models import Notification
        Notification.objects.create(
            recipient=reviewer, sender=request.user, type='system',
            content=f"You've been invited to review: {rr.title}",
        )
    return Response(ReviewRequestDetailSerializer(rr).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def complete_review_round(request, request_id):
    """Mark review round as completed (author or staff)."""
    rr = get_object_or_404(ReviewRequest, id=request_id)
    if rr.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    rr.status = 'completed'
    rr.save(update_fields=['status'])
    return Response(ReviewRequestDetailSerializer(rr).data)


# ═══════════════════════════════════════════════════════════════
# REVIEWS
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def write_review(request, request_id):
    """Submit a review for a review request."""
    rr = get_object_or_404(ReviewRequest, id=request_id)
    if not rr.is_open_for_review:
        return Response({"error": "This submission is not currently open for review"}, status=400)
    if rr.author == request.user:
        return Response({"error": "You cannot review your own submission"}, status=400)

    # Get or create review
    review, created = Review.objects.get_or_create(
        review_request=rr, reviewer=request.user,
        defaults={'status': 'draft'}
    )

    serializer = ReviewSerializer(review, data=request.data, partial=True)
    if serializer.is_valid():
        review = serializer.save(status='completed')
        rr.comment_count = rr.reviews.count()
        rr.save(update_fields=['comment_count'])
        # Notify author
        from experiment.models import Notification
        Notification.objects.create(
            recipient=rr.author, sender=request.user, type='system',
            content=f"A new review has been submitted for your paper: {rr.title}",
        )
        return Response(ReviewSerializer(review).data)
    return Response(serializer.errors, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_reviews_given(request):
    """List all reviews written by the current user."""
    qs = Review.objects.filter(reviewer=request.user).select_related('review_request').order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    data = [{
        **ReviewSerializer(r).data,
        'review_request_title': r.review_request.title,
        'review_request_id': str(r.review_request.id),
    } for r in page]
    return paginator.get_paginated_response(data)


# ═══════════════════════════════════════════════════════════════
# REVIEW COMMENTS (Dialogue)
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_review_comment(request, review_id):
    """Add a comment to a review (author ↔ reviewer dialogue)."""
    review = get_object_or_404(Review, id=review_id)
    # Only the review author or the submission author can comment
    if request.user not in (review.reviewer, review.review_request.author) and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    serializer = ReviewCommentSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(review=review, author=request.user)
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def resolve_comment(request, comment_id):
    """Mark a comment as resolved."""
    comment = get_object_or_404(ReviewComment, id=comment_id)
    if comment.author != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    comment.is_resolved = True
    comment.save(update_fields=['is_resolved'])
    return Response(ReviewCommentSerializer(comment).data)


# ═══════════════════════════════════════════════════════════════
# HELPFULNESS VOTING
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def vote_review_helpful(request, review_id):
    review = get_object_or_404(Review, id=review_id)
    direction = request.data.get('direction', 'up')
    if direction not in ('up', 'down'):
        return Response({"error": "direction must be 'up' or 'down'"}, status=400)
    ReviewVote.objects.update_or_create(
        review=review, voter=request.user,
        defaults={'direction': direction}
    )
    return Response({'status': 'voted'})


# ═══════════════════════════════════════════════════════════════
# SUMMARY / STATS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def review_summary(request, request_id):
    """Get aggregated summary of all reviews for a submission."""
    rr = get_object_or_404(ReviewRequest, id=request_id)
    if rr.author != request.user and rr.visibility != 'public' and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)

    completed = rr.completed_reviews
    scores = {
        'overall': round(completed.aggregate(v=Avg('overall_score'))['v'] or 0, 2),
        'originality': round(completed.aggregate(v=Avg('originality_score'))['v'] or 0, 2),
        'soundness': round(completed.aggregate(v=Avg('soundness_score'))['v'] or 0, 2),
        'presentation': round(completed.aggregate(v=Avg('presentation_score'))['v'] or 0, 2),
        'significance': round(completed.aggregate(v=Avg('significance_score'))['v'] or 0, 2),
    }

    # Average from text reviews via keyword extraction (simple)
    keywords = []
    for r in completed:
        if r.strengths:
            keywords.extend([w for w in r.strengths.lower().split() if len(w) > 3])
    from collections import Counter
    keyword_counts = Counter(keywords).most_common(10)

    return Response({
        'request_id': str(rr.id),
        'title': rr.title,
        'total_reviews': completed.count(),
        'scores': scores,
        'common_strengths': [w for w, _ in keyword_counts],
        'reviewer_comments': [{
            'reviewer': r.reviewer.username if not r.is_anonymous else 'Anonymous',
            'summary': r.summary[:300],
            'score': r.overall_score,
        } for r in completed],
    })


# ═══════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def template_list(request):
    if request.method == 'GET':
        qs = ReviewTemplate.objects.all()
        type_filter = request.query_params.get('submission_type')
        if type_filter: qs = qs.filter(submission_type=type_filter)
        return Response(ReviewTemplateSerializer(qs, many=True).data)
    elif request.method == 'POST':
        serializer = ReviewTemplateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# BADGES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def badge_list(request):
    return Response(ReviewBadgeSerializer(ReviewBadge.objects.all(), many=True).data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_badges(request):
    badges = ReviewerBadge.objects.filter(reviewer=request.user).select_related('badge')
    return Response(ReviewerBadgeSerializer(badges, many=True).data)


# ═══════════════════════════════════════════════════════════════
# GLOBAL STATS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def global_stats(request):
    return Response({
        'open_for_review': ReviewRequest.objects.filter(status='in_review').count(),
        'total_reviews': Review.objects.filter(status='completed').count(),
        'active_reviewers': Review.objects.filter(status='completed').values('reviewer').distinct().count(),
    })


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def _auto_assign_reviewers(rr):
    """Auto-assign interested reviewers based on tags/similarity."""
    if rr.reviewers.exists():
        return  # already assigned
    # Find users who have reviewed similar tags recently
    similar_reviewers = User.objects.filter(
        reviews_given__review_request__tags__contains=rr.tags
    ).distinct()[:rr.reviewers_required]
    rr.reviewers.add(*similar_reviewers)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def ai_review_assist(request, review_request_id):
    """Generate AI-assisted review for a paper."""
    review_request = get_object_or_404(ReviewRequest, id=review_request_id)
    paper_content = request.data.get('paper_content', review_request.abstract or '')

    assistant = AIReviewAssistant()
    review = assistant.generate_review(paper_content)
    soundness = assistant.check_argument_soundness(paper_content)

    return Response({
        'review': review,
        'soundness_issues': soundness,
    })
