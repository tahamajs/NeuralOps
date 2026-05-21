# community/views.py
"""Community Q&A REST API — full Stack Overflow + ResearchGate functionality."""
import logging
from django.shortcuts import get_object_or_404
from django.db.models import Q, Sum, Count, F, Window
from django.db.models.functions import Rank
from django.utils import timezone
from datetime import timedelta
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import *
from .serializers import *
from .signals import update_activity_log

logger = logging.getLogger(__name__)

class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 50

REP = {
    'question_upvoted': 5, 'question_downvoted': -2,
    'answer_upvoted': 10, 'answer_downvoted': -2,
    'answer_accepted': 15, 'answer_accept_given': 2,
    'downvoted_answer': -1, 'association_bonus': 100,
}

# ═══════════════════════════════════════════════════════════════
# QUESTIONS — list, create, detail, edit, delete
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def question_list(request):
    if request.method == 'GET':
        qs = Question.objects.filter(status='open').select_related('author').prefetch_related('tags')
        # Filters
        tag = request.query_params.get('tag')
        if tag: qs = qs.filter(tags__slug=tag)
        search = request.query_params.get('q')
        if search: qs = qs.filter(Q(title__icontains=search) | Q(body__icontains=search))
        sort = request.query_params.get('sort', 'newest')
        sort_map = {'newest': '-created_at', 'active': '-last_activity_at',
                    'votes': '-score', 'unanswered': 'answer_count'}
        qs = qs.order_by(sort_map.get(sort, '-created_at'))
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(QuestionListSerializer(page, many=True).data)

    elif request.method == 'POST':
        serializer = QuestionDetailSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.save(author=request.user)
            # Tag association
            tag_names = request.data.get('tags', [])
            for tag_name in tag_names:
                tag, _ = Tag.objects.get_or_create(name=tag_name.lower().strip(),
                    defaults={'slug': tag_name.lower().strip().replace(' ', '-')})
                question.tags.add(tag)
            update_activity_log(request.user, 'questions_asked')
            return Response(QuestionDetailSerializer(question).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def question_detail(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    if request.method == 'GET':
        question.view_count += 1
        question.save(update_fields=['view_count'])
        return Response(QuestionDetailSerializer(
            question, context={'request': request}).data)
    elif request.method == 'PATCH':
        if question.author != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        serializer = QuestionDetailSerializer(question, data=request.data, partial=True)
        if serializer.is_valid(): serializer.save(); return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == 'DELETE':
        if question.author != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        question.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# ANSWERS
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def post_answer(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    if question.status != 'open':
        return Response({"error": "This question is not open for answers"}, status=400)
    serializer = AnswerSerializer(data=request.data)
    if serializer.is_valid():
        answer = serializer.save(question=question, author=request.user)
        question.answer_count = question.answers.count()
        question.last_activity_at = timezone.now()
        question.save(update_fields=['answer_count', 'last_activity_at'])
        update_activity_log(request.user, 'answers_posted')
        return Response(AnswerSerializer(answer).data, status=201)
    return Response(serializer.errors, status=400)


@api_view(['PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def answer_detail(request, answer_id):
    answer = get_object_or_404(Answer, id=answer_id)
    if request.method == 'PATCH':
        if answer.author != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        serializer = AnswerSerializer(answer, data=request.data, partial=True)
        if serializer.is_valid(): serializer.save(); return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == 'DELETE':
        if answer.author != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        q = answer.question
        answer.delete()
        q.answer_count = q.answers.count()
        q.save(update_fields=['answer_count'])
        return Response(status=204)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def accept_answer(request, answer_id):
    answer = get_object_or_404(Answer, id=answer_id)
    if answer.question.author != request.user and not request.user.is_staff:
        return Response({"error": "Only the question author can accept"}, status=403)
    answer.is_accepted = True
    answer.save(update_fields=['is_accepted'])
    answer.question.accepted_answer = answer
    answer.question.save(update_fields=['accepted_answer'])
    # Reputation events
    ReputationEvent.objects.create(user=answer.author, event_type='answer_accepted', amount=REP['answer_accepted'])
    ReputationEvent.objects.create(user=request.user, event_type='answer_accept_given', amount=REP['answer_accept_given'])
    return Response(AnswerSerializer(answer).data)


# ═══════════════════════════════════════════════════════════════
# VOTING
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def vote_question(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    direction = request.data.get('direction')
    if direction not in ('up', 'down'):
        return Response({"error": "direction must be 'up' or 'down'"}, status=400)
    if question.author == request.user:
        return Response({"error": "Cannot vote your own post"}, status=400)
    vote, created = Vote.objects.update_or_create(
        voter=request.user, question=question, defaults={'direction': direction})
    question.update_score()
    rep_amount = REP['question_upvoted'] if direction == 'up' else REP['question_downvoted']
    ReputationEvent.objects.get_or_create(user=question.author, question=question,
        event_type=f'question_{direction}voted', defaults={'amount': rep_amount})
    update_activity_log(request.user, 'votes_cast')
    return Response({'score': question.score, 'status': 'voted'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def vote_answer(request, answer_id):
    answer = get_object_or_404(Answer, id=answer_id)
    direction = request.data.get('direction')
    if direction not in ('up', 'down'):
        return Response({"error": "direction must be 'up' or 'down'"}, status=400)
    if answer.author == request.user:
        return Response({"error": "Cannot vote your own post"}, status=400)
    vote, created = Vote.objects.update_or_create(
        voter=request.user, answer=answer, defaults={'direction': direction})
    answer.update_score()
    rep_amount = REP['answer_upvoted'] if direction == 'up' else REP['answer_downvoted']
    ReputationEvent.objects.get_or_create(user=answer.author, answer=answer,
        event_type=f'answer_{direction}voted', defaults={'amount': rep_amount})
    if direction == 'down':
        ReputationEvent.objects.get_or_create(user=request.user, answer=answer,
            event_type='downvoted_answer', defaults={'amount': REP['downvoted_answer']})
    update_activity_log(request.user, 'votes_cast')
    return Response({'score': answer.score, 'status': 'voted'})


# ═══════════════════════════════════════════════════════════════
# COMMENTS
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def post_comment(request, question_id=None, answer_id=None):
    data = request.data.copy()
    if question_id: data['question'] = question_id
    if answer_id: data['answer'] = answer_id
    serializer = CommentSerializer(data=data)
    if serializer.is_valid():
        serializer.save(author=request.user)
        update_activity_log(request.user, 'comments_posted')
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# BOUNTY
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_bounty(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    total_rep = request.user.reputation_events.aggregate(s=Sum('amount'))['s'] or 0
    if total_rep < 75:
        return Response({"error": "Minimum 75 reputation required"}, status=400)
    amount = int(request.data.get('amount', 50))
    if amount < 50 or amount > 500:
        return Response({"error": "Bounty must be between 50 and 500"}, status=400)
    if total_rep < amount:
        return Response({"error": "Insufficient reputation"}, status=400)
    Bounty.objects.create(
        question=question, offerer=request.user, amount=amount,
        reason=request.data.get('reason', 'draw_attention'),
        notice=request.data.get('notice', ''),
        expires_at=timezone.now() + timedelta(days=7))
    ReputationEvent.objects.create(user=request.user, event_type='bounty_offered', amount=-amount)
    question.has_bounty = True
    question.save(update_fields=['has_bounty'])
    return Response({'status': 'bounty started', 'amount': amount})


# ═══════════════════════════════════════════════════════════════
# TAGS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def tag_list(request):
    qs = Tag.objects.annotate(qc=Count('questions')).order_by('-qc')
    return Response(TagSerializer(qs, many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def tag_detail(request, slug):
    tag = get_object_or_404(Tag, slug=slug)
    return Response(TagSerializer(tag).data)


# ═══════════════════════════════════════════════════════════════
# USER REPUTATION & PROFILE
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def user_reputation(request, username):
    from django.contrib.auth.models import User
    user = get_object_or_404(User, username=username)
    total = user.reputation_events.aggregate(s=Sum('amount'))['s'] or 0
    events = user.reputation_events.order_by('-created_at')[:50]
    badges = UserBadge.objects.filter(user=user).select_related('badge')
    return Response({
        'username': user.username, 'reputation': total,
        'events': ReputationEventSerializer(events, many=True).data,
        'badges': UserBadgeSerializer(badges, many=True).data,
    })


# ═══════════════════════════════════════════════════════════════
# LEADERBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def leaderboard(request):
    period = request.query_params.get('period', 'all')
    qs = ReputationEvent.objects.values('user__username').annotate(
        total=Sum('amount')
    ).order_by('-total')[:50]
    return Response([{'username': r['user__username'], 'reputation': r['total']} for r in qs])


# ═══════════════════════════════════════════════════════════════
# BADGES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def badge_list(request):
    return Response(BadgeSerializer(Badge.objects.all(), many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_badges(request):
    badges = UserBadge.objects.filter(user=request.user).select_related('badge')
    return Response(UserBadgeSerializer(badges, many=True).data)


# ═══════════════════════════════════════════════════════════════
# BOOKMARK
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def bookmark_question(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    Bookmark.objects.get_or_create(user=request.user, question=question)
    return Response({'status': 'bookmarked'})

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def unbookmark_question(request, question_id):
    Bookmark.objects.filter(user=request.user, question_id=question_id).delete()
    return Response({'status': 'unbookmarked'})


# ═══════════════════════════════════════════════════════════════
# ACTIVITY GRAPH
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def activity_graph(request, username):
    from django.contrib.auth.models import User
    user = get_object_or_404(User, username=username)
    year_ago = timezone.now().date() - timedelta(days=365)
    logs = ActivityLog.objects.filter(user=user, date__gte=year_ago).values('date').annotate(
        total=F('questions_asked') + F('answers_posted') + F('votes_cast') + F('comments_posted')
    ).order_by('date')
    return Response([{'date': str(l['date']), 'count': l['total']} for l in logs])


# ═══════════════════════════════════════════════════════════════
# DIGEST SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def digest_settings(request):
    if request.method == 'GET':
        sub, _ = DigestSubscription.objects.get_or_create(user=request.user)
        return Response({'frequency': sub.frequency, 'include_trending': sub.include_trending,
                         'include_my_tags': sub.include_my_tags})
    elif request.method == 'POST':
        sub, _ = DigestSubscription.objects.get_or_create(user=request.user)
        sub.frequency = request.data.get('frequency', sub.frequency)
        sub.include_trending = request.data.get('include_trending', sub.include_trending)
        sub.include_my_tags = request.data.get('include_my_tags', sub.include_my_tags)
        sub.save()
        return Response({'status': 'updated'})


# ═══════════════════════════════════════════════════════════════
# PRIVILEGES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def privilege_list(request):
    total_rep = request.user.reputation_events.aggregate(s=Sum('amount'))['s'] or 0
    privileges = Privilege.objects.all().order_by('reputation_required')
    data = [{'name': p.name, 'required': p.reputation_required,
             'unlocked': total_rep >= p.reputation_required} for p in privileges]
    return Response(data)


# ═══════════════════════════════════════════════════════════════
# CLOSE QUESTION
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def close_question(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    total_rep = request.user.reputation_events.aggregate(s=Sum('amount'))['s'] or 0
    if total_rep < 250 and question.author != request.user and not request.user.is_staff:
        return Response({"error": "250 reputation required to close questions"}, status=403)
    question.status = 'closed'
    question.close_reason = request.data.get('reason', '')
    question.save(update_fields=['status', 'close_reason'])
    return Response({'status': 'closed'})


# ═══════════════════════════════════════════════════════════════
# RELATED QUESTIONS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def related_questions(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    tag_ids = question.tags.values_list('id', flat=True)
    related = Question.objects.filter(tags__id__in=tag_ids, status='open').exclude(
        id=question_id).distinct().order_by('-score')[:10]
    return Response(QuestionListSerializer(related, many=True).data)


# ═══════════════════════════════════════════════════════════════
# TRUST LEVEL
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_trust_level(request):
    tl, _ = TrustLevel.objects.get_or_create(user=request.user)
    return Response(TrustLevelSerializer(tl).data)