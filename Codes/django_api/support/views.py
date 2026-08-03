import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Solution, SupportTicket
from .serializers import (
    FeedbackSerializer,
    SolutionSerializer,
    SupportTicketCreateSerializer,
    SupportTicketSerializer,
    TicketMessageSerializer,
)
from .tasks import process_ticket_task

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Support Tickets
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def ticket_list(request):
    """List the authenticated user's support tickets, or create a new one."""
    if request.method == "GET":
        qs = SupportTicket.objects.filter(user=request.user).order_by("-created_at")
        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            SupportTicketSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = SupportTicketCreateSerializer(data=request.data)
        if serializer.is_valid():
            ticket = serializer.save(user=request.user)
            # Trigger AI processing in background
            process_ticket_task.delay(str(ticket.id))
            return Response(SupportTicketSerializer(ticket).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def ticket_detail(request, ticket_id):
    """Get, update, or close a support ticket."""
    ticket = get_object_or_404(SupportTicket, id=ticket_id, user=request.user)

    if request.method == "GET":
        return Response(SupportTicketSerializer(ticket).data)

    elif request.method == "PATCH":
        if "status" in request.data:
            ticket.status = request.data["status"]
            ticket.save()
        return Response(SupportTicketSerializer(ticket).data)

    elif request.method == "DELETE":
        ticket.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Messages (for chat history)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def ticket_messages(request, ticket_id):
    """Get all messages for a ticket."""
    ticket = get_object_or_404(SupportTicket, id=ticket_id, user=request.user)
    msgs = ticket.messages.order_by("created_at")
    return Response(TicketMessageSerializer(msgs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# Solutions (knowledge base)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def solution_list(request):
    """Search across past solutions."""
    q = request.query_params.get("q", "").strip()
    qs = Solution.objects.all()
    if q:
        qs = qs.filter(content__icontains=q)
    qs = qs.order_by("-upvotes")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(SolutionSerializer(page, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upvote_solution(request, solution_id):
    """Upvote a solution."""
    solution = get_object_or_404(Solution, id=solution_id)
    solution.upvotes += 1
    solution.save(update_fields=["upvotes"])
    return Response({"upvotes": solution.upvotes})


# ═══════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_feedback(request, ticket_id):
    """Submit feedback on a ticket's AI response."""
    ticket = get_object_or_404(SupportTicket, id=ticket_id, user=request.user)
    serializer = FeedbackSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(ticket=ticket, user=request.user)
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)
