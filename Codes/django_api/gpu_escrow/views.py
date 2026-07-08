# gpu_escrow/views.py
import logging

from django.db import transaction
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import EscrowAccount
from .serializers import (DepositWithdrawSerializer, EscrowAccountSerializer,
                          EscrowTransactionSerializer)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


def _get_account(user) -> EscrowAccount:
    """Get or create the escrow account for a user."""
    account, _ = EscrowAccount.objects.get_or_create(user=user)
    return account


# ═══════════════════════════════════════════════════════════════
# BALANCE
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def balance(request):
    """Get the authenticated user's escrow balance."""
    account = _get_account(request.user)
    return Response(EscrowAccountSerializer(account).data)


# ═══════════════════════════════════════════════════════════════
# DEPOSIT (Mock – in production, integrate with Stripe / PayPal)
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def deposit(request):
    """Add funds to the user's escrow balance."""
    serializer = DepositWithdrawSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    amount = serializer.validated_data["amount_usd"]
    description = serializer.validated_data.get("description", "Manual deposit")

    account = _get_account(request.user)
    try:
        with transaction.atomic():
            account.deposit(amount, description)
    except Exception as e:
        logger.error("Deposit failed: %s", e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(EscrowAccountSerializer(account).data)


# ═══════════════════════════════════════════════════════════════
# WITHDRAW
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def withdraw(request):
    """Withdraw available funds from the escrow account."""
    serializer = DepositWithdrawSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    amount = serializer.validated_data["amount_usd"]
    description = serializer.validated_data.get("description", "Manual withdrawal")

    account = _get_account(request.user)
    if amount > account.balance_usd:
        return Response(
            {"error": "Insufficient available balance"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        with transaction.atomic():
            account.withdraw(amount, description)
    except Exception as e:
        logger.error("Withdrawal failed: %s", e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(EscrowAccountSerializer(account).data)


# ═══════════════════════════════════════════════════════════════
# TRANSACTIONS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def transaction_list(request):
    """List all transactions for the authenticated user."""
    account = _get_account(request.user)
    qs = account.transactions.select_related(
        "account__user", "counterparty_account__user", "job"
    ).order_by("-created_at")

    # Optional filters
    tx_type = request.query_params.get("tx_type")
    if tx_type:
        qs = qs.filter(tx_type=tx_type)

    job_id = request.query_params.get("job_id")
    if job_id:
        qs = qs.filter(job_id=job_id)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        EscrowTransactionSerializer(page, many=True).data
    )
