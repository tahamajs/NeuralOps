"""GPU Finance REST API — indices, futures, billing, payouts, subscriptions."""
import logging
from decimal import Decimal
from django.shortcuts import get_object_or_404
from django.db.models import Q, Sum, Count
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import *
from .serializers import *
from .pricing_engine import recalculate_all_indices, get_index_history, calculate_settlement_price

logger = logging.getLogger(__name__)

class StandardPagination(PageNumberPagination):
    page_size = 20; page_size_query_param = 'page_size'; max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# COMPUTE INDICES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def index_list(request):
    qs = ComputeIndex.objects.all().order_by('gpu_model')
    return Response(ComputeIndexSerializer(qs, many=True).data)

@api_view(['GET'])
@permission_classes([AllowAny])
def index_detail(request, index_id):
    idx = get_object_or_404(ComputeIndex, id=index_id)
    data = ComputeIndexSerializer(idx).data
    data['history'] = IndexHistorySerializer(
        idx.history.order_by('-recorded_at')[:30], many=True
    ).data
    return Response(data)

@api_view(['GET'])
@permission_classes([AllowAny])
def index_chart(request, index_id):
    idx = get_object_or_404(ComputeIndex, id=index_id)
    days = int(request.query_params.get('days', 30))
    return Response(get_index_history(idx.index_name, days))

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def recalculate_indices(request):
    if not request.user.is_staff:
        return Response({"error": "Staff only"}, status=403)
    return Response(recalculate_all_indices())


# ═══════════════════════════════════════════════════════════════
# FUTURES CONTRACTS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def futures_list(request):
    qs = FuturesContract.objects.filter(status='open').order_by('contract_year', 'contract_month')
    return Response(FuturesContractSerializer(qs, many=True).data)

@api_view(['GET'])
@permission_classes([AllowAny])
def futures_detail(request, contract_id):
    contract = get_object_or_404(FuturesContract, id=contract_id)
    data = FuturesContractSerializer(contract).data
    data['order_book'] = {
        'bids': FuturesOrderSerializer(
            contract.orders.filter(order_type='buy', status='open').order_by('-limit_price')[:10],
            many=True
        ).data,
        'asks': FuturesOrderSerializer(
            contract.orders.filter(order_type='sell', status='open').order_by('limit_price')[:10],
            many=True
        ).data,
    }
    return Response(data)


# ═══════════════════════════════════════════════════════════════
# FUTURES ORDERS
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def place_futures_order(request, contract_id):
    contract = get_object_or_404(FuturesContract, id=contract_id)
    if contract.status != 'open':
        return Response({"error": "Contract is not open for trading"}, status=400)

    serializer = FuturesOrderSerializer(data=request.data)
    if serializer.is_valid():
        order = serializer.save(contract=contract, trader=request.user)
        return Response(FuturesOrderSerializer(order).data, status=201)
    return Response(serializer.errors, status=400)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_futures_orders(request):
    qs = FuturesOrder.objects.filter(trader=request.user).order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(FuturesOrderSerializer(page, many=True).data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_futures_positions(request):
    qs = FuturesPosition.objects.filter(trader=request.user).order_by('-opened_at')
    total_pnl = qs.aggregate(s=Sum('realized_pnl_usd'))['s'] or 0
    data = FuturesPositionSerializer(qs, many=True).data
    return Response({'positions': data, 'total_realized_pnl': float(total_pnl)})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cancel_futures_order(request, order_id):
    order = get_object_or_404(FuturesOrder, id=order_id, trader=request.user)
    if order.status != 'open':
        return Response({"error": "Order already filled or canceled"}, status=400)
    order.status = 'canceled'; order.save(update_fields=['status'])
    return Response({'status': 'canceled'})


# ═══════════════════════════════════════════════════════════════
# INVOICES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def invoice_list(request):
    qs = Invoice.objects.filter(user=request.user).order_by('-created_at')
    status_filter = request.query_params.get('status')
    if status_filter: qs = qs.filter(status=status_filter)
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(InvoiceSerializer(page, many=True).data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def invoice_detail(request, invoice_id):
    inv = get_object_or_404(Invoice, id=invoice_id, user=request.user)
    return Response(InvoiceSerializer(inv).data)


# ═══════════════════════════════════════════════════════════════
# PAYOUTS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def payout_list(request):
    qs = Payout.objects.filter(provider=request.user).order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(PayoutSerializer(page, many=True).data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def request_payout(request):
    amount = request.data.get('amount_usd')
    method = request.data.get('method', 'credits')
    if not amount: return Response({"error": "amount_usd required"}, status=400)
    from gpu_escrow.models import EscrowAccount
    account = get_object_or_404(EscrowAccount, user=request.user)
    if Decimal(str(amount)) > account.balance_usd:
        return Response({"error": "Insufficient balance"}, status=400)
    payout = Payout.objects.create(
        provider=request.user, amount_usd=amount, method=method,
        fee_usd=0, net_amount_usd=amount
    )
    return Response(PayoutSerializer(payout).data, status=201)


# ═══════════════════════════════════════════════════════════════
# PAYMENT METHODS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def payment_method_list(request):
    if request.method == 'GET':
        return Response(PaymentMethodSerializer(
            request.user.payment_methods.all(), many=True
        ).data)
    elif request.method == 'POST':
        serializer = PaymentMethodSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def payment_method_delete(request, method_id):
    pm = get_object_or_404(PaymentMethod, id=method_id, user=request.user)
    pm.delete()
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# SUBSCRIPTIONS & PLANS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def plan_list(request):
    return Response(BillingPlanSerializer(
        BillingPlan.objects.filter(is_active=True), many=True
    ).data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_subscription(request):
    sub, _ = Subscription.objects.get_or_create(user=request.user)
    return Response(SubscriptionSerializer(sub).data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_plan(request, plan_id):
    plan = get_object_or_404(BillingPlan, id=plan_id, is_active=True)
    sub, _ = Subscription.objects.get_or_create(user=request.user)
    sub.plan = plan; sub.status = 'active'
    sub.current_period_start = timezone.now()
    sub.current_period_end = timezone.now() + timezone.timedelta(days=30)
    sub.save()
    return Response(SubscriptionSerializer(sub).data)


# ═══════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_usage(request):
    qs = UsageRecord.objects.filter(user=request.user).order_by('-recorded_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    total = qs.aggregate(
        hours=Sum('hours_consumed'), cost=Sum('cost_usd'), fee=Sum('platform_fee_usd')
    )
    data = {
        'records': UsageRecordSerializer(page, many=True).data,
        'total_hours': float(total['hours'] or 0),
        'total_cost': float(total['cost'] or 0),
        'total_platform_fees': float(total['fee'] or 0),
    }
    return paginator.get_paginated_response(data)


# ═══════════════════════════════════════════════════════════════
# CONSUMER DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def consumer_finance_dashboard(request):
    invoices = Invoice.objects.filter(user=request.user)
    unpaid = invoices.filter(status__in=['open', 'overdue']).aggregate(s=Sum('total_usd'))['s'] or 0
    usage = UsageRecord.objects.filter(user=request.user)
    return Response({
        'balance_due_usd': float(unpaid),
        'total_spent_usd': float(usage.aggregate(s=Sum('cost_usd'))['s'] or 0),
        'total_hours': float(usage.aggregate(s=Sum('hours_consumed'))['s'] or 0),
        'open_invoices': invoices.filter(status='open').count(),
        'futures_positions': FuturesPosition.objects.filter(trader=request.user, status='open').count(),
        'futures_pnl': float(FuturesPosition.objects.filter(trader=request.user)
                            .aggregate(s=Sum('realized_pnl_usd'))['s'] or 0),
    })


# ═══════════════════════════════════════════════════════════════
# PROVIDER DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def provider_finance_dashboard(request):
    payouts = Payout.objects.filter(provider=request.user)
    from gpu_marketplace.models import GPUListing
    listings = GPUListing.objects.filter(provider=request.user)
    return Response({
        'total_earned_usd': float(listings.aggregate(s=Sum('total_earnings_usd'))['s'] or 0),
        'pending_payouts': float(payouts.filter(status='pending').aggregate(s=Sum('net_amount_usd'))['s'] or 0),
        'lifetime_payouts': float(payouts.filter(status='completed').aggregate(s=Sum('net_amount_usd'))['s'] or 0),
        'total_hours_rented': float(listings.aggregate(s=Sum('total_hours_rented'))['s'] or 0),
    })