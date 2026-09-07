from django.db.models import Avg, Min, Max, Sum, Count
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from gpu_escrow.models import EscrowAccount
from gpu_finance.models import Invoice, UsageRecord
from gpu_marketplace.models import GPUListing, FavoriteListing, ComputeOrder, ComputeJob
from gpu_marketplace.serializers import GPUListingSerializer
from gpu_reputation.models import ReputationScore, UserReputationBadge

from gpu_marketplace import views as marketplace_views
from gpu_benchmark import views as benchmark_views


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def listings(request):
    return marketplace_views.listing_list(request)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def listing_detail(request, listing_id):
    return marketplace_views.listing_detail(request, listing_id)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def listing_history(request, listing_id):
    get_object_or_404(GPUListing, id=listing_id)
    return Response([])


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def depth(request):
    qs = (
        GPUListing.objects.filter(status='online')
        .values('gpu_model')
        .annotate(
            total_available=Sum('gpu_count'),
            avg_price=Avg('current_price_per_hour'),
            min_price=Min('current_price_per_hour'),
            max_price=Max('current_price_per_hour'),
            sellers_count=Count('id'),
        )
        .order_by('gpu_model')
    )
    return Response(
        [
            {
                'model': row['gpu_model'],
                'total_available': row['total_available'] or 0,
                'avg_price': float(row['avg_price'] or 0),
                'min_price': float(row['min_price'] or 0),
                'max_price': float(row['max_price'] or 0),
                'sellers_count': row['sellers_count'] or 0,
            }
            for row in qs
        ]
    )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def place_order(request, listing_id):
    listing = get_object_or_404(GPUListing, id=listing_id)
    side = str(request.data.get('side', 'BUY')).lower()
    order_type = 'sell' if side == 'sell' else 'buy'
    quantity = request.data.get('quantity') or 1
    price = request.data.get('price')

    order = ComputeOrder.objects.create(
        user=request.user,
        order_type=order_type,
        order_mode='limit' if price is not None else 'market',
        gpu_model_preference=listing.gpu_model,
        min_vram_gb=listing.vram_gb,
        min_gpu_count=max(int(quantity), 1),
        max_price_per_hour=price or listing.current_price_per_hour,
        required_hours=1,
        docker_image=request.data.get('docker_image', ''),
        command=request.data.get('command', ''),
    )
    return Response({'id': str(order.id), 'status': order.status}, status=201)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def listing_order_book(request, listing_id):
    listing = get_object_or_404(GPUListing, id=listing_id)
    base_qs = ComputeOrder.objects.filter(
        status='open',
        gpu_model_preference__icontains=listing.gpu_model,
    ).order_by('-created_at')
    bids = [
        {
            'id': str(order.id),
            'price': float(order.max_price_per_hour or 0),
            'quantity': order.min_gpu_count,
        }
        for order in base_qs.filter(order_type='buy')[:50]
    ]
    asks = [
        {
            'id': str(order.id),
            'price': float(order.max_price_per_hour or 0),
            'quantity': order.min_gpu_count,
        }
        for order in base_qs.filter(order_type='sell')[:50]
    ]
    return Response({'bids': bids, 'asks': asks})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def listing_depth(request, listing_id):
    listing = get_object_or_404(GPUListing, id=listing_id)
    qs = GPUListing.objects.filter(status='online', gpu_model__iexact=listing.gpu_model)
    return Response(
        {
            'model': listing.gpu_model,
            'total_available': qs.aggregate(total=Sum('gpu_count'))['total'] or 0,
            'avg_price': float(qs.aggregate(avg=Avg('current_price_per_hour'))['avg'] or 0),
            'min_price': float(qs.aggregate(min=Min('current_price_per_hour'))['min'] or 0),
            'max_price': float(qs.aggregate(max=Max('current_price_per_hour'))['max'] or 0),
            'sellers_count': qs.count(),
        }
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def listing_price_history(request, listing_id):
    get_object_or_404(GPUListing, id=listing_id)
    return Response([])


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def watchlist(request):
    if request.method == 'POST':
        listing_id = request.data.get('listing_id')
        listing = get_object_or_404(GPUListing, id=listing_id)
        FavoriteListing.objects.get_or_create(user=request.user, listing=listing)
        return Response({'status': 'ok'})

    favorites = FavoriteListing.objects.filter(user=request.user).select_related('listing')
    return Response(GPUListingSerializer([f.listing for f in favorites], many=True).data)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def remove_watchlist(request, listing_id):
    FavoriteListing.objects.filter(user=request.user, listing_id=listing_id).delete()
    return Response(status=204)


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def orders(request):
    if request.method == 'GET':
        return marketplace_views.my_orders(request)
    return marketplace_views.order_list(request)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_orders(request):
    return marketplace_views.my_orders(request)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cancel_order(request, order_id):
    order = get_object_or_404(ComputeOrder, id=order_id, user=request.user)
    if order.status != 'open':
        return Response({'error': 'Only open orders can be canceled'}, status=400)
    order.status = 'canceled'
    order.save(update_fields=['status'])
    return Response({'status': 'canceled'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def jobs(request):
    response = marketplace_views.job_list(request)
    if isinstance(response.data, dict) and 'results' in response.data:
        return Response(response.data['results'])
    return response


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_detail(request, job_id):
    return marketplace_views.job_detail(request, job_id)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def balance(request):
    account, _ = EscrowAccount.objects.get_or_create(user=request.user)
    return Response(
        {
            'balance_usd': account.balance_usd,
            'locked_balance_usd': account.locked_balance_usd,
        }
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def invoices(request):
    qs = Invoice.objects.filter(user=request.user).order_by('-created_at')[:100]
    return Response(
        [
            {
                'id': str(inv.id),
                'invoice_number': inv.invoice_number,
                'status': inv.status,
                'total_usd': inv.total_usd,
            }
            for inv in qs
        ]
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def usage(request):
    qs = UsageRecord.objects.filter(user=request.user).order_by('-recorded_at')[:100]
    total_hours = qs.aggregate(total=Sum('hours_consumed'))['total'] or 0
    total_cost = qs.aggregate(total=Sum('cost_usd'))['total'] or 0
    return Response(
        {
            'total_hours': float(total_hours),
            'total_cost': float(total_cost),
            'records': [
                {
                    'job_id': str(item.job_id) if item.job_id else None,
                    'gpu_model': item.gpu_model,
                    'hours': float(item.hours_consumed or 0),
                    'cost': float(item.cost_usd or 0),
                }
                for item in qs
            ],
        }
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def leaderboard(request):
    response = benchmark_views.gpu_leaderboard(request)
    data = response.data if isinstance(response.data, dict) else {}
    return Response(data.get('results', []))


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def reputation(request):
    score, _ = ReputationScore.objects.get_or_create(user=request.user)
    badges = UserReputationBadge.objects.filter(user=request.user).select_related('badge')
    return Response(
        {
            'provider_score': score.provider_score,
            'consumer_score': score.consumer_score,
            'badges': [
                {
                    'name': badge.badge.name,
                    'icon': badge.badge.icon,
                    'tier': badge.badge.tier,
                }
                for badge in badges
            ],
        }
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def provider_stats(request):
    return marketplace_views.provider_dashboard(request)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def provider_usage(request):
    jobs_qs = (
        ComputeJob.objects.filter(provider=request.user)
        .exclude(hours_consumed__isnull=True)
        .order_by('-created_at')[:100]
    )
    return Response(
        {
            'total_hours': float(jobs_qs.aggregate(total=Sum('hours_consumed'))['total'] or 0),
            'total_earnings': float(jobs_qs.aggregate(total=Sum('total_cost_usd'))['total'] or 0),
            'records': [
                {
                    'job_id': str(job.id),
                    'gpu_model': job.listing.gpu_model,
                    'hours': float(job.hours_consumed or 0),
                    'earnings': float(job.total_cost_usd or 0),
                }
                for job in jobs_qs
            ],
        }
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def provider_revenue(request):
    return Response([])
