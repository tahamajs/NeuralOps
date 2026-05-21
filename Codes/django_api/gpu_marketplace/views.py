"""
GPU Marketplace REST API — complete CRUD + search + matching + billing.
"""
import logging
from django.shortcuts import get_object_or_404
from django.db.models import Q, Count, Avg, F
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import *
from .serializers import *
from .tasks import provision_job

logger = logging.getLogger(__name__)

class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# GPU MODEL SPECS (Reference Data)
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def gpu_model_list(request):
    """List all known GPU model specifications."""
    qs = GPUModelSpec.objects.filter(is_active=True).annotate(
        listing_count=Count('listings')
    ).order_by('manufacturer', 'name')
    search = request.query_params.get('q')
    if search:
        qs = qs.filter(Q(name__icontains=search) | Q(manufacturer__icontains=search))
    return Response(GPUModelSpecSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# GPU LISTINGS — Search, Create, Detail, Update, Delete
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def listing_list(request):
    """List available GPU listings or create a new one."""
    if request.method == 'GET':
        qs = GPUListing.objects.select_related('provider', 'gpu_model_spec').all()

        # Filters (like Vast.ai search)
        status_filter = request.query_params.get('status', 'online')
        qs = qs.filter(status=status_filter)

        gpu_model = request.query_params.get('gpu_model')
        if gpu_model:
            qs = qs.filter(gpu_model__icontains=gpu_model)

        min_vram = request.query_params.get('min_vram')
        if min_vram:
            qs = qs.filter(vram_gb__gte=int(min_vram))

        max_price = request.query_params.get('max_price')
        if max_price:
            qs = qs.filter(current_price_per_hour__lte=float(max_price))

        min_reliability = request.query_params.get('min_reliability')
        if min_reliability:
            qs = qs.filter(reliability_score__gte=float(min_reliability))

        listing_type = request.query_params.get('listing_type')
        if listing_type:
            qs = qs.filter(listing_type=listing_type)

        verification = request.query_params.get('verification')
        if verification:
            qs = qs.filter(verification_level=verification)

        location = request.query_params.get('location')
        if location:
            qs = qs.filter(location__icontains=location)

        sort = request.query_params.get('sort', 'reliability')
        sort_map = {
            'price_asc': 'current_price_per_hour',
            'price_desc': '-current_price_per_hour',
            'reliability': '-reliability_score',
            'dlperf': '-dlperf_score',
            'newest': '-created_at',
        }
        qs = qs.order_by(sort_map.get(sort, '-reliability_score'))

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(GPUListingSerializer(page, many=True).data)

    elif request.method == 'POST':
        serializer = GPUListingCreateSerializer(data=request.data)
        if serializer.is_valid():
            listing = serializer.save(
                provider=request.user,
                current_price_per_hour=serializer.validated_data['base_price_per_hour']
            )
            return Response(GPUListingSerializer(listing).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def listing_detail(request, listing_id):
    """Get, update, or delete a GPU listing."""
    listing = get_object_or_404(GPUListing, id=listing_id)

    if request.method == 'GET':
        return Response(GPUListingSerializer(listing).data)

    elif request.method == 'PATCH':
        if listing.provider != request.user:
            return Response({"error": "Only the provider can edit"}, status=403)
        serializer = GPUListingCreateSerializer(listing, data=request.data, partial=True)
        if serializer.is_valid():
            updated = serializer.save()
            if 'base_price_per_hour' in request.data:
                updated.current_price_per_hour = updated.base_price_per_hour
                updated.save()
            return Response(GPUListingSerializer(updated).data)
        return Response(serializer.errors, status=400)

    elif request.method == 'DELETE':
        if listing.provider != request.user:
            return Response({"error": "Only the provider can delete"}, status=403)
        listing.delete()
        return Response(status=204)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def listing_toggle_online(request, listing_id):
    """Toggle a listing between online and offline."""
    listing = get_object_or_404(GPUListing, id=listing_id, provider=request.user)
    listing.status = 'offline' if listing.status == 'online' else 'online'
    listing.save(update_fields=['status'])
    return Response({'status': listing.status})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_listings(request):
    """List the authenticated user's own GPU listings."""
    qs = GPUListing.objects.filter(provider=request.user).order_by('-created_at')
    serializer = GPUListingSerializer(qs, many=True)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# COMPUTE ORDERS — Place, List, Detail, Cancel
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def order_list(request):
    """List open orders or place a new one."""
    if request.method == 'GET':
        qs = ComputeOrder.objects.select_related('user').all()
        order_type = request.query_params.get('order_type')
        if order_type:
            qs = qs.filter(order_type=order_type)
        status_filter = request.query_params.get('status', 'open')
        if status_filter:
            qs = qs.filter(status=status_filter)
        gpu_model = request.query_params.get('gpu_model')
        if gpu_model:
            qs = qs.filter(gpu_model_preference__icontains=gpu_model)
        qs = qs.order_by('-created_at')
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(ComputeOrderSerializer(page, many=True).data)

    elif request.method == 'POST':
        serializer = ComputeOrderSerializer(data=request.data)
        if serializer.is_valid():
            order = serializer.save(user=request.user)
            return Response(ComputeOrderSerializer(order).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def order_detail(request, order_id):
    """Get or cancel a compute order."""
    order = get_object_or_404(ComputeOrder, id=order_id)

    if request.method == 'GET':
        return Response(ComputeOrderSerializer(order).data)

    elif request.method == 'DELETE':
        if order.user != request.user:
            return Response({"error": "Not allowed"}, status=403)
        if order.status != 'open':
            return Response({"error": "Only open orders can be canceled"}, status=400)
        order.status = 'canceled'
        order.save(update_fields=['status'])
        return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_orders(request):
    """List the authenticated user's own orders."""
    qs = ComputeOrder.objects.filter(user=request.user).order_by('-created_at')
    serializer = ComputeOrderSerializer(qs, many=True)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# AUCTION BIDS
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def place_bid(request, order_id):
    """Place a bid on an auction-mode order."""
    order = get_object_or_404(ComputeOrder, id=order_id, order_mode='auction', status='open')
    listing_id = request.data.get('listing_id')
    price = request.data.get('price_per_hour')

    if not listing_id or not price:
        return Response({"error": "listing_id and price_per_hour required"}, status=400)

    listing = get_object_or_404(GPUListing, id=listing_id, status='online')

    bid, created = Bid.objects.update_or_create(
        order=order, listing=listing, bidder=request.user,
        defaults={'price_per_hour': price}
    )
    return Response(BidSerializer(bid).data, status=201 if created else 200)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def order_bids(request, order_id):
    """List all bids on an auction order."""
    order = get_object_or_404(ComputeOrder, id=order_id)
    bids = order.bids.order_by('-price_per_hour')
    return Response(BidSerializer(bids, many=True).data)


# ═══════════════════════════════════════════════════════════════
# COMPUTE JOBS — List, Detail, Complete, Fail
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_list(request):
    """List the authenticated user's jobs."""
    qs = ComputeJob.objects.filter(
        Q(consumer=request.user) | Q(provider=request.user)
    ).select_related('listing', 'consumer', 'provider').order_by('-created_at')

    status_filter = request.query_params.get('status')
    if status_filter:
        qs = qs.filter(status=status_filter)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ComputeJobSerializer(page, many=True).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_detail(request, job_id):
    """Get detailed info about a compute job."""
    job = get_object_or_404(ComputeJob, id=job_id)
    if job.consumer != request.user and job.provider != request.user:
        return Response({"error": "Not allowed"}, status=403)
    return Response(ComputeJobSerializer(job).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_complete(request, job_id):
    """Mark a job as completed (provider action)."""
    job = get_object_or_404(ComputeJob, id=job_id, provider=request.user)
    if job.status not in ('running', 'verifying'):
        return Response({"error": f"Cannot complete job in status: {job.status}"}, status=400)
    output_url = request.data.get('output_data_url', '')
    job.mark_completed(output_url)
    JobEvent.objects.create(job=job, event_type='completed')
    return Response(ComputeJobSerializer(job).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_fail(request, job_id):
    """Mark a job as failed (provider action)."""
    job = get_object_or_404(ComputeJob, id=job_id, provider=request.user)
    error_msg = request.data.get('error_message', '')
    job.mark_failed(error_msg)
    JobEvent.objects.create(job=job, event_type='failed',
                            description=error_msg)
    return Response(ComputeJobSerializer(job).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def job_dispute(request, job_id):
    """Dispute a job (consumer action)."""
    job = get_object_or_404(ComputeJob, id=job_id, consumer=request.user)
    if job.status not in ('completed', 'failed'):
        return Response({"error": "Can only dispute completed or failed jobs"}, status=400)
    job.status = 'disputed'
    job.save(update_fields=['status'])
    JobEvent.objects.create(job=job, event_type='disputed',
                            description=request.data.get('reason', ''))
    return Response(ComputeJobSerializer(job).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def provision_job_view(request, job_id):
    """Trigger provisioning of a matched job."""
    job = get_object_or_404(ComputeJob, id=job_id)
    if job.status != 'queued':
        return Response({"error": "Job must be in queued status"}, status=400)
    provision_job.delay(str(job.id))
    return Response({'status': 'provisioning started'})


# ═══════════════════════════════════════════════════════════════
# FAVORITES
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def favorite_listing(request, listing_id):
    listing = get_object_or_404(GPUListing, id=listing_id)
    fav, created = FavoriteListing.objects.get_or_create(
        user=request.user, listing=listing
    )
    return Response({'status': 'favorited' if created else 'already_favorited'})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def unfavorite_listing(request, listing_id):
    FavoriteListing.objects.filter(user=request.user, listing_id=listing_id).delete()
    return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def my_favorites(request):
    favs = FavoriteListing.objects.filter(user=request.user).select_related('listing')
    return Response(FavoriteListingSerializer(favs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# PRICING RULES (Admin / Staff)
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def pricing_rule_list(request):
    qs = PricingRule.objects.filter(is_active=True).order_by('gpu_model')
    return Response(PricingRuleSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# MARKETPLACE STATS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([AllowAny])
def marketplace_stats(request):
    """Public marketplace statistics."""
    return Response({
        'total_gpus_online': GPUListing.objects.filter(status='online').count(),
        'total_listings': GPUListing.objects.count(),
        'active_jobs': ComputeJob.objects.filter(status__in=['queued','provisioning','running']).count(),
        'completed_jobs': ComputeJob.objects.filter(status='completed').count(),
        'avg_price_per_hour': round(
            GPUListing.objects.filter(status='online').aggregate(
                a=Avg('current_price_per_hour'))['a'] or 0, 4
        ),
        'most_available': list(
            GPUListing.objects.filter(status='online')
            .values('gpu_model').annotate(c=Count('id')).order_by('-c')[:5]
        ),
    })


# ═══════════════════════════════════════════════════════════════
# PROVIDER DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def provider_dashboard(request):
    """Provider earnings and stats dashboard."""
    listings = GPUListing.objects.filter(provider=request.user)
    jobs = ComputeJob.objects.filter(provider=request.user)
    return Response({
        'total_listings': listings.count(),
        'online_listings': listings.filter(status='online').count(),
        'total_earnings': float(
            listings.aggregate(s=models.Sum('total_earnings_usd'))['s'] or 0
        ),
        'total_hours_rented': float(
            listings.aggregate(s=models.Sum('total_hours_rented'))['s'] or 0
        ),
        'active_jobs': jobs.filter(status__in=['running', 'provisioning']).count(),
        'completed_jobs': jobs.filter(status='completed').count(),
        'failed_jobs': jobs.filter(status='failed').count(),
        'avg_reliability': round(
            listings.aggregate(a=Avg('reliability_score'))['a'] or 0, 2
        ),
    })