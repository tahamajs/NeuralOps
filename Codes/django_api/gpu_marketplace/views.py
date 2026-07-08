"""
GPU Marketplace REST API — complete CRUD + search + matching + billing.
"""

import logging

from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *
from .tasks import provision_job

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# GPU MODEL SPECS (Reference Data)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def gpu_model_list(request):
    """List all known GPU model specifications."""
    qs = (
        GPUModelSpec.objects.filter(is_active=True)
        .annotate(listing_count=Count("listings"))
        .order_by("manufacturer", "name")
    )
    search = request.query_params.get("q")
    if search:
        qs = qs.filter(Q(name__icontains=search) | Q(manufacturer__icontains=search))
    return Response(GPUModelSpecSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# GPU LISTINGS — Search, Create, Detail, Update, Delete
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def listing_list(request):
    """List available GPU listings or create a new one."""
    if request.method == "GET":
        qs = GPUListing.objects.select_related("provider", "gpu_model_spec").all()

        # Filters (like Vast.ai search)
        status_filter = request.query_params.get("status", "online")
        qs = qs.filter(status=status_filter)

        gpu_model = request.query_params.get("gpu_model")
        if gpu_model:
            qs = qs.filter(gpu_model__icontains=gpu_model)

        min_vram = request.query_params.get("min_vram")
        if min_vram:
            qs = qs.filter(vram_gb__gte=int(min_vram))

        max_price = request.query_params.get("max_price")
        if max_price:
            qs = qs.filter(current_price_per_hour__lte=float(max_price))

        min_reliability = request.query_params.get("min_reliability")
        if min_reliability:
            qs = qs.filter(reliability_score__gte=float(min_reliability))

        listing_type = request.query_params.get("listing_type")
        if listing_type:
            qs = qs.filter(listing_type=listing_type)

        verification = request.query_params.get("verification")
        if verification:
            qs = qs.filter(verification_level=verification)

        location = request.query_params.get("location")
        if location:
            qs = qs.filter(location__icontains=location)

        sort = request.query_params.get("sort", "reliability")
        sort_map = {
            "price_asc": "current_price_per_hour",
            "price_desc": "-current_price_per_hour",
            "reliability": "-reliability_score",
            "dlperf": "-dlperf_score",
            "newest": "-created_at",
        }
        qs = qs.order_by(sort_map.get(sort, "-reliability_score"))

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            GPUListingSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = GPUListingCreateSerializer(data=request.data)
        if serializer.is_valid():
            listing = serializer.save(
                provider=request.user,
                current_price_per_hour=serializer.validated_data["base_price_per_hour"],
            )
            return Response(GPUListingSerializer(listing).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def listing_detail(request, listing_id):
    """Get, update, or delete a GPU listing."""
    listing = get_object_or_404(GPUListing, id=listing_id)

    if request.method == "GET":
        return Response(GPUListingSerializer(listing).data)

    elif request.method == "PATCH":
        if listing.provider != request.user:
            return Response({"error": "Only the provider can edit"}, status=403)
        serializer = GPUListingCreateSerializer(
            listing, data=request.data, partial=True
        )
        if serializer.is_valid():
            updated = serializer.save()
            if "base_price_per_hour" in request.data:
                updated.current_price_per_hour = updated.base_price_per_hour
                updated.save()
            return Response(GPUListingSerializer(updated).data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        if listing.provider != request.user:
            return Response({"error": "Only the provider can delete"}, status=403)
        listing.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def listing_toggle_online(request, listing_id):
    """Toggle a listing between online and offline."""
    listing = get_object_or_404(GPUListing, id=listing_id, provider=request.user)
    listing.status = "offline" if listing.status == "online" else "online"
    listing.save(update_fields=["status"])
    return Response({"status": listing.status})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_listings(request):
    """List the authenticated user's own GPU listings."""
    qs = GPUListing.objects.filter(provider=request.user).order_by("-created_at")
    serializer = GPUListingSerializer(qs, many=True)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# COMPUTE ORDERS — Place, List, Detail, Cancel
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def order_list(request):
    """List open orders or place a new one."""
    if request.method == "GET":
        qs = ComputeOrder.objects.select_related("user").all()
        order_type = request.query_params.get("order_type")
        if order_type:
            qs = qs.filter(order_type=order_type)
        status_filter = request.query_params.get("status", "open")
        if status_filter:
            qs = qs.filter(status=status_filter)
        gpu_model = request.query_params.get("gpu_model")
        if gpu_model:
            qs = qs.filter(gpu_model_preference__icontains=gpu_model)
        qs = qs.order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ComputeOrderSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ComputeOrderSerializer(data=request.data)
        if serializer.is_valid():
            order = serializer.save(user=request.user)
            return Response(ComputeOrderSerializer(order).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def order_detail(request, order_id):
    """Get or cancel a compute order."""
    order = get_object_or_404(ComputeOrder, id=order_id)

    if request.method == "GET":
        return Response(ComputeOrderSerializer(order).data)

    elif request.method == "DELETE":
        if order.user != request.user:
            return Response({"error": "Not allowed"}, status=403)
        if order.status != "open":
            return Response({"error": "Only open orders can be canceled"}, status=400)
        order.status = "canceled"
        order.save(update_fields=["status"])
        return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_orders(request):
    """List the authenticated user's own orders."""
    qs = ComputeOrder.objects.filter(user=request.user).order_by("-created_at")
    serializer = ComputeOrderSerializer(qs, many=True)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# AUCTION BIDS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def place_bid(request, order_id):
    """Place a bid on an auction-mode order."""
    order = get_object_or_404(
        ComputeOrder, id=order_id, order_mode="auction", status="open"
    )
    listing_id = request.data.get("listing_id")
    price = request.data.get("price_per_hour")

    if not listing_id or not price:
        return Response({"error": "listing_id and price_per_hour required"}, status=400)

    listing = get_object_or_404(GPUListing, id=listing_id, status="online")

    bid, created = Bid.objects.update_or_create(
        order=order,
        listing=listing,
        bidder=request.user,
        defaults={"price_per_hour": price},
    )
    return Response(BidSerializer(bid).data, status=201 if created else 200)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def order_bids(request, order_id):
    """List all bids on an auction order."""
    order = get_object_or_404(ComputeOrder, id=order_id)
    bids = order.bids.order_by("-price_per_hour")
    return Response(BidSerializer(bids, many=True).data)


# ═══════════════════════════════════════════════════════════════
# COMPUTE JOBS — List, Detail, Complete, Fail
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def job_list(request):
    """List the authenticated user's jobs."""
    qs = (
        ComputeJob.objects.filter(Q(consumer=request.user) | Q(provider=request.user))
        .select_related("listing", "consumer", "provider")
        .order_by("-created_at")
    )

    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ComputeJobSerializer(page, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def job_detail(request, job_id):
    """Get detailed info about a compute job."""
    job = get_object_or_404(ComputeJob, id=job_id)
    if job.consumer != request.user and job.provider != request.user:
        return Response({"error": "Not allowed"}, status=403)
    return Response(ComputeJobSerializer(job).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_complete(request, job_id):
    """Mark a job as completed (provider action)."""
    job = get_object_or_404(ComputeJob, id=job_id, provider=request.user)
    if job.status not in ("running", "verifying"):
        return Response(
            {"error": f"Cannot complete job in status: {job.status}"}, status=400
        )
    output_url = request.data.get("output_data_url", "")
    job.mark_completed(output_url)
    JobEvent.objects.create(job=job, event_type="completed")
    return Response(ComputeJobSerializer(job).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_fail(request, job_id):
    """Mark a job as failed (provider action)."""
    job = get_object_or_404(ComputeJob, id=job_id, provider=request.user)
    error_msg = request.data.get("error_message", "")
    job.mark_failed(error_msg)
    JobEvent.objects.create(job=job, event_type="failed", description=error_msg)
    return Response(ComputeJobSerializer(job).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_dispute(request, job_id):
    """Dispute a job (consumer action)."""
    job = get_object_or_404(ComputeJob, id=job_id, consumer=request.user)
    if job.status not in ("completed", "failed"):
        return Response(
            {"error": "Can only dispute completed or failed jobs"}, status=400
        )
    job.status = "disputed"
    job.save(update_fields=["status"])
    JobEvent.objects.create(
        job=job, event_type="disputed", description=request.data.get("reason", "")
    )
    return Response(ComputeJobSerializer(job).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def provision_job_view(request, job_id):
    """Trigger provisioning of a matched job."""
    job = get_object_or_404(ComputeJob, id=job_id)
    if job.status != "queued":
        return Response({"error": "Job must be in queued status"}, status=400)
    provision_job.delay(str(job.id))
    return Response({"status": "provisioning started"})


# ═══════════════════════════════════════════════════════════════
# FAVORITES
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def favorite_listing(request, listing_id):
    listing = get_object_or_404(GPUListing, id=listing_id)
    fav, created = FavoriteListing.objects.get_or_create(
        user=request.user, listing=listing
    )
    return Response({"status": "favorited" if created else "already_favorited"})


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def unfavorite_listing(request, listing_id):
    FavoriteListing.objects.filter(user=request.user, listing_id=listing_id).delete()
    return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_favorites(request):
    favs = FavoriteListing.objects.filter(user=request.user).select_related("listing")
    return Response(FavoriteListingSerializer(favs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# PRICING RULES (Admin / Staff)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def pricing_rule_list(request):
    qs = PricingRule.objects.filter(is_active=True).order_by("gpu_model")
    return Response(PricingRuleSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# MARKETPLACE STATS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def marketplace_stats(request):
    """Public marketplace statistics."""
    return Response(
        {
            "total_gpus_online": GPUListing.objects.filter(status="online").count(),
            "total_listings": GPUListing.objects.count(),
            "active_jobs": ComputeJob.objects.filter(
                status__in=["queued", "provisioning", "running"]
            ).count(),
            "completed_jobs": ComputeJob.objects.filter(status="completed").count(),
            "avg_price_per_hour": round(
                GPUListing.objects.filter(status="online").aggregate(
                    a=Avg("current_price_per_hour")
                )["a"]
                or 0,
                4,
            ),
            "most_available": list(
                GPUListing.objects.filter(status="online")
                .values("gpu_model")
                .annotate(c=Count("id"))
                .order_by("-c")[:5]
            ),
        }
    )


# ═══════════════════════════════════════════════════════════════
# PROVIDER DASHBOARD
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def provider_dashboard(request):
    """Provider earnings and stats dashboard."""
    listings = GPUListing.objects.filter(provider=request.user)
    jobs = ComputeJob.objects.filter(provider=request.user)
    return Response(
        {
            "total_listings": listings.count(),
            "online_listings": listings.filter(status="online").count(),
            "total_earnings": float(
                listings.aggregate(s=models.Sum("total_earnings_usd"))["s"] or 0
            ),
            "total_hours_rented": float(
                listings.aggregate(s=models.Sum("total_hours_rented"))["s"] or 0
            ),
            "active_jobs": jobs.filter(status__in=["running", "provisioning"]).count(),
            "completed_jobs": jobs.filter(status="completed").count(),
            "failed_jobs": jobs.filter(status="failed").count(),
            "avg_reliability": round(
                listings.aggregate(a=Avg("reliability_score"))["a"] or 0, 2
            ),
        }
    )


# gpu_marketplace/views.py — ADDITIONAL VIEWS

# ── Import other apps (optional, wrap in try/except) ──────────
try:
    from experiment.models import Notification
except ImportError:
    Notification = None

try:
    from portfolios.models import Portfolio, ShowcaseItem
except ImportError:
    Portfolio = ShowcaseItem = None

try:
    from notes.models import ResearcherNote
except ImportError:
    ResearcherNote = None


# ═══════════════════════════════════════════════════════════════
# LINKED RESOURCES — Show related experiments/notes
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def listing_linked_experiments(request, listing_id):
    """Get experiments (TrainingRuns) associated with this listing."""
    listing = get_object_or_404(GPUListing, id=listing_id)
    from jobscheduler.models import Job

    jobs = Job.objects.filter(
        provider=listing.provider,
        benchmark_submissions__isnull=False,  # jobs that are part of benchmarks
    ).order_by("-created_at")[:10]
    return Response(
        [{"id": str(j.id), "name": j.name, "status": j.status} for j in jobs]
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def listing_linked_notes(request, listing_id):
    """Get researcher notes about this GPU."""
    listing = get_object_or_404(GPUListing, id=listing_id)
    if ResearcherNote:
        notes = ResearcherNote.objects.filter(
            tags__contains=[listing.gpu_model],
            deleted_at__isnull=True,
        ).order_by("-created_at")[:10]
        return Response([{"id": str(n.id), "title": n.title} for n in notes])
    return Response([])


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO INTEGRATION — Show provider's showcase items
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def provider_portfolio(request, listing_id):
    """Get the provider's public portfolio."""
    listing = get_object_or_404(GPUListing, id=listing_id)
    if Portfolio:
        portfolio = Portfolio.objects.filter(user=listing.provider).first()
        if portfolio:
            from portfolios.serializers import PortfolioSerializer

            return Response(PortfolioSerializer(portfolio).data)
    return Response({"error": "Portfolio not available"}, status=404)


# ═══════════════════════════════════════════════════════════════
# NOTIFICATIONS — List marketplace-related notifications
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def marketplace_notifications(request):
    """List marketplace-related notifications for the current user."""
    if Notification:
        notifs = Notification.objects.filter(
            recipient=request.user,
            type__in=[
                "job_matched",
                "job_completed",
                "job_failed",
                "payment_released",
                "dispute_raised",
            ],
        ).order_by("-created_at")[:50]
        return Response(
            [
                {
                    "id": str(n.id),
                    "type": n.type,
                    "content": n.content,
                    "is_read": n.is_read,
                    "created_at": n.created_at,
                }
                for n in notifs
            ]
        )
    return Response([])


# gpu_marketplace/views.py — ADD THESE VIEWS

# ═══════════════════════════════════════════════════════════════
# PROVIDER AGENT ENDPOINTS — Registration, Heartbeat, Job Updates
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def register_listing(request):
    """
    Create or update a GPU listing from the provider agent.
    Called once when the agent starts.
    """
    gpu_model = request.data.get("gpu_model")
    # Try to match with known GPU model spec
    gpu_spec = GPUModelSpec.objects.filter(name__icontains=gpu_model).first()

    listing, created = GPUListing.objects.update_or_create(
        provider=request.user,
        gpu_model=gpu_model,
        defaults={
            "gpu_model_spec": gpu_spec,
            "gpu_count": request.data.get("gpu_count", 1),
            "vram_gb": request.data.get("vram_gb", 0),
            "cpu_cores": request.data.get("cpu_cores", 8),
            "ram_gb": request.data.get("ram_gb", 32),
            "storage_gb": request.data.get("storage_gb", 100),
            "base_price_per_hour": request.data.get("base_price_per_hour", 0.50),
            "current_price_per_hour": request.data.get("base_price_per_hour", 0.50),
            "min_price_per_hour": request.data.get("min_price_per_hour"),
            "listing_type": request.data.get("listing_type", "on_demand"),
            "location": request.data.get("location", ""),
            "bandwidth_mbps": request.data.get("bandwidth_mbps", 1000),
            "supports_ssh": request.data.get("supports_ssh", True),
            "supports_jupyter": request.data.get("supports_jupyter", False),
            "status": "online",
        },
    )
    return Response(GPUListingSerializer(listing).data, status=201 if created else 200)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_update_status(request):
    """
    Update the online/offline status of the provider's listing.
    Called on agent startup/shutdown.
    """
    status_val = request.data.get("status", "offline")
    listing = GPUListing.objects.filter(provider=request.user).first()
    if listing:
        listing.status = status_val
        listing.save(update_fields=["status"])
        return Response({"status": listing.status})
    return Response({"error": "No listing found"}, status=404)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_heartbeat(request):
    """
    Provider agent heartbeat — updates last active timestamp.
    """
    listing = GPUListing.objects.filter(provider=request.user).first()
    if listing:
        listing.save(update_fields=["updated_at"])  # just update timestamp
        return Response({"status": "ok"})
    return Response({"error": "No listing found"}, status=404)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_update_status(request, job_id):
    """Agent reports a job status change (e.g., provisioning, running)."""
    job = get_object_or_404(ComputeJob, id=job_id, provider=request.user)
    new_status = request.data.get("status")
    if new_status:
        job.status = new_status
        if new_status == "running" and not job.started_at:
            job.started_at = timezone.now()
        job.save()
        JobEvent.objects.create(
            job=job,
            event_type=new_status,
            description=request.data.get("description", ""),
        )
    return Response(ComputeJobSerializer(job).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_update_logs(request, job_id):
    """Agent streams logs for a running job."""
    job = get_object_or_404(ComputeJob, id=job_id, provider=request.user)
    logs = request.data.get("logs", "")
    if logs:
        # Append to a JobEvent for auditing
        JobEvent.objects.create(
            job=job, event_type="heartbeat", description=logs[:5000]
        )
    return Response({"status": "ok"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_check_update(request):
    """Check if a newer agent version is available."""
    current_version = request.data.get("version", "0.0.0")
    # In production, compare with a known latest version stored in DB or env
    latest_version = getattr(settings, "AGENT_LATEST_VERSION", "1.0.0")
    if current_version < latest_version:
        return Response(
            {
                "version": latest_version,
                "download_url": f"https://releases.neuralops.com/agent/{latest_version}/neuralops-agent",
                "sha256": "placeholder-hash",  # will be generated by CI
            }
        )
    return Response({"version": current_version, "update_available": False})


@api_view(["GET"])
def agent_latest_version(request):
    """Return the latest agent version info for auto-update."""
    latest = AgentRelease.objects.filter(is_latest=True).first()
    if not latest:
        return Response({"version": "0.0.0", "update_available": False})

    current = request.query_params.get("current", "0.0.0")
    needs_update = _version_newer(latest.version, current)

    platform = request.query_params.get("platform", "linux")
    download_url = ""
    sha256 = ""
    if platform == "macos" or platform == "darwin":
        download_url = latest.macos_dmg_url
        sha256 = latest.macos_sha256
    elif platform == "windows" or platform == "win32":
        download_url = latest.windows_exe_url
        sha256 = latest.windows_sha256
    else:
        download_url = latest.linux_appimage_url
        sha256 = latest.linux_sha256

    return Response(
        {
            "version": latest.version,
            "update_available": needs_update,
            "download_url": download_url,
            "sha256": sha256,
            "release_notes": latest.release_notes,
        }
    )
