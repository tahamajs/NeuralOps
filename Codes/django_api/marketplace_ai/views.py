import logging

from django.shortcuts import get_object_or_404
from gpu_marketplace.models import GPUListing
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .agent import check_compliance, generate_explanation, parse_query
from .models import (
    ComparisonSet,
    ComplianceCheck,
    MarketplaceQuery,
    SmartRecommendation,
)
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Agent Mode — Natural Language Query
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_query(request):
    """
    Accept a natural language query, parse it into filters,
    search the marketplace, and return matching listings with explanations.
    """
    query_text = request.data.get("query", "").strip()
    if not query_text:
        return Response({"error": "query is required"}, status=400)

    # Parse the query
    filters = parse_query(query_text)

    # Record the query
    mq = MarketplaceQuery.objects.create(
        user=request.user,
        query_text=query_text,
        parsed_filters=filters,
        status="processing",
    )

    # Search listings
    qs = GPUListing.objects.filter(status="online")

    if "gpu_model" in filters:
        qs = qs.filter(gpu_model__icontains=filters["gpu_model"])
    if "min_vram" in filters:
        qs = qs.filter(vram_gb__gte=filters["min_vram"])
    if "max_price" in filters:
        qs = qs.filter(current_price_per_hour__lte=filters["max_price"])
    if "listing_type" in filters:
        qs = qs.filter(listing_type=filters["listing_type"])

    sort_map = {
        "price_asc": "current_price_per_hour",
        "price_desc": "-current_price_per_hour",
        "reliability": "-reliability_score",
        "dlperf": "-dlperf_score",
    }
    sort = filters.get("sort", "reliability")
    qs = qs.order_by(sort_map.get(sort, "-reliability_score"))

    results = []
    for listing in qs[:20]:
        results.append(
            {
                "listing_id": str(listing.id),
                "gpu_model": listing.gpu_model,
                "vram_gb": listing.vram_gb,
                "price_per_hour": str(listing.current_price_per_hour),
                "reliability": listing.reliability_score,
                "location": listing.location or "Unknown",
                "listing_type": listing.listing_type,
                "explanation": generate_explanation(filters, listing),
            }
        )

    mq.result_count = len(results)
    mq.status = "completed"
    mq.save(update_fields=["result_count", "status"])

    return Response(
        {
            "query_id": str(mq.id),
            "filters": filters,
            "results": results,
            "count": len(results),
        }
    )


# ═══════════════════════════════════════════════════════════════
# Comparison Sets
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def comparison_list(request):
    """List my comparison sets, or create a new one."""
    if request.method == "GET":
        qs = ComparisonSet.objects.filter(user=request.user).order_by("-created_at")
        return Response(ComparisonSetSerializer(qs, many=True).data)

    elif request.method == "POST":
        name = request.data.get("name", "")
        listing_ids = request.data.get("listing_ids", [])
        comp = ComparisonSet.objects.create(user=request.user, name=name)
        comp.listings.set(listing_ids)
        return Response(ComparisonSetSerializer(comp).data, status=201)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def comparison_detail(request, comparison_id):
    """Get or delete a comparison set."""
    comp = get_object_or_404(ComparisonSet, id=comparison_id, user=request.user)

    if request.method == "GET":
        listings = comp.listings.all()
        data = ComparisonSetSerializer(comp).data
        data["listings_detail"] = [
            {
                "id": str(l.id),
                "gpu_model": l.gpu_model,
                "vram_gb": l.vram_gb,
                "price_per_hour": str(l.current_price_per_hour),
                "reliability": l.reliability_score,
                "dlperf_score": l.dlperf_score,
                "uptime": l.uptime_percentage,
                "location": l.location or "Unknown",
                "listing_type": l.listing_type,
            }
            for l in listings
        ]
        return Response(data)

    elif request.method == "DELETE":
        comp.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Compliance Checking
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def check_compliance_view(request, listing_id):
    """Check if a GPU listing complies with the user's requirements."""
    listing = get_object_or_404(GPUListing, id=listing_id)
    requirements = request.data.get("requirements", {})

    result = check_compliance(listing, requirements)

    check = ComplianceCheck.objects.create(
        user=request.user,
        listing=listing,
        requirements=requirements,
        passed=result["passed"],
        failures=result["failures"],
        warnings=result["warnings"],
    )

    return Response(ComplianceCheckSerializer(check).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def compliance_history(request):
    """Get compliance check history for the current user."""
    qs = ComplianceCheck.objects.filter(user=request.user).order_by("-created_at")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ComplianceCheckSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Smart Recommendations
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def smart_recommendations(request):
    """Get AI‑generated recommendations for the current user."""
    qs = SmartRecommendation.objects.filter(
        user=request.user, is_dismissed=False
    ).order_by("-score")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        SmartRecommendationSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, recommendation_id):
    """Dismiss a recommendation."""
    rec = get_object_or_404(
        SmartRecommendation, id=recommendation_id, user=request.user
    )
    rec.is_dismissed = True
    rec.save(update_fields=["is_dismissed"])
    return Response({"status": "dismissed"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_recommendations(request):
    """
    Generate AI recommendations for the current user based on their usage history.
    """
    # Get the user's past orders to understand their preferences
    from gpu_marketplace.models import ComputeOrder

    orders = ComputeOrder.objects.filter(user=request.user, status="filled").order_by(
        "-created_at"
    )[:10]

    # Determine preferred GPU models, VRAM, and price range
    preferred_models = set()
    max_price = 0
    min_vram = 0
    for order in orders:
        if order.gpu_model_preference:
            preferred_models.add(order.gpu_model_preference)
        if order.max_price_per_hour and float(order.max_price_per_hour) > max_price:
            max_price = float(order.max_price_per_hour)
        if order.min_vram_gb and order.min_vram_gb > min_vram:
            min_vram = order.min_vram_gb

    # Search for similar listings
    qs = GPUListing.objects.filter(status="online")
    if preferred_models:
        qs = qs.filter(gpu_model__in=preferred_models)
    if min_vram:
        qs = qs.filter(vram_gb__gte=min_vram)
    if max_price:
        qs = qs.filter(current_price_per_hour__lte=max_price * 1.2)

    qs = qs.order_by("-reliability_score")[:20]

    created = 0
    for listing in qs:
        # Simple scoring: reliability + DLPerf + price fit
        score = (listing.reliability_score / 100.0) * 0.4
        if listing.dlperf_score:
            score += (listing.dlperf_score / 100.0) * 0.3
        if max_price > 0:
            score += (
                1 - min(float(listing.current_price_per_hour) / max_price, 1.0)
            ) * 0.3
        score = round(score, 2)

        rec, is_new = SmartRecommendation.objects.get_or_create(
            user=request.user,
            listing=listing,
            defaults={
                "score": score,
                "reason": (
                    "Based on your past orders and preferences"
                    if orders
                    else "Recommended for your profile"
                ),
            },
        )
        if is_new:
            created += 1

    return Response(
        {"message": f"Generated {created} new recommendations.", "count": created}
    )
