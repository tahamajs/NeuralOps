from datetime import timedelta

from django.contrib.auth.models import User
from django.db.models import Sum
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import ActivityEvent, Endorsement, Portfolio, ShowcaseItem
from .serializers import (PortfolioSerializer, ShowcaseItemSerializer,
                          WidgetEmbedSerializer)


# ═══════════════════════════════════════════════════════════════
# PUBLIC PORTFOLIO VIEW
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def public_portfolio(request, username):
    """View a user's public portfolio by username. No auth required."""
    user = get_object_or_404(User, username=username)
    portfolio, _ = Portfolio.objects.get_or_create(user=user)
    if (
        not portfolio.is_public
        and request.user != user
        and not (request.user and request.user.is_staff)
    ):
        return Response({"error": "This portfolio is private"}, status=403)
    serializer = PortfolioSerializer(portfolio)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# MY PORTFOLIO (AUTHENTICATED)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def my_portfolio(request):
    """View or update the authenticated user's portfolio."""
    portfolio, _ = Portfolio.objects.get_or_create(user=request.user)
    if request.method == "GET":
        serializer = PortfolioSerializer(portfolio)
        return Response(serializer.data)
    elif request.method == "PATCH":
        serializer = PortfolioSerializer(portfolio, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# SHOWCASE ITEMS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_showcase_item(request):
    """Add an item to the authenticated user's portfolio."""
    portfolio, _ = Portfolio.objects.get_or_create(user=request.user)
    serializer = ShowcaseItemSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(portfolio=portfolio)
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_showcase_item(request, item_id):
    """Remove a showcase item (owner only)."""
    item = get_object_or_404(ShowcaseItem, id=item_id, portfolio__user=request.user)
    item.delete()
    return Response(status=204)


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def toggle_pin_item(request, item_id):
    """Toggle pinned status of a showcase item."""
    item = get_object_or_404(ShowcaseItem, id=item_id, portfolio__user=request.user)
    item.is_pinned = not item.is_pinned
    item.save(update_fields=["is_pinned"])
    return Response(ShowcaseItemSerializer(item).data)


# ═══════════════════════════════════════════════════════════════
# ENDORSEMENTS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def endorse_skill(request, username):
    """Endorse a skill on a user's portfolio."""
    target_user = get_object_or_404(User, username=username)
    portfolio, _ = Portfolio.objects.get_or_create(user=target_user)
    if target_user == request.user:
        return Response({"error": "Cannot endorse yourself"}, status=400)
    skill = request.data.get("skill")
    if not skill:
        return Response({"error": "skill required"}, status=400)
    endorsement, created = Endorsement.objects.get_or_create(
        portfolio=portfolio, endorser=request.user, skill=skill
    )
    return Response({"status": "endorsed" if created else "already endorsed"})


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_endorsement(request, username, skill):
    """Remove an endorsement."""
    target_user = get_object_or_404(User, username=username)
    portfolio = get_object_or_404(Portfolio, user=target_user)
    Endorsement.objects.filter(
        portfolio=portfolio, endorser=request.user, skill=skill
    ).delete()
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# ACTIVITY GRAPH (GitHub-style contribution chart)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def activity_graph(request, username):
    """Return daily activity data for the past year."""
    user = get_object_or_404(User, username=username)
    year_ago = timezone.now().date() - timedelta(days=365)
    events = (
        ActivityEvent.objects.filter(user=user, date__gte=year_ago)
        .values("date")
        .annotate(total=Sum("count"))
        .order_by("date")
    )
    return Response([{"date": str(e["date"]), "count": e["total"]} for e in events])


# ═══════════════════════════════════════════════════════════════
# EMBED WIDGET
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def embed_widget(request, username):
    """Return a minimal JSON/HTML snippet for embedding on personal websites."""
    user = get_object_or_404(User, username=username)
    portfolio, _ = Portfolio.objects.get_or_create(user=user)
    if not portfolio.is_public:
        return Response({"error": "Portfolio is private"}, status=403)
    items = portfolio.items.all()[:5]
    html = f'<div style="font-family:sans-serif;max-width:400px;border:1px solid {portfolio.theme_color};border-radius:8px;padding:16px;text-align:center">'
    html += f'<h3 style="color:{portfolio.theme_color}">{user.username}</h3>'
    html += f"<p>{portfolio.headline}</p>"
    for item in items:
        html += f'<div style="margin:8px 0"><a href="{item.url}">{item.title}</a></div>'
    html += "</div>"
    return Response({"html": html})


# ═══════════════════════════════════════════════════════════════
# CREATE WIDGET CONFIG
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_widget(request):
    """Create a custom embed widget configuration."""
    portfolio, _ = Portfolio.objects.get_or_create(user=request.user)
    serializer = WidgetEmbedSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(portfolio=portfolio)
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)
