import logging

from agents.models import ResearchPaper
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (
    CitationPath,
    PaperFeedItem,
    PaperGraphCache,
    PaperRecommendation,
    ResearchLandscape,
)
from .serializers import *
from .tasks import generate_recommendations_for_user

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Citation Graph
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def paper_graph(request, paper_id):
    """Get the citation graph data for a paper."""
    paper = get_object_or_404(ResearchPaper, id=paper_id)
    cache, _ = PaperGraphCache.objects.get_or_create(paper=paper)
    # If cache is empty, trigger async build
    if not cache.citations and not cache.references:
        from .tasks import build_citation_graph

        build_citation_graph.delay(str(paper.id))
    return Response(PaperGraphCacheSerializer(cache).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def paper_graph_visualisation(request, paper_id):
    """Return graph data (nodes + edges) for visualisation."""
    paper = get_object_or_404(ResearchPaper, id=paper_id)
    cache, _ = PaperGraphCache.objects.get_or_create(paper=paper)

    nodes = [{"id": str(paper.id), "title": paper.title, "type": "source"}]
    edges = []

    # Add citations
    cited_ids = cache.citations[:20]
    cited_papers = ResearchPaper.objects.filter(id__in=cited_ids).values("id", "title")
    for cp in cited_papers:
        nodes.append({"id": str(cp["id"]), "title": cp["title"], "type": "cited"})
        edges.append({"from": str(paper.id), "to": str(cp["id"]), "type": "cites"})

    # Add references
    ref_ids = cache.references[:20]
    ref_papers = ResearchPaper.objects.filter(id__in=ref_ids).values("id", "title")
    for rp in ref_papers:
        nodes.append({"id": str(rp["id"]), "title": rp["title"], "type": "reference"})
        edges.append({"from": str(rp["id"]), "to": str(paper.id), "type": "cites"})

    return Response({"nodes": nodes, "edges": edges})


# ═══════════════════════════════════════════════════════════════
# Citation Pathfinding
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def find_path(request):
    """
    Find the shortest citation path between two papers.
    Body: { "source_id": "...", "target_id": "..." }
    """
    source_id = request.data.get("source_id")
    target_id = request.data.get("target_id")
    if not source_id or not target_id:
        return Response({"error": "source_id and target_id required"}, status=400)

    source = get_object_or_404(ResearchPaper, id=source_id)
    target = get_object_or_404(ResearchPaper, id=target_id)

    # Check if path already cached
    existing = (
        CitationPath.objects.filter(source_paper=source, target_paper=target)
        .order_by("-created_at")
        .first()
    )
    if existing:
        return Response(CitationPathSerializer(existing).data)

    # Trigger async pathfinding
    from .tasks import find_citation_path as find_path_task

    find_path_task.delay(str(source.id), str(target.id), request.user.id)
    return Response({"status": "pathfinding started"}, status=202)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def saved_paths(request):
    """Get saved citation paths for the current user."""
    paths = CitationPath.objects.filter(created_by=request.user).order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(paths, request)
    return paginator.get_paginated_response(
        CitationPathSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# Recommendations (Spotify-style)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def recommendations(request):
    """Get paper recommendations for the current user."""
    qs = PaperRecommendation.objects.filter(
        user=request.user, is_dismissed=False
    ).order_by("-score")

    source_type = request.query_params.get("source_type")
    if source_type:
        qs = qs.filter(source_type=source_type)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        PaperRecommendationSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_recommendations(request):
    """Trigger recommendation generation for the current user."""
    generate_recommendations_for_user.delay(request.user.id)
    return Response({"status": "generation started"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, rec_id):
    """Dismiss a recommendation."""
    rec = get_object_or_404(PaperRecommendation, id=rec_id, user=request.user)
    rec.is_dismissed = True
    rec.save(update_fields=["is_dismissed"])
    return Response({"status": "dismissed"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def save_recommendation(request, rec_id):
    """Save a recommended paper to user's library."""
    rec = get_object_or_404(PaperRecommendation, id=rec_id, user=request.user)
    rec.is_saved = True
    rec.save(update_fields=["is_saved"])
    # Actually save the paper
    if not ResearchPaper.objects.filter(
        saved_by=request.user, id=rec.paper_id
    ).exists():
        rec.paper.saved_by.add(request.user)
    return Response({"status": "saved"})


# ═══════════════════════════════════════════════════════════════
# Research Landscape (Cluster/Map)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def landscape_list(request):
    """List or create research landscapes."""
    if request.method == "GET":
        qs = ResearchLandscape.objects.filter(user=request.user).order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ResearchLandscapeSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ResearchLandscapeSerializer(data=request.data)
        if serializer.is_valid():
            landscape = serializer.save(user=request.user)
            return Response(ResearchLandscapeSerializer(landscape).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def landscape_detail(request, landscape_id):
    """Get or delete a research landscape."""
    landscape = get_object_or_404(ResearchLandscape, id=landscape_id, user=request.user)
    if request.method == "GET":
        return Response(ResearchLandscapeSerializer(landscape).data)
    elif request.method == "DELETE":
        landscape.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_landscape(request):
    """
    Generate a research landscape from seed papers.
    Body: { "name": "...", "seed_paper_ids": [...] }
    """
    name = request.data.get("name", "Untitled Landscape")
    seed_ids = request.data.get("seed_paper_ids", [])

    if not seed_ids:
        return Response({"error": "seed_paper_ids required"}, status=400)

    from .tasks import generate_landscape_task

    landscape = ResearchLandscape.objects.create(
        user=request.user, name=name, seed_papers=seed_ids
    )
    generate_landscape_task.delay(str(landscape.id), seed_ids)
    return Response(ResearchLandscapeSerializer(landscape).data, status=201)


# ═══════════════════════════════════════════════════════════════
# Continuous Paper Feed
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def paper_feed(request):
    """Get the continuous paper feed for the current user."""
    qs = PaperFeedItem.objects.filter(user=request.user, seen=False).order_by(
        "-created_at"
    )

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        PaperFeedItemSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_feed_seen(request, feed_id):
    """Mark a feed item as seen."""
    item = get_object_or_404(PaperFeedItem, id=feed_id, user=request.user)
    item.seen = True
    item.save(update_fields=["seen"])
    return Response({"status": "seen"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_feed_clicked(request, feed_id):
    """Mark a feed item as clicked."""
    item = get_object_or_404(PaperFeedItem, id=feed_id, user=request.user)
    item.clicked = True
    item.save(update_fields=["clicked"])
    return Response({"status": "clicked"})
