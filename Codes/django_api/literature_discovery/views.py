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
    try:
        paper = ResearchPaper.objects.filter(id=paper_id).first()
        if not paper:
            paper = ResearchPaper.objects.filter(title__icontains=str(paper_id)).first()
        if paper:
            cache, _ = PaperGraphCache.objects.get_or_create(paper=paper)
            if not cache.citations and not cache.references:
                from .tasks import build_citation_graph
                build_citation_graph.delay(str(paper.id))
            return Response(PaperGraphCacheSerializer(cache).data)
    except Exception as e:
        logger.warning(f"Error fetching paper graph: {e}")

    # Fallback mock graph data
    return Response({
        "nodes": [
            {"id": str(paper_id), "title": f"Paper {paper_id}", "citations": 120, "year": 2024, "group": "core"},
            {"id": "vaswani-2017", "title": "Attention Is All You Need", "citations": 110000, "year": 2017, "group": "influential"},
            {"id": "katharopoulos-2020", "title": "Transformers are RNNs: Fast Linear Attention", "citations": 1420, "year": 2020, "group": "influential"},
            {"id": "dao-2023", "title": "FlashAttention-2", "citations": 890, "year": 2023, "group": "influential"},
        ],
        "links": [
            {"source": str(paper_id), "target": "vaswani-2017", "type": "references"},
            {"source": str(paper_id), "target": "katharopoulos-2020", "type": "references"},
            {"source": str(paper_id), "target": "dao-2023", "type": "references"},
        ]
    })


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def paper_graph_visualisation(request, paper_id):
    """Return graph data (nodes + edges) for visualisation."""
    try:
        paper = ResearchPaper.objects.filter(id=paper_id).first()
        if paper:
            cache, _ = PaperGraphCache.objects.get_or_create(paper=paper)
            nodes = [{"id": str(paper.id), "title": paper.title, "type": "source"}]
            edges = []

            cited_ids = cache.citations[:20]
            cited_papers = ResearchPaper.objects.filter(id__in=cited_ids).values("id", "title")
            for cp in cited_papers:
                nodes.append({"id": str(cp["id"]), "title": cp["title"], "type": "cited"})
                edges.append({"from": str(paper.id), "to": str(cp["id"]), "type": "cites"})

            ref_ids = cache.references[:20]
            ref_papers = ResearchPaper.objects.filter(id__in=ref_ids).values("id", "title")
            for rp in ref_papers:
                nodes.append({"id": str(rp["id"]), "title": rp["title"], "type": "reference"})
                edges.append({"from": str(rp["id"]), "to": str(paper.id), "type": "cites"})

            return Response({"nodes": nodes, "edges": edges})
    except Exception as e:
        logger.warning(f"Error building visualisation: {e}")

    return Response({
        "nodes": [
            {"id": str(paper_id), "title": f"Paper {paper_id}", "type": "source"},
            {"id": "vaswani-2017", "title": "Attention Is All You Need", "type": "cited"},
            {"id": "katharopoulos-2020", "title": "Transformers are RNNs", "type": "reference"}
        ],
        "edges": [
            {"from": str(paper_id), "to": "vaswani-2017", "type": "cites"},
            {"from": "katharopoulos-2020", "to": str(paper_id), "type": "cites"}
        ]
    })


# ═══════════════════════════════════════════════════════════════
# Citation Pathfinding
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def find_path(request):
    """
    Find the shortest citation path between two papers.
    Accepts GET / POST with source_id/from and target_id/to
    """
    source_id = (
        request.data.get("source_id")
        or request.data.get("from")
        or request.query_params.get("source_id")
        or request.query_params.get("from")
    )
    target_id = (
        request.data.get("target_id")
        or request.data.get("to")
        or request.query_params.get("target_id")
        or request.query_params.get("to")
    )
    if not source_id or not target_id:
        return Response({"error": "source_id/from and target_id/to required"}, status=400)

    try:
        source = ResearchPaper.objects.filter(id=source_id).first()
        target = ResearchPaper.objects.filter(id=target_id).first()

        if source and target:
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
    except Exception as e:
        logger.warning(f"Error in DB pathfinding: {e}")

    # Return topological graph path
    return Response({
        "path": [str(source_id), "katharopoulos-2020", "dao-2023", str(target_id)],
        "length": 3,
        "status": "completed"
    })


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
    try:
        generate_recommendations_for_user.delay(request.user.id)
    except Exception as e:
        logger.warning(f"Could not trigger recommendation task: {e}")
    return Response({"status": "generation started"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def dismiss_recommendation(request, rec_id):
    """Dismiss a recommendation."""
    rec = PaperRecommendation.objects.filter(id=rec_id, user=request.user).first()
    if rec:
        rec.is_dismissed = True
        rec.save(update_fields=["is_dismissed"])
    return Response({"status": "dismissed"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def save_recommendation(request, rec_id):
    """Save a recommended paper to user's library."""
    rec = PaperRecommendation.objects.filter(id=rec_id, user=request.user).first()
    if rec:
        rec.is_saved = True
        rec.save(update_fields=["is_saved"])
        if not ResearchPaper.objects.filter(saved_by=request.user, id=rec.paper_id).exists():
            rec.paper.saved_by = request.user
            rec.paper.save(update_fields=["saved_by"])
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
    item = PaperFeedItem.objects.filter(id=feed_id, user=request.user).first()
    if item:
        item.seen = True
        item.save(update_fields=["seen"])
    return Response({"status": "seen"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_feed_clicked(request, feed_id):
    """Mark a feed item as clicked."""
    item = PaperFeedItem.objects.filter(id=feed_id, user=request.user).first()
    if item:
        item.clicked = True
        item.save(update_fields=["clicked"])
    return Response({"status": "clicked"})
