import logging

from celery import shared_task
from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)


@shared_task
def build_citation_graph(paper_id):
    """Build the citation graph cache for a paper."""
    from agents.models import ResearchPaper

    from .models import PaperGraphCache

    paper = ResearchPaper.objects.get(id=paper_id)
    # Fetch from Semantic Scholar API or use existing relations
    try:
        from semanticscholar import SemanticScholar

        sch = SemanticScholar(timeout=30)
        details = sch.get_paper(
            paper.source_id or paper.title, fields=["citations", "references"]
        )
        citations = [
            c.get("paperId") for c in (details.citations or []) if c.get("paperId")
        ]
        references = [
            r.get("paperId") for r in (details.references or []) if r.get("paperId")
        ]
    except Exception:
        # Fallback: use locally stored citation edges
        citations = list(
            paper.citations_out.values_list("target_paper__source_id", flat=True)
        )
        references = list(
            paper.citations_in.values_list("source_paper__source_id", flat=True)
        )

    cache, _ = PaperGraphCache.objects.get_or_create(paper=paper)
    cache.citations = citations
    cache.references = references
    cache.save()
    return f"Built citation graph for {paper.title}"


@shared_task
def find_citation_path(source_id, target_id, user_id):
    """
    Find the shortest citation path between two papers using BFS.
    """
    from agents.models import ResearchPaper

    from .models import CitationPath, PaperGraphCache

    source = ResearchPaper.objects.get(id=source_id)
    target = ResearchPaper.objects.get(id=target_id)
    user = User.objects.get(id=user_id)

    # Build graph (simplified BFS over cached citation data)
    visited = {str(source.id)}
    queue = [[str(source.id)]]

    while queue:
        path = queue.pop(0)
        current_id = path[-1]

        if current_id == str(target.id):
            # Path found
            path_papers = ResearchPaper.objects.filter(id__in=path)
            path_obj = CitationPath.objects.create(
                source_paper=source,
                target_paper=target,
                path=path,
                length=len(path) - 1,
                created_by=user,
            )
            return f"Path found: {len(path)-1} steps"

        # Get citations of current paper
        try:
            cache = PaperGraphCache.objects.get(paper_id=current_id)
            neighbors = cache.citations + cache.references
        except PaperGraphCache.DoesNotExist:
            neighbors = []

        for neighbor_id in neighbors:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                new_path = path + [neighbor_id]
                queue.append(new_path)
                if len(new_path) > 6:  # Max depth
                    break

    return "No path found within 6 steps"


@shared_task
def generate_recommendations_for_user(user_id):
    """
    Generate paper recommendations for a user based on their saved papers.
    Uses citation relationships and semantic similarity.
    """
    user = User.objects.get(id=user_id)
    from agents.models import ResearchPaper

    from .models import PaperRecommendation

    saved_papers = ResearchPaper.objects.filter(saved_by=user)
    if not saved_papers.exists():
        return "No saved papers to base recommendations on"

    # Get all papers cited by the user's saved papers
    all_cited = set()
    for paper in saved_papers:
        try:
            cache = PaperGraphCache.objects.get(paper=paper)
            all_cited.update(cache.citations)
        except PaperGraphCache.DoesNotExist:
            pass

    # Also get co‑cited papers
    co_cited = set()
    for paper in saved_papers:
        try:
            cache = PaperGraphCache.objects.get(paper=paper)
            co_cited.update(cache.co_cited)
        except PaperGraphCache.DoesNotExist:
            pass

    # Combine and rank
    recommended_ids = all_cited | co_cited
    created = 0

    for paper_id in list(recommended_ids)[:100]:
        try:
            paper = ResearchPaper.objects.get(source_id=paper_id)
        except ResearchPaper.DoesNotExist:
            continue

        # Score based on citation count and how many of user's papers cite it
        citing_count = (
            ResearchPaper.objects.filter(
                id__in=saved_papers.values_list("id", flat=True)
            )
            .filter(citations_out__target_paper=paper)
            .count()
        )

        score = min(1.0, (paper.citation_count or 0) / 1000.0 + citing_count * 0.1)

        rec, is_new = PaperRecommendation.objects.update_or_create(
            user=user,
            paper=paper,
            source_type="citation",
            defaults={
                "score": round(score, 2),
                "reason": (
                    f"Cited by {citing_count} of your saved papers"
                    if citing_count
                    else "Highly cited paper in your field"
                ),
            },
        )
        if is_new:
            created += 1

    return f"Generated {created} new recommendations"


@shared_task
def generate_landscape_task(landscape_id, seed_paper_ids):
    """
    Generate a research landscape graph from seed papers.
    Builds a network of connected papers with clustering.
    """
    from agents.models import ResearchPaper

    from .models import PaperGraphCache, ResearchLandscape

    landscape = ResearchLandscape.objects.get(id=landscape_id)
    seed_papers = ResearchPaper.objects.filter(id__in=seed_paper_ids)

    nodes = []
    edges = []
    visited = set()

    # BFS from seed papers, up to depth 3
    queue = list(seed_papers)
    depth = {str(p.id): 0 for p in queue}

    while queue:
        current = queue.pop(0)
        current_id = str(current.id)
        if current_id in visited:
            continue
        visited.add(current_id)

        nodes.append(
            {
                "id": current_id,
                "title": current.title,
                "year": current.year,
                "citations": current.citation_count,
                "depth": depth[current_id],
            }
        )

        if depth[current_id] >= 3:
            continue

        try:
            cache = PaperGraphCache.objects.get(paper=current)
            for neighbor_id in cache.citations + cache.references:
                if neighbor_id not in visited:
                    try:
                        neighbor = ResearchPaper.objects.get(source_id=neighbor_id)
                        queue.append(neighbor)
                        depth[str(neighbor.id)] = depth[current_id] + 1
                    except ResearchPaper.DoesNotExist:
                        pass
                edges.append(
                    {
                        "from": current_id,
                        "to": neighbor_id,
                    }
                )
        except PaperGraphCache.DoesNotExist:
            pass

    landscape.graph_data = {"nodes": nodes, "edges": edges}
    landscape.save()
    return f"Landscape generated with {len(nodes)} nodes and {len(edges)} edges"
