# agents/paper_recommender.py
"""
Paper recommendation engine using Semantic Scholar Recommendations API.
Given a training run, finds related papers based on the task, model, and approach.
"""

from typing import Dict, List

import requests
from experiment.models import TrainingRun

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org"


def search_papers_for_topic(topic: str, limit: int = 5) -> List[Dict]:
    """Search papers relevant to a topic."""
    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/graph/v1/paper/search",
            params={
                "query": topic,
                "limit": limit,
                "fields": "title,authors,abstract,year,citationCount,url,externalIds",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "paperId": p.get("paperId"),
                    "title": p.get("title", ""),
                    "authors": [a.get("name", "") for a in p.get("authors", [])],
                    "abstract": p.get("abstract", "") or "",
                    "year": p.get("year"),
                    "citationCount": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                }
                for p in data.get("data", [])
            ]
    except Exception:
        pass
    return []


def get_paper_recommendations(positive_ids: List[str], limit: int = 10) -> List[Dict]:
    """
    Get paper recommendations from Semantic Scholar.
    Requires a list of positive example paper IDs.
    """
    try:
        resp = requests.post(
            f"{SEMANTIC_SCHOLAR_API}/recommendations/v1/papers",
            params={
                "limit": limit,
                "fields": "title,authors,abstract,year,citationCount,url",
            },
            json={"positivePaperIds": positive_ids},
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "paperId": p.get("paperId"),
                    "title": p.get("title", ""),
                    "authors": [a.get("name", "") for a in p.get("authors", [])],
                    "abstract": p.get("abstract", "") or "",
                    "year": p.get("year"),
                    "citationCount": p.get("citationCount", 0),
                    "url": p.get(
                        "url",
                        f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                    ),
                }
                for p in data.get("recommendedPapers", [])
            ]
    except Exception:
        pass
    return []


def get_paper_recommendations_for_run(run: TrainingRun, limit: int = 5) -> List[Dict]:
    """Get paper recommendations based on a training run's configuration."""
    config = run.config or {}

    # Build search topics from config
    topics = []
    if config.get("attention_type") and config["attention_type"] != "full":
        topics.append(f"{config['attention_type']} attention mechanism")
    if config.get("method") and config["method"] != "baseline":
        topics.append(f"{config['method']} fine-tuning")
    if config.get("task_name"):
        topics.append(f"{config['task_name']} benchmark")
    if config.get("model_name"):
        topics.append(f"efficient {config['model_name']}")

    all_papers = []
    for topic in topics[:3]:
        papers = search_papers_for_topic(topic, limit=3)
        all_papers.extend(papers)

    # Deduplicate by paperId
    seen = set()
    unique = []
    for p in all_papers:
        pid = p.get("paperId")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)

    return unique[:limit]


def get_recommendations_from_saved_papers(user_id: int, limit: int = 5) -> List[Dict]:
    """
    Get recommendations based on the user's saved paper library.
    Uses saved papers as positive examples for the Recommendations API.
    """
    from agents.models import ResearchPaper

    saved = ResearchPaper.objects.filter(saved_by_id=user_id).order_by("-created_at")[
        :20
    ]
    positive_ids = [
        p.source_id for p in saved if p.source == "semantic_scholar" and p.source_id
    ]

    if not positive_ids:
        return []

    return get_paper_recommendations(positive_ids, limit)
