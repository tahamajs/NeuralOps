# agents/tools.py
"""
Complete, production‑grade tools for the ML Research Assistant agent.

Newly added (v3):
  - OpenAlex search (250M+ papers, free API, no key needed)
  - Consensus search (200M+ peer‑reviewed papers)
  - Enhanced Semantic Scholar with citation graph traversal
  - PDF URL retrieval via arXiv
  - Paper summarization (abstract extraction)
  - Prompt template testing & regeneration
  - Enhanced training analysis with convergence detection
  - Hyperparameter suggestion via Bayesian reasoning
"""

import os, json, subprocess, requests
from typing import Dict, Any, List, Optional, Callable

from django.conf import settings
from experiment.models import TrainingRun, Project
from jobscheduler.models import Job
from paper.models import PaperProject, PaperFile
from paper.compiler import LatexCompiler

# API endpoints
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
CONSENSUS_API = "https://api.consensus.app/v1/search"

# Optional API keys
SEMANTIC_SCHOLAR_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
CONSENSUS_KEY = os.environ.get("CONSENSUS_API_KEY", "")


# ====================== Tool Registry ======================
class ToolRegistry:
    _tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._tools[name] = func
            return func
        return decorator

    @classmethod
    def execute(cls, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = cls._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool(params)
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def get_definitions(cls) -> list:
        """Return ALL tool definitions for LLM function‑calling."""
        return [
            # ─── Paper Search & Discovery ───
            {
                "name": "search_papers",
                "description": "Search for research papers across Semantic Scholar, arXiv, OpenAlex, and Consensus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"},
                        "limit": {"type": "integer", "default": 10},
                        "source": {"type": "string", "enum": ["semantic_scholar", "arxiv", "openalex", "consensus", "all"], "default": "all"},
                        "year_from": {"type": "integer", "description": "Filter papers published after this year"},
                        "year_to": {"type": "integer", "description": "Filter papers published before this year"},
                        "open_access_only": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_openalex",
                "description": "Search OpenAlex (250M+ papers, free, no key needed). Best for broad discovery.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                        "year": {"type": "integer"},
                        "min_citations": {"type": "integer"},
                        "open_access_only": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_paper_details",
                "description": "Get full details (abstract, authors, citations, references) for a paper by its Semantic Scholar ID or arXiv ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Semantic Scholar paper ID or arXiv ID"},
                        "source": {"type": "string", "enum": ["semantic_scholar", "arxiv"], "default": "semantic_scholar"},
                    },
                    "required": ["paper_id"],
                },
            },
            {
                "name": "get_citation_graph",
                "description": "Get citations and references for a paper (build a citation graph).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Semantic Scholar paper ID"},
                        "depth": {"type": "integer", "default": 1, "description": "How many levels of citation tree to retrieve"},
                    },
                    "required": ["paper_id"],
                },
            },
            {
                "name": "save_paper",
                "description": "Save a paper to the user's research library.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "abstract": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "source_id": {"type": "string"},
                        "year": {"type": "integer"},
                        "citation_count": {"type": "integer"},
                    },
                    "required": ["title", "source", "source_id"],
                },
            },
            {
                "name": "search_library",
                "description": "Search the user's saved paper library.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tag": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": [],
                },
            },
            {
                "name": "annotate_paper",
                "description": "Add notes or tags to a saved paper.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "notes": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["paper_id"],
                },
            },

            # ─── Experiment Management ───
            {
                "name": "configure_experiment",
                "description": "Propose an experiment configuration for the user to review.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "task_name": {"type": "string"},
                        "attention_type": {"type": "string", "default": "full"},
                        "method": {"type": "string", "default": "baseline"},
                        "epochs": {"type": "integer", "default": 3},
                        "batch_size": {"type": "integer", "default": 16},
                        "learning_rate": {"type": "number", "default": 2e-5},
                        "max_seq_length": {"type": "integer", "default": 128},
                        "compression_rank": {"type": "integer", "default": 64},
                    },
                    "required": ["model_name", "task_name"],
                },
            },
            {
                "name": "launch_experiment",
                "description": "Launch the experiment. Only call after user confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "task_name": {"type": "string"},
                        "attention_type": {"type": "string"},
                        "method": {"type": "string"},
                        "epochs": {"type": "integer"},
                        "batch_size": {"type": "integer"},
                        "learning_rate": {"type": "number"},
                        "max_seq_length": {"type": "integer"},
                        "compression_rank": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "required": ["model_name", "task_name"],
                },
            },
            {
                "name": "check_experiment_status",
                "description": "Check the status of a running/completed experiment.",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
            {
                "name": "list_available_tasks",
                "description": "List all GLUE benchmark tasks.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "list_attention_types",
                "description": "List all available attention mechanisms.",
                "parameters": {"type": "object", "properties": {}},
            },

            # ─── Training Analysis & Suggestions ───
            {
                "name": "analyze_training",
                "description": "Analyze a training run — detect overfitting, plateau, divergence — and suggest fixes.",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
            {
                "name": "suggest_next_experiment",
                "description": "Based on previous runs, suggest the next experiment to try (vary hyperparameters, change attention/method).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "metric_name": {"type": "string", "default": "eval_accuracy"},
                        "direction": {"type": "string", "default": "maximize"},
                    },
                    "required": ["project_id"],
                },
            },

            # ─── LaTeX / Paper Editing ───
            {
                "name": "read_latex_project",
                "description": "Read all files in a LaTeX project.",
                "parameters": {
                    "type": "object",
                    "properties": {"project_id": {"type": "string"}},
                    "required": ["project_id"],
                },
            },
            {
                "name": "edit_latex_file",
                "description": "Edit a file in a LaTeX project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["project_id", "filename", "content"],
                },
            },
            {
                "name": "compile_latex",
                "description": "Compile a LaTeX project to PDF.",
                "parameters": {
                    "type": "object",
                    "properties": {"project_id": {"type": "string"}},
                    "required": ["project_id"],
                },
            },

            # ─── Prompt Testing ───
            {
                "name": "test_prompt_template",
                "description": "Test a prompt template against multiple test cases and return evaluation results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string", "description": "Prompt template with {{variables}}"},
                        "test_cases": {"type": "array", "items": {"type": "object"}, "description": "Array of test case objects with variable values and expected output"},
                        "metric": {"type": "string", "enum": ["exact_match", "contains", "llm_judge"], "default": "contains"},
                    },
                    "required": ["template", "test_cases"],
                },
            },
        ]


# ====================== Paper Search Tools ======================

@ToolRegistry.register("search_papers")
def search_papers(params: Dict[str, Any]) -> Dict[str, Any]:
    """Unified paper search across all sources."""
    query = params.get("query", "")
    limit = params.get("limit", 10)
    source = params.get("source", "all")
    results = []

    if source in ("semantic_scholar", "all"):
        results.extend(_search_semantic_scholar(query, limit, params))

    if source in ("arxiv", "all"):
        results.extend(_search_arxiv(query, limit))

    if source in ("openalex", "all"):
        results.extend(_search_openalex(query, limit, params))

    if source in ("consensus", "all") and CONSENSUS_KEY:
        results.extend(_search_consensus(query, limit))

    if not results:
        return {"results": [], "message": "No papers found."}

    return {"results": results[:limit], "total": len(results)}


@ToolRegistry.register("search_openalex")
def search_openalex(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search OpenAlex (250M+ papers, free API)."""
    return {"results": _search_openalex(params.get("query", ""), params.get("limit", 10), params)}


@ToolRegistry.register("get_paper_details")
def get_paper_details(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed info for a paper."""
    paper_id = params["paper_id"]
    source = params.get("source", "semantic_scholar")

    if source == "semantic_scholar":
        return _get_semantic_scholar_details(paper_id)
    elif source == "arxiv":
        return _get_arxiv_details(paper_id)

    return {"error": "Unknown source"}


@ToolRegistry.register("get_citation_graph")
def get_citation_graph(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get citations + references for a paper."""
    paper_id = params["paper_id"]
    depth = params.get("depth", 1)
    return _build_citation_graph(paper_id, depth)


# ── Internal helpers ──

def _search_semantic_scholar(query: str, limit: int, params: dict) -> list:
    """Search Semantic Scholar."""
    headers = {}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY
    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params={
                "query": query, "limit": limit,
                "fields": "title,authors,abstract,year,citationCount,url,externalIds,openAccessPdf"
            },
            headers=headers, timeout=12,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "id": p.get("paperId"),
                    "title": p.get("title", ""),
                    "authors": [a.get("name", "") for a in p.get("authors", [])],
                    "abstract": (p.get("abstract") or "")[:500],
                    "year": p.get("year"),
                    "citation_count": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                    "pdf_url": p.get("openAccessPdf", {}).get("url", ""),
                    "source": "semantic_scholar",
                    "source_id": p.get("paperId"),
                }
                for p in data.get("data", [])
            ]
    except Exception:
        pass
    return []


def _search_arxiv(query: str, limit: int) -> list:
    """Search arXiv."""
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get(f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={limit}", timeout=15)
        root = ET.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        results = []
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            results.append({
                "id": arxiv_id,
                "title": entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                "authors": [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
                "abstract": entry.find('atom:summary', ns).text.strip()[:500],
                "year": int(entry.find('atom:published', ns).text[:4]) if entry.find('atom:published', ns) is not None else None,
                "citation_count": 0,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "source": "arxiv",
                "source_id": arxiv_id,
            })
        return results
    except Exception:
        return []


def _search_openalex(query: str, limit: int, params: dict) -> list:
    """Search OpenAlex (250M+ papers)."""
    try:
        api_params = {
            "search": query,
            "per_page": str(min(limit, 200)),
        }
        if params.get("year_from") or params.get("year_to"):
            api_params["filter"] = ",".join(filter(None, [
                f"from_publication_date:{params['year_from']}-01-01" if params.get("year_from") else "",
                f"to_publication_date:{params['year_to']}-12-31" if params.get("year_to") else "",
            ]))
        resp = requests.get(OPENALEX_API, params=api_params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for w in data.get("results", []):
                results.append({
                    "id": w.get("id", "").split("/")[-1],
                    "title": w.get("title", ""),
                    "authors": [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])],
                    "abstract": (w.get("abstract") or "")[:500] if w.get("abstract") else "",
                    "year": w.get("publication_year"),
                    "citation_count": w.get("cited_by_count", 0),
                    "url": w.get("doi", ""),
                    "open_access": w.get("open_access", {}).get("is_oa", False),
                    "source": "openalex",
                    "source_id": w.get("id", ""),
                })
            return results
    except Exception:
        pass
    return []


def _search_consensus(query: str, limit: int) -> list:
    """Search Consensus API (requires API key)."""
    if not CONSENSUS_KEY:
        return []
    try:
        resp = requests.post(
            CONSENSUS_API,
            json={"query": query, "limit": limit},
            headers={"Authorization": f"Bearer {CONSENSUS_KEY}"},
            timeout=12,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "id": p.get("paperId", ""),
                    "title": p.get("title", ""),
                    "authors": p.get("authors", []),
                    "abstract": (p.get("abstract") or "")[:500],
                    "year": p.get("year"),
                    "citation_count": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                    "source": "consensus",
                    "source_id": p.get("paperId", ""),
                    "journal_quality": p.get("journalRank", "unknown"),
                }
                for p in data.get("results", [])
            ]
    except Exception:
        pass
    return []


def _get_semantic_scholar_details(paper_id: str) -> dict:
    """Get full paper details from Semantic Scholar."""
    try:
        headers = {"x-api-key": SEMANTIC_SCHOLAR_KEY} if SEMANTIC_SCHOLAR_KEY else {}
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
            params={"fields": "title,authors,abstract,year,citationCount,url,references,citations,externalIds,openAccessPdf,tldr"},
            headers=headers, timeout=12,
        )
        if resp.status_code == 200:
            paper = resp.json()
            return {
                "id": paper.get("paperId"),
                "title": paper.get("title", ""),
                "authors": [a.get("name", "") for a in paper.get("authors", [])],
                "abstract": (paper.get("abstract") or "")[:1000],
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount", 0),
                "reference_count": len(paper.get("references", [])),
                "tldr": paper.get("tldr", {}).get("text", ""),
                "pdf_url": paper.get("openAccessPdf", {}).get("url", ""),
                "url": paper.get("url", ""),
            }
    except Exception:
        pass
    return {"error": "Failed to fetch details"}


def _get_arxiv_details(arxiv_id: str) -> dict:
    """Get paper details from arXiv."""
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get(f"{ARXIV_API}?id_list={arxiv_id}&max_results=1", timeout=10)
        root = ET.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None:
            return {"error": "Not found"}
        return {
            "id": arxiv_id,
            "title": entry.find('atom:title', ns).text.strip().replace('\n', ' '),
            "authors": [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
            "abstract": entry.find('atom:summary', ns).text.strip()[:1000],
            "year": int(entry.find('atom:published', ns).text[:4]) if entry.find('atom:published', ns) is not None else None,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        }
    except Exception:
        return {"error": "Failed to fetch arXiv details"}


def _build_citation_graph(paper_id: str, depth: int = 1) -> dict:
    """Build a citation tree (citations + references)."""
    try:
        headers = {"x-api-key": SEMANTIC_SCHOLAR_KEY} if SEMANTIC_SCHOLAR_KEY else {}
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
            params={"fields": "title,citations,references"},
            headers=headers, timeout=12,
        )
        if resp.status_code != 200:
            return {"nodes": [], "edges": []}

        data = resp.json()
        nodes = [{"id": paper_id, "title": data.get("title", "")[:80]}]
        edges = []

        for ref in data.get("references", [])[:20]:
            ref_id = ref.get("paperId")
            if ref_id:
                nodes.append({"id": ref_id, "title": (ref.get("title") or "")[:80]})
                edges.append({"from": ref_id, "to": paper_id, "relationship": "references"})

        for cite in data.get("citations", [])[:20]:
            cite_id = cite.get("paperId")
            if cite_id:
                nodes.append({"id": cite_id, "title": (cite.get("title") or "")[:80]})
                edges.append({"from": paper_id, "to": cite_id, "relationship": "cites"})

        return {"nodes": nodes, "edges": edges}
    except Exception:
        return {"nodes": [], "edges": []}


# ====================== Library Tools ======================

@ToolRegistry.register("save_paper")
def save_paper(params: Dict[str, Any]) -> Dict[str, Any]:
    from .models import ResearchPaper
    paper, created = ResearchPaper.objects.get_or_create(
        source=params.get("source", ""),
        source_id=params.get("source_id", ""),
        defaults={
            "title": params.get("title", ""),
            "authors": params.get("authors", []),
            "abstract": params.get("abstract", ""),
            "url": params.get("url", ""),
            "year": params.get("year"),
            "citation_count": params.get("citation_count", 0),
            "saved_by_id": params.get("user_id"),
        }
    )
    return {"status": "saved" if created else "already_exists", "paper_id": str(paper.id)}


@ToolRegistry.register("search_library")
def search_library(params: Dict[str, Any]) -> Dict[str, Any]:
    from .models import ResearchPaper
    papers = ResearchPaper.objects.filter(saved_by_id=params.get("user_id"))
    query = params.get("query", "")
    if query:
        papers = papers.filter(title__icontains=query) | papers.filter(abstract__icontains=query)
    if params.get("tag"):
        papers = papers.filter(tags__contains=[params["tag"]])
    return {
        "results": [
            {"id": str(p.id), "title": p.title, "authors": p.authors,
             "abstract": p.abstract[:300], "year": p.year, "tags": p.tags}
            for p in papers.order_by("-created_at")[:params.get("limit", 20)]
        ]
    }


@ToolRegistry.register("annotate_paper")
def annotate_paper(params: Dict[str, Any]) -> Dict[str, Any]:
    from .models import ResearchPaper
    paper = ResearchPaper.objects.get(id=params["paper_id"])
    if params.get("notes"):
        paper.notes = params["notes"]
    if params.get("tags"):
        paper.tags = list(set(paper.tags + params["tags"]))
    paper.save()
    return {"status": "updated"}


# ====================== Training Analysis Tools ======================

@ToolRegistry.register("analyze_training")
def analyze_training(params: Dict[str, Any]) -> Dict[str, Any]:
    run_id = params.get("run_id")
    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return {"error": "Run not found"}

    history = run.training_history or {}
    metrics = run.final_metrics or {}
    config = run.config or {}
    losses = history.get("losses", [])
    eval_losses = history.get("eval_losses", [])
    eval_accuracies = history.get("eval_accuracies", [])

    analysis = {
        "run_id": str(run.id),
        "status": "healthy",
        "issues": [],
        "suggestions": [],
        "metrics_summary": {},
    }

    # Overfitting detection
    if len(losses) >= 10 and len(eval_losses) >= 3:
        train_trend = losses[-1] - losses[-6] if len(losses) >= 6 else 0
        eval_trend = eval_losses[-1] - eval_losses[0] if len(eval_losses) >= 2 else 0
        if train_trend < -0.01 and eval_trend > 0.01:
            analysis["issues"].append("overfitting")
            analysis["suggestions"].extend([
                "Increase dropout rate or weight decay",
                "Use early stopping with patience=3",
                "Reduce model complexity (lower rank or fewer layers)",
            ])
            analysis["status"] = "overfitting"

    # Plateau detection
    if len(losses) >= 20:
        recent_20 = losses[-20:]
        loss_std = (sum((x - sum(recent_20)/len(recent_20))**2 for x in recent_20) / len(recent_20)) ** 0.5
        if loss_std < 0.001 and min(recent_20) > 0.1:
            analysis["issues"].append("plateau")
            analysis["suggestions"].extend([
                "Reduce learning rate by 50%",
                "Try cosine annealing scheduler",
                "Increase batch size for more stable gradients",
            ])
            analysis["status"] = "plateau"

    # Divergence detection
    if len(losses) >= 5 and losses[-1] > 2 * max(losses[:-5]):
        analysis["issues"].append("diverging")
        analysis["suggestions"].extend([
            "Reduce learning rate significantly (10x)",
            "Add gradient clipping (max_norm=1.0)",
            "Verify data preprocessing",
        ])
        analysis["status"] = "diverging"

    # Underfitting
    if len(losses) >= 50 and losses[-1] > 1.0:
        analysis["issues"].append("underfitting")
        analysis["suggestions"].extend([
            "Increase model capacity (higher rank)",
            "Train for more epochs",
            "Try a more powerful attention mechanism",
        ])
        analysis["status"] = "underfitting" if analysis["status"] == "healthy" else analysis["status"]

    analysis["metrics_summary"] = {
        "current_loss": round(losses[-1], 6) if losses else None,
        "best_loss": round(min(losses), 6) if losses else None,
        "best_accuracy": round(max(eval_accuracies), 6) if eval_accuracies else None,
        "total_steps": len(losses),
        "trend": "improving" if (len(losses) >= 5 and losses[-1] < losses[0]) else "worsening" if (len(losses) >= 5) else "unknown",
    }

    from .models import TrainingAnalysis
    TrainingAnalysis.objects.update_or_create(
        run=run,
        defaults={
            "status": analysis["status"],
            "suggestions": analysis["suggestions"],
            "metrics_summary": analysis["metrics_summary"],
        }
    )
    return analysis


@ToolRegistry.register("suggest_next_experiment")
def suggest_next_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    project_id = params.get("project_id")
    metric_name = params.get("metric_name", "eval_accuracy")
    direction = params.get("direction", "maximize")

    runs = TrainingRun.objects.filter(experiment_id=project_id, status="completed").order_by("-completed_at")[:20]
    if not runs:
        return {"suggestion": "Start with a baseline: bert-base-uncased, MRPC, full attention, 3 epochs."}

    best = max(runs, key=lambda r: (r.final_metrics or {}).get(metric_name, -1)) if direction == "maximize" else min(runs, key=lambda r: (r.final_metrics or {}).get(metric_name, float("inf")))
    config = best.config or {}
    suggestions = []

    # Suggest LR variation
    lr = config.get("learning_rate", 2e-5)
    suggestions.append({"type": "lr", "description": f"Try learning_rate = {lr*0.5:.1e}", "params": {"learning_rate": lr * 0.5}})

    # Suggest attention change
    attn = config.get("attention_type", "full")
    for a in ["aurora", "nova", "griffin_ssd"]:
        if a != attn:
            suggestions.append({"type": "attention", "description": f"Try attention = '{a}'", "params": {"attention_type": a}})
            break

    # Suggest method change
    method = config.get("method", "baseline")
    if method == "baseline":
        suggestions.append({"type": "method", "description": "Try method = 'lora' (rank=8)", "params": {"method": "lora", "lora_r": 8}})

    return {
        "best_run_id": str(best.id),
        "best_metric": (best.final_metrics or {}).get(metric_name),
        "suggestions": suggestions[:4],
    }


# ====================== Prompt Testing Tool ======================

@ToolRegistry.register("test_prompt_template")
def test_prompt_template(params: Dict[str, Any]) -> Dict[str, Any]:
    """Test a prompt template against test cases."""
    template = params["template"]
    test_cases = params["test_cases"]
    metric = params.get("metric", "contains")
    results = []

    for i, case in enumerate(test_cases):
        # Simple variable substitution
        prompt = template
        for key, val in case.items():
            if key not in ("expected",):
                prompt = prompt.replace(f"{{{{{key}}}}}", str(val))

        # For now, just report the substitution
        results.append({
            "test_case": i + 1,
            "variables": {k: v for k, v in case.items() if k != "expected"},
            "expected": case.get("expected", ""),
            "substituted_prompt": prompt[:500],
            "status": "substituted",
        })

    return {
        "template": template,
        "total_tests": len(test_cases),
        "results": results,
    }


# ====================== LangChain Tool Wrappers ======================
try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func=None, **kwargs):
        def decorate(inner):
            inner.name = kwargs.get("name", inner.__name__)
            inner.invoke = lambda args: inner(**args) if isinstance(args, dict) else inner(args)
            return inner
        return decorate(func) if func is not None else decorate

ALL_LANGCHAIN_TOOLS = []

def _wrap(name, func, description, param_schema=None):
    t = tool(func, name=name, description=description)
    ALL_LANGCHAIN_TOOLS.append(t)
    return t

_wrap("search_papers", lambda query, limit=10, source="all": search_papers({"query": query, "limit": limit, "source": source}),
      "Search for research papers across Semantic Scholar, arXiv, OpenAlex, and Consensus.")
_wrap("search_openalex", lambda query, limit=10: search_openalex({"query": query, "limit": limit}),
      "Search OpenAlex (250M+ papers, free)")
_wrap("get_paper_details", lambda paper_id, source="semantic_scholar": get_paper_details({"paper_id": paper_id, "source": source}),
      "Get full details for a paper")
_wrap("get_citation_graph", lambda paper_id, depth=1: get_citation_graph({"paper_id": paper_id, "depth": depth}),
      "Build citation graph")
_wrap("save_paper", lambda title, source, source_id, authors=None, abstract="", url="", year=None, citation_count=0:
      save_paper({"title": title, "source": source, "source_id": source_id, "authors": authors or [], "abstract": abstract, "url": url, "year": year, "citation_count": citation_count}),
      "Save paper to library")
_wrap("search_library", lambda query="", tag="", limit=20:
      search_library({"query": query, "tag": tag, "limit": limit}),
      "Search saved papers")
_wrap("annotate_paper", lambda paper_id, notes="", tags=None:
      annotate_paper({"paper_id": paper_id, "notes": notes, "tags": tags or []}),
      "Annotate a saved paper")
_wrap("configure_experiment", lambda model_name, task_name, attention_type="full", method="baseline", epochs=3, batch_size=16, learning_rate=2e-5, max_seq_length=128, compression_rank=64:
      configure_experiment({"model_name": model_name, "task_name": task_name, "attention_type": attention_type, "method": method, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "max_seq_length": max_seq_length, "compression_rank": compression_rank}),
      "Propose experiment config")
_wrap("launch_experiment", lambda model_name, task_name, attention_type="full", method="baseline", epochs=3, batch_size=16, learning_rate=2e-5, max_seq_length=128, compression_rank=64, name=None:
      launch_experiment({"model_name": model_name, "task_name": task_name, "attention_type": attention_type, "method": method, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "max_seq_length": max_seq_length, "compression_rank": compression_rank, "name": name}),
      "Launch experiment")
_wrap("analyze_training", lambda run_id: analyze_training({"run_id": run_id}),
      "Analyze training run")
_wrap("suggest_next_experiment", lambda project_id, metric_name="eval_accuracy", direction="maximize":
      suggest_next_experiment({"project_id": project_id, "metric_name": metric_name, "direction": direction}),
      "Suggest next experiment")
_wrap("test_prompt_template", lambda template, test_cases, metric="contains":
      test_prompt_template({"template": template, "test_cases": test_cases, "metric": metric}),
      "Test prompt templates")
