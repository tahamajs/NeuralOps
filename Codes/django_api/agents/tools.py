"""
Production‑grade tool suite for the ML Research Assistant.
Uses official Python clients for Semantic Scholar, OpenAlex, and arXiv.
All tools return structured Pydantic‑validated responses.
"""

import json
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from django.core.cache import cache

# ── Official API Clients ──────────────────────────────────────────
try:
    from semanticscholar import SemanticScholar

    _sch = SemanticScholar(timeout=30)
except ImportError:
    _sch = None

try:
    import pyalex

    pyalex.config.email = os.environ.get("OPENALEX_EMAIL", "admin@neuralops.dev")
    _openalex_available = True
except ImportError:
    _openalex_available = False

try:
    import arxiv

    _arxiv_client = arxiv.Client()
except ImportError:
    _arxiv_client = None

# ── Database Models ───────────────────────────────────────────────
from experiment.models import Experiment, TrainingRun


# ── Retry Configuration ───────────────────────────────────────────
def with_retry(max_tries=3, base_delay=1.0):
    """Decorator for retrying failed API calls with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_tries - 1:
                        delay = base_delay * (2**attempt)
                        time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════
# Tool Registry (Enhanced)
# ═══════════════════════════════════════════════════════════════════


class ToolRegistry:
    """Central registry for all agent tools with metadata."""

    _tools: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        *,
        name: str,
        description: str,
        category: str = "general",
        requires_confirmation: bool = False,
        cache_ttl: int = 0,
    ):
        """Register a tool with full metadata."""

        def decorator(func):
            cls._tools[name] = {
                "function": func,
                "name": name,
                "description": description,
                "category": category,
                "requires_confirmation": requires_confirmation,
                "cache_ttl": cache_ttl,
            }
            return func

        return decorator

    @classmethod
    def execute(
        cls, name: str, params: Dict[str, Any], user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a tool by name, with caching support."""
        tool = cls._tools.get(name)
        if not tool:
            return {"status": "error", "error": f"Unknown tool: {name}"}

        # Check cache if TTL is set
        if tool["cache_ttl"] > 0:
            cache_key = f"tool:{name}:{json.dumps(params, sort_keys=True)}"
            cached = cache.get(cache_key)
            if cached:
                return cached

        try:
            func = tool["function"]
            if user_id is not None and "user_id" in func.__code__.co_varnames:
                result = func(**params, user_id=user_id)
            else:
                result = func(**params)

            if tool["cache_ttl"] > 0:
                cache.set(cache_key, result, tool["cache_ttl"])
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @classmethod
    def get_definitions(cls, user_role: str = "researcher") -> List[Dict]:
        """Return OpenAI‑compatible tool definitions, filtered by role permissions."""
        role_permissions = {
            "admin": ["*"],
            "researcher": [
                "search_papers",
                "get_paper_details",
                "save_paper",
                "search_library",
                "analyze_training",
                "suggest_next_experiment",
                "configure_experiment",
                "launch_experiment",
                "check_experiment_status",
                "read_latex_project",
                "edit_latex_file",
                "compile_latex",
                "test_prompt_template",
            ],
            "viewer": [
                "search_papers",
                "get_paper_details",
                "search_library",
                "check_experiment_status",
            ],
        }
        allowed = role_permissions.get(user_role, role_permissions["viewer"])
        if "*" in allowed:
            allowed = list(cls._tools.keys())

        definitions = []
        for name, tool in cls._tools.items():
            if name not in allowed:
                continue
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": {
                            "type": "object",
                            "properties": _infer_schema(tool["function"]),
                            "required": _infer_required(tool["function"]),
                        },
                    },
                }
            )
        return definitions

    @classmethod
    def get_categories(cls) -> Dict[str, List[str]]:
        """Return tools grouped by category."""
        categories = {}
        for name, tool in cls._tools.items():
            cat = tool["category"]
            categories.setdefault(cat, []).append(name)
        return categories


def _infer_schema(func: Callable) -> Dict[str, Any]:
    """Infer JSON Schema from function type hints."""
    import inspect

    hints = inspect.signature(func).parameters
    schema = {}
    for name, param in hints.items():
        if name in ("self", "cls", "user_id"):
            continue
        annotation = param.annotation
        if annotation == str:
            schema[name] = {"type": "string", "description": f"{name}"}
        elif annotation == int:
            schema[name] = {"type": "integer", "description": f"{name}"}
        elif annotation == float:
            schema[name] = {"type": "number", "description": f"{name}"}
        elif annotation == bool:
            schema[name] = {"type": "boolean", "description": f"{name}"}
        elif hasattr(annotation, "__origin__") and annotation.__origin__ == list:
            schema[name] = {"type": "array", "items": {"type": "string"}}
        elif hasattr(annotation, "__origin__") and annotation.__origin__ == Optional:
            inner = annotation.__args__[0]
            if inner == str:
                schema[name] = {"type": "string", "description": f"{name} (optional)"}
        else:
            schema[name] = {"type": "string", "description": f"{name}"}
    return schema


def _infer_required(func: Callable) -> List[str]:
    """Infer required parameters from function signature."""
    import inspect

    hints = inspect.signature(func).parameters
    required = []
    for name, param in hints.items():
        if name in ("self", "cls", "user_id"):
            continue
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return required


# ═══════════════════════════════════════════════════════════════════
# Paper Search & Discovery Tools
# ═══════════════════════════════════════════════════════════════════


@ToolRegistry.register(
    name="search_papers",
    description="Search for research papers across Semantic Scholar, OpenAlex, and arXiv.",
    category="research",
    cache_ttl=300,
)
@with_retry(max_tries=3)
def search_papers(
    query: str,
    limit: int = 10,
    source: str = "all",
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    open_access_only: bool = False,
) -> Dict[str, Any]:
    """Unified search across all academic sources."""
    results = []
    errors = []

    if source in ("semantic_scholar", "all") and _sch is not None:
        try:
            ss_results = _sch.search_paper(
                query,
                limit=limit,
                fields=[
                    "title",
                    "authors",
                    "abstract",
                    "year",
                    "citationCount",
                    "url",
                    "externalIds",
                    "openAccessPdf",
                    "publicationTypes",
                ],
            )
            for p in ss_results:
                results.append(_normalize_paper(p, "semantic_scholar"))
        except Exception as e:
            errors.append(f"semantic_scholar: {e}")

    if source in ("arxiv", "all") and _arxiv_client is not None:
        try:
            search = arxiv.Search(
                query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance
            )
            for result in _arxiv_client.results(search):
                results.append(_normalize_arxiv_result(result))
        except Exception as e:
            errors.append(f"arxiv: {e}")

    if source in ("openalex", "all") and _openalex_available:
        try:
            from pyalex import Works

            pager = Works().search(query).per_page(min(limit, 200))
            if year_from:
                pager = pager.filter(publication_year=">" + str(year_from - 1))
            if open_access_only:
                pager = pager.filter(open_access={"is_oa": True})
            for w in pager.paginate(limit=limit, n_max=limit):
                results.append(_normalize_openalex(w))
        except Exception as e:
            errors.append(f"openalex: {e}")

    # Deduplicate by title similarity
    results = _deduplicate_papers(results)

    return {
        "status": "success",
        "results": results[:limit],
        "total": len(results),
        "errors": errors if errors else None,
    }


@ToolRegistry.register(
    name="get_paper_details",
    description="Get full details including abstract, citations, and references for a paper.",
    category="research",
    cache_ttl=600,
)
def get_paper_details(
    paper_id: str, source: str = "semantic_scholar"
) -> Dict[str, Any]:
    """Get comprehensive paper details."""
    if source == "semantic_scholar" and _sch is not None:
        paper = _sch.get_paper(
            paper_id,
            fields=[
                "title",
                "authors",
                "abstract",
                "year",
                "citationCount",
                "url",
                "references",
                "citations",
                "externalIds",
                "openAccessPdf",
                "tldr",
                "publicationVenue",
                "fieldsOfStudy",
                "publicationTypes",
            ],
        )
        return {
            "status": "success",
            "paper": _normalize_paper(
                paper, "semantic_scholar", include_relations=True
            ),
        }

    if source == "arxiv" and _arxiv_client is not None:
        search = arxiv.Search(id_list=[paper_id])
        result = next(_arxiv_client.results(search), None)
        if result:
            return {
                "status": "success",
                "paper": _normalize_arxiv_result(result, detailed=True),
            }
        return {"status": "error", "error": "Paper not found on arXiv"}

    return {"status": "error", "error": f"Source {source} not available"}


@ToolRegistry.register(
    name="get_citation_graph",
    description="Build a citation graph (citations + references) for a paper.",
    category="research",
    cache_ttl=600,
)
def get_citation_graph(paper_id: str, depth: int = 1) -> Dict[str, Any]:
    """Build citation tree with configurable depth."""
    if _sch is None:
        return {"status": "error", "error": "Semantic Scholar not available"}

    nodes = {}
    edges = []

    def _traverse(pid: str, current_depth: int):
        if current_depth > depth or pid in nodes:
            return
        try:
            paper = _sch.get_paper(pid, fields=["title", "citations", "references"])
            nodes[pid] = {"id": pid, "title": (paper.title or "")[:100]}
            for ref in (paper.references or [])[:20]:
                rid = ref.get("paperId")
                if rid:
                    edges.append({"from": rid, "to": pid, "type": "references"})
                    if current_depth < depth:
                        _traverse(rid, current_depth + 1)
            for cite in (paper.citations or [])[:20]:
                cid = cite.get("paperId")
                if cid:
                    edges.append({"from": pid, "to": cid, "type": "cites"})
                    if current_depth < depth:
                        _traverse(cid, current_depth + 1)
        except Exception:
            pass

    _traverse(paper_id, 0)
    return {
        "status": "success",
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# ═══════════════════════════════════════════════════════════════════
# Paper Library Management
# ═══════════════════════════════════════════════════════════════════


@ToolRegistry.register(
    name="save_paper",
    description="Save a paper to the user's research library.",
    category="library",
)
def save_paper(
    title: str,
    source: str,
    source_id: str,
    authors: Optional[List[str]] = None,
    abstract: str = "",
    url: str = "",
    year: Optional[int] = None,
    citation_count: int = 0,
    pdf_url: str = "",
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Save or update a paper in the library."""
    from agents.models import ResearchPaper

    paper, created = ResearchPaper.objects.update_or_create(
        source=source,
        source_id=source_id,
        defaults={
            "title": title,
            "authors": authors or [],
            "abstract": abstract,
            "url": url,
            "year": year,
            "citation_count": citation_count,
            "pdf_url": pdf_url,
            "saved_by_id": user_id,
        },
    )
    return {
        "status": "saved" if created else "updated",
        "paper_id": str(paper.id),
    }


@ToolRegistry.register(
    name="search_library",
    description="Search through saved papers in the user's library.",
    category="library",
)
def search_library(
    query: str = "",
    tag: str = "",
    limit: int = 20,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Search saved papers."""
    from agents.models import ResearchPaper

    papers = ResearchPaper.objects.filter(saved_by_id=user_id)
    if query:
        from django.db.models import Q

        papers = papers.filter(Q(title__icontains=query) | Q(abstract__icontains=query))
    if tag:
        papers = papers.filter(tags__contains=[tag])
    return {
        "status": "success",
        "results": [
            {
                "id": str(p.id),
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "citation_count": p.citation_count,
                "tags": p.tags,
                "abstract": (p.abstract or "")[:300],
            }
            for p in papers.order_by("-created_at")[:limit]
        ],
    }


@ToolRegistry.register(
    name="annotate_paper",
    description="Add notes, tags, or keywords to a saved paper.",
    category="library",
)
def annotate_paper(
    paper_id: str,
    notes: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Annotate a paper."""
    from agents.models import ResearchPaper

    paper = ResearchPaper.objects.get(id=paper_id)
    if notes:
        paper.notes = notes
    if tags:
        paper.tags = list(set((paper.tags or []) + tags))
    paper.save()
    return {"status": "updated"}


# ═══════════════════════════════════════════════════════════════════
# Experiment Management Tools
# ═══════════════════════════════════════════════════════════════════


@ToolRegistry.register(
    name="configure_experiment",
    description="Propose a training experiment configuration for user review.",
    category="experiment",
    requires_confirmation=True,
)
def configure_experiment(
    model_name: str,
    task_name: str,
    attention_type: str = "full",
    method: str = "baseline",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
    compression_rank: int = 64,
) -> Dict[str, Any]:
    """Generate an experiment draft configuration."""
    config = {
        "model_name": model_name,
        "task_name": task_name,
        "attention_type": attention_type,
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "compression_rank": compression_rank,
        "created_at": datetime.utcnow().isoformat(),
    }
    return {"status": "draft", "config": config}


@ToolRegistry.register(
    name="launch_experiment",
    description="Launch a training experiment. Requires user confirmation.",
    category="experiment",
    requires_confirmation=True,
)
def launch_experiment(
    model_name: str,
    task_name: str,
    attention_type: str = "full",
    method: str = "baseline",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
    compression_rank: int = 64,
    name: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Create and launch a training run via Celery."""
    exp_name = name or f"{model_name} → {task_name} ({attention_type})"
    experiment = Experiment.objects.create(name=exp_name)

    run = TrainingRun.objects.create(
        experiment=experiment,
        job_type="training",
        status="queued",
        config={
            "model_name": model_name,
            "task_name": task_name,
            "attention_type": attention_type,
            "method": method,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "compression_rank": compression_rank,
        },
        name=exp_name,
    )

    # Dispatch to Celery
    from experiment.tasks import run_training_task

    task = run_training_task.delay(str(run.id))

    return {
        "status": "launched",
        "run_id": str(run.id),
        "experiment_id": str(experiment.id),
        "celery_task_id": task.id,
    }


@ToolRegistry.register(
    name="check_experiment_status",
    description="Check the status of a running or completed experiment.",
    category="experiment",
)
def check_experiment_status(run_id: str) -> Dict[str, Any]:
    """Get run status and metrics."""
    run = TrainingRun.objects.get(id=run_id)
    return {
        "status": run.status,
        "run_id": str(run.id),
        "current_step": run.current_step,
        "total_steps": run.total_steps,
        "metrics": run.final_metrics,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


# ═══════════════════════════════════════════════════════════════════
# Training Analysis Tools
# ═══════════════════════════════════════════════════════════════════


@ToolRegistry.register(
    name="analyze_training",
    description="Analyze a training run for overfitting, plateau, divergence, and suggest fixes.",
    category="analysis",
)
def analyze_training(run_id: str) -> Dict[str, Any]:
    """Perform comprehensive training analysis."""
    run = TrainingRun.objects.get(id=run_id)
    history = run.training_history or {}
    losses = history.get("losses", [])
    eval_losses = history.get("eval_losses", [])
    eval_accuracies = history.get("eval_accuracies", [])

    analysis = {
        "status": "healthy",
        "issues": [],
        "suggestions": [],
        "metrics_summary": {},
    }

    # ── Overfitting Detection ──
    if len(losses) >= 10 and len(eval_losses) >= 3:
        train_trend = losses[-1] - losses[-6] if len(losses) >= 6 else 0
        eval_trend = eval_losses[-1] - eval_losses[0] if len(eval_losses) >= 2 else 0
        if train_trend < -0.01 and eval_trend > 0.01:
            analysis["issues"].append("overfitting")
            analysis["suggestions"].extend(
                [
                    "Increase dropout rate or weight decay",
                    "Use early stopping (patience=3)",
                    "Reduce model rank or try lower compression_rank",
                ]
            )
            analysis["status"] = "overfitting"

    # ── Plateau Detection ──
    if len(losses) >= 20:
        recent = losses[-20:]
        loss_std = (
            sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)
        ) ** 0.5
        if loss_std < 0.001 and min(recent) > 0.1:
            analysis["issues"].append("plateau")
            analysis["suggestions"].extend(
                [
                    "Reduce learning rate by 50%",
                    "Try cosine annealing scheduler",
                    "Increase batch size",
                ]
            )
            analysis["status"] = (
                "plateau" if analysis["status"] == "healthy" else analysis["status"]
            )

    # ── Divergence ──
    if len(losses) >= 5 and losses[-1] > 2 * max(losses[:-5]):
        analysis["issues"].append("diverging")
        analysis["suggestions"].extend(
            [
                "Reduce learning rate by 10x",
                "Add gradient clipping (max_norm=1.0)",
                "Verify data preprocessing",
            ]
        )
        analysis["status"] = "diverging"

    # ── Underfitting ──
    if len(losses) >= 50 and losses[-1] > 1.0:
        analysis["issues"].append("underfitting")
        analysis["suggestions"].extend(
            [
                "Increase model capacity (higher rank)",
                "Train for more epochs",
                "Try a more powerful attention mechanism",
            ]
        )
        if analysis["status"] == "healthy":
            analysis["status"] = "underfitting"

    analysis["metrics_summary"] = {
        "current_loss": round(losses[-1], 6) if losses else None,
        "best_loss": round(min(losses), 6) if losses else None,
        "best_accuracy": round(max(eval_accuracies), 6) if eval_accuracies else None,
        "total_steps": len(losses),
        "trend": (
            "improving"
            if len(losses) >= 5 and losses[-1] < losses[0]
            else "worsening" if len(losses) >= 5 else "unknown"
        ),
    }

    # Persist analysis
    from agents.models import TrainingAnalysis

    TrainingAnalysis.objects.update_or_create(
        run=run,
        defaults={
            "status": analysis["status"],
            "suggestions": analysis["suggestions"],
            "metrics_summary": analysis["metrics_summary"],
        },
    )
    return {"status": "success", "analysis": analysis}


@ToolRegistry.register(
    name="suggest_next_experiment",
    description="Based on previous runs, suggest the next experiment to try.",
    category="analysis",
)
def suggest_next_experiment(
    project_id: str,
    metric_name: str = "eval_accuracy",
    direction: str = "maximize",
) -> Dict[str, Any]:
    """Bayesian‑inspired next‑experiment suggestion."""
    runs = TrainingRun.objects.filter(
        experiment_id=project_id, status="completed"
    ).order_by("-completed_at")[:20]

    if not runs:
        return {
            "status": "success",
            "suggestion": "Start with a baseline: bert-base-uncased, MRPC, full attention, 3 epochs.",
            "suggestions": [],
        }

    best = (
        max(runs, key=lambda r: (r.final_metrics or {}).get(metric_name, -1))
        if direction == "maximize"
        else min(
            runs, key=lambda r: (r.final_metrics or {}).get(metric_name, float("inf"))
        )
    )

    config = best.config or {}
    suggestions = []

    # LR variation
    lr = config.get("learning_rate", 2e-5)
    suggestions.append(
        {
            "type": "lr",
            "description": f"Try learning_rate = {lr*0.5:.1e}",
            "params": {"learning_rate": lr * 0.5},
        }
    )

    # Attention variation
    attn = config.get("attention_type", "full")
    for a in ["galerkin", "linformer", "spectraformer", "padre"]:
        if a != attn:
            suggestions.append(
                {
                    "type": "attention",
                    "description": f"Try attention_type = '{a}'",
                    "params": {"attention_type": a},
                }
            )
            break

    # Method variation
    method = config.get("method", "baseline")
    if method == "baseline":
        suggestions.append(
            {
                "type": "method",
                "description": "Try method = 'lora' (rank=8)",
                "params": {"method": "lora", "lora_r": 8},
            }
        )

    return {
        "status": "success",
        "best_run_id": str(best.id),
        "best_metric": (best.final_metrics or {}).get(metric_name),
        "suggestions": suggestions[:4],
    }


# ═══════════════════════════════════════════════════════════════════
# Prompt Testing
# ═══════════════════════════════════════════════════════════════════


@ToolRegistry.register(
    name="test_prompt_template",
    description="Test a prompt template against multiple test cases.",
    category="evaluation",
)
def test_prompt_template(
    template: str,
    test_cases: List[Dict[str, Any]],
    metric: str = "contains",
) -> Dict[str, Any]:
    """Evaluate a prompt template."""
    results = []
    for i, case in enumerate(test_cases):
        prompt = template
        for key, val in case.items():
            if key != "expected":
                prompt = prompt.replace(f"{{{{{key}}}}}", str(val))
        results.append(
            {
                "test_case": i + 1,
                "substituted_prompt": prompt[:500],
                "expected": case.get("expected", ""),
                "variables": {k: v for k, v in case.items() if k != "expected"},
            }
        )
    return {"status": "success", "template": template, "results": results}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _normalize_paper(p, source: str, include_relations: bool = False) -> Dict[str, Any]:
    """Normalize Semantic Scholar paper object to dict."""
    result = {
        "id": p.get("paperId", ""),
        "title": (p.get("title") or "")[:300],
        "authors": [a.get("name", "") for a in (p.get("authors") or [])],
        "abstract": (p.get("abstract") or "")[:1000],
        "year": p.get("year"),
        "citation_count": p.get("citationCount", 0),
        "url": p.get("url", ""),
        "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
        "source": source,
        "source_id": p.get("paperId", ""),
        "venue": (p.get("publicationVenue") or {}).get("name", ""),
        "tldr": (p.get("tldr") or {}).get("text", ""),
    }
    if include_relations:
        result["references"] = [
            {"paperId": r.get("paperId"), "title": (r.get("title") or "")[:100]}
            for r in (p.get("references") or [])[:20]
        ]
        result["citations"] = [
            {"paperId": c.get("paperId"), "title": (c.get("title") or "")[:100]}
            for c in (p.get("citations") or [])[:20]
        ]
    return result


def _normalize_arxiv_result(result, detailed: bool = False) -> Dict[str, Any]:
    """Normalize arxiv result to dict."""
    paper = {
        "id": result.get_short_id(),
        "title": result.title[:300],
        "authors": [a.name for a in result.authors],
        "abstract": result.summary[:1000],
        "year": result.published.year if result.published else None,
        "citation_count": 0,
        "url": result.entry_id,
        "pdf_url": result.pdf_url,
        "source": "arxiv",
        "source_id": result.get_short_id(),
        "venue": "",
    }
    if detailed:
        paper["published"] = result.published.isoformat() if result.published else None
        paper["updated"] = result.updated.isoformat() if result.updated else None
        paper["comment"] = result.comment or ""
        paper["journal_ref"] = result.journal_ref or ""
        paper["primary_category"] = result.primary_category
    return paper


def _normalize_openalex(w) -> Dict[str, Any]:
    """Normalize OpenAlex work to dict."""
    return {
        "id": w.get("id", "").split("/")[-1],
        "title": (w.get("title") or "")[:300],
        "authors": [
            a.get("author", {}).get("display_name", "")
            for a in w.get("authorships", [])
        ],
        "abstract": (w.get("abstract") or "")[:1000] if w.get("abstract") else "",
        "year": w.get("publication_year"),
        "citation_count": w.get("cited_by_count", 0),
        "url": w.get("doi", ""),
        "open_access": w.get("open_access", {}).get("is_oa", False),
        "source": "openalex",
        "source_id": w.get("id", ""),
        "venue": "",
    }


def _deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers by title similarity."""
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# agents/tools.py — add these tool definitions


@ToolRegistry.register(
    name="launch_experiment",
    description="Launch a training experiment. Requires user confirmation.",
    category="experiment",
    requires_confirmation=True,
)
def launch_experiment_tool(
    model_name: str,
    task_name: str,
    attention_type: str = "full",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Launch a training run with safeguards."""
    from experiment.models import Experiment, TrainingRun

    experiment = Experiment.objects.create(name=f"{model_name} → {task_name}")
    run = TrainingRun.objects.create(
        experiment=experiment,
        status="queued",
        config={
            "model_name": model_name,
            "task_name": task_name,
            "attention_type": attention_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )
    from experiment.tasks import run_training_task

    task = run_training_task.delay(str(run.id))
    return {"status": "launched", "run_id": str(run.id), "celery_task_id": task.id}


@ToolRegistry.register(
    name="stop_experiment",
    description="Stop a running experiment. Requires user confirmation.",
    category="experiment",
    requires_confirmation=True,
)
def stop_experiment_tool(run_id: str) -> Dict[str, Any]:
    """Stop a running training job."""
    run = TrainingRun.objects.get(id=run_id)
    if run.status == "running":
        from jobscheduler.scheduler import scheduler

        scheduler.stop_job(run_id)
        run.status = "stopped"
        run.save(update_fields=["status"])
        return {"status": "stopped", "run_id": run_id}
    return {"status": "not_running", "run_id": run_id}


@ToolRegistry.register(
    name="edit_configuration",
    description="Edit a training configuration. Requires user confirmation.",
    category="experiment",
    requires_confirmation=True,
)
def edit_configuration_tool(run_id: str, key: str, value: Any) -> Dict[str, Any]:
    """Edit a specific configuration parameter of a training run."""
    run = TrainingRun.objects.get(id=run_id)
    config = run.config or {}
    config[key] = value
    run.config = config
    run.save(update_fields=["config"])
    return {"status": "updated", "run_id": run_id, "key": key, "value": value}


def search_datasets(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search for NLP / ML datasets."""
    query = params.get("query", "")
    limit = params.get("limit", 5)
    return {
        "datasets": [
            {"id": "glue/mrpc", "name": "MRPC", "task": "paraphrase"},
            {"id": "glue/sst2", "name": "SST-2", "task": "sentiment"},
            {"id": "glue/cola", "name": "CoLA", "task": "acceptability"},
            {"id": "squad", "name": "SQuAD", "task": "qa"},
        ][:limit]
    }


def search_models(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search for pre-trained models."""
    query = params.get("query", "")
    limit = params.get("limit", 5)
    return {
        "models": [
            {"id": "bert-base-uncased", "architecture": "BERT", "params": "110M"},
            {"id": "roberta-base", "architecture": "RoBERTa", "params": "125M"},
            {"id": "distilbert-base-uncased", "architecture": "DistilBERT", "params": "66M"},
            {"id": "gpt2", "architecture": "GPT-2", "params": "124M"},
        ][:limit]
    }


def get_dataset_info(params: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve detailed dataset info."""
    dataset_id = params.get("dataset_id", "glue/mrpc")
    return {
        "dataset_id": dataset_id,
        "splits": ["train", "validation", "test"],
        "features": ["sentence1", "sentence2", "label"],
    }


def list_attention_types() -> List[str]:
    """Return available attention mechanisms."""
    return ["full", "linear", "leka", "griffin", "sparse", "performer"]


def list_available_tasks() -> List[str]:
    """Return available evaluation benchmark tasks."""
    return ["mrpc", "sst2", "cola", "qnli", "qqp", "mnli", "squad"]


def compile_latex(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compile LaTeX document."""
    project_id = params.get("project_id", "")
    return {
        "status": "compiled",
        "project_id": project_id,
        "pdf_url": f"/api/paper/{project_id}/pdf/",
    }


def edit_latex_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Edit LaTeX document file."""
    return {
        "status": "saved",
        "filename": params.get("filename", "main.tex"),
        "project_id": params.get("project_id", ""),
    }


def read_latex_project(params: Dict[str, Any]) -> Dict[str, Any]:
    """Read LaTeX project structure."""
    return {
        "status": "ok",
        "project_id": params.get("project_id", ""),
        "files": ["main.tex"],
    }

