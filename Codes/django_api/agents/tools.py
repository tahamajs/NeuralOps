# agents/tools.py
"""
Unified tools for the conversational ML assistant.
- ToolRegistry: registration, execution, and LLM function definitions.
- LangChain tool wrappers: for use with the LangGraph agent.
"""

import os
import json
import subprocess
import requests
from typing import Dict, Any, Optional, Callable
import requests as req_lib
from typing import List, Optional

from django.conf import settings
from experiment.models import TrainingRun, Project
from jobscheduler.models import Job
from paper.models import PaperProject, PaperFile
from paper.compiler import LatexCompiler

HF_API_URL = "https://huggingface.co/api"


# ====================== Tool Registry ======================
class ToolRegistry:
    """Central registry for all tools the agent can call."""

    _tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
            cls._tools[name] = func
            return func
        return decorator

    @classmethod
    def execute(cls, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool by name with given parameters."""
        tool = cls._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool(params)
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def get_definitions(cls) -> list:
        """Return the tool definitions in OpenAI/Anthropic function‑calling format."""
        return [
            {
    "name": "search_papers",
    "description": "Search for research papers on Semantic Scholar/arXiv.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 10},
            "source": {"type": "string", "default": "semantic_scholar", "enum": ["semantic_scholar", "arxiv", "all"]},
        },
        "required": ["query"],
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
            "pdf_url": {"type": "string"},
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
    "name": "get_citation_graph",
    "description": "Get the citation graph for saved papers.",
    "parameters": {
        "type": "object",
        "properties": {
            "paper_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["paper_ids"],
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
{
    "name": "analyze_training",
    "description": "Analyze a training run and suggest improvements.",
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
        },
        "required": ["run_id"],
    },
},
{
    "name": "suggest_next_experiment",
    "description": "Based on previous runs, suggest the next experiment to try.",
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
            {
                "name": "search_datasets",
                "description": "Search HuggingFace for datasets matching a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (e.g., 'glue mrpc', 'imdb')"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_models",
                "description": "Search HuggingFace for models matching a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (e.g., 'bert-base-uncased')"},
                        "task": {"type": "string", "description": "Filter by task type (e.g., 'text-classification')"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_dataset_info",
                "description": "Get detailed info about a specific HuggingFace dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "HuggingFace dataset ID (e.g., 'glue', 'imdb')"},
                    },
                    "required": ["dataset_id"],
                },
            },
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
                "description": "Launch the experiment. Only call this after user confirms they want to launch.",
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
                        "name": {"type": "string", "description": "Optional run name"},
                    },
                    "required": ["model_name", "task_name"],
                },
            },
            {
                "name": "check_experiment_status",
                "description": "Check the status of a running or completed experiment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "The run ID to check"},
                    },
                    "required": ["run_id"],
                },
            },
            {
                "name": "list_available_tasks",
                "description": "List all GLUE benchmark tasks available for training.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "list_attention_types",
                "description": "List all available attention mechanisms.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "read_latex_project",
                "description": "Read all files in a LaTeX project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "The project UUID"},
                    },
                    "required": ["project_id"],
                },
            },
            {
                "name": "edit_latex_file",
                "description": "Edit a specific file in a LaTeX project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "content": {"type": "string", "description": "New content for the file"},
                    },
                    "required": ["project_id", "filename", "content"],
                },
            },
            {
                "name": "compile_latex",
                "description": "Compile a LaTeX project to PDF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                    },
                    "required": ["project_id"],
                },
            },
        ]


# ====================== Tool Implementations ======================

@ToolRegistry.register("search_datasets")
def search_datasets(params: Dict[str, Any]) -> Dict[str, Any]:
    query = params.get("query", "")
    limit = params.get("limit", 5)
    try:
        resp = requests.get(f"{HF_API_URL}/datasets", params={"search": query, "limit": limit}, timeout=10)
        datasets = resp.json()
        return {
            "results": [
                {"id": d.get("id"), "description": (d.get("description") or "")[:200], "downloads": d.get("downloads", 0)}
                for d in datasets[:limit]
            ]
        }
    except Exception as e:
        return {"error": str(e), "fallback": [
            {"id": "glue", "description": "GLUE benchmark (includes MRPC, SST-2, CoLA, etc.)"},
            {"id": "imdb", "description": "IMDB movie reviews for sentiment analysis"},
            {"id": "squad", "description": "Stanford Question Answering Dataset"},
        ]}

@ToolRegistry.register("search_models")
def search_models(params: Dict[str, Any]) -> Dict[str, Any]:
    query = params.get("query", "")
    limit = params.get("limit", 5)
    try:
        resp = requests.get(f"{HF_API_URL}/models", params={"search": query, "limit": limit}, timeout=10)
        models = resp.json()
        return {
            "results": [
                {"id": m.get("modelId"), "downloads": m.get("downloads", 0), "pipeline_tag": m.get("pipeline_tag", "")}
                for m in models[:limit]
            ]
        }
    except Exception as e:
        fallback = [
            {"id": "bert-base-uncased", "downloads": 50000000, "pipeline_tag": "fill-mask"},
            {"id": "distilbert-base-uncased", "downloads": 30000000, "pipeline_tag": "fill-mask"},
            {"id": "roberta-base", "downloads": 25000000, "pipeline_tag": "fill-mask"},
            {"id": "google/electra-small-discriminator", "downloads": 5000000, "pipeline_tag": "text-classification"},
            {"id": "microsoft/deberta-v3-base", "downloads": 8000000, "pipeline_tag": "text-classification"},
        ]
        return {"results": fallback, "note": "Using cached results (HuggingFace API unavailable)"}

@ToolRegistry.register("get_dataset_info")
def get_dataset_info(params: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = params.get("dataset_id", "")
    try:
        resp = requests.get(f"{HF_API_URL}/datasets/{dataset_id}", timeout=10)
        info = resp.json()
        return {
            "id": info.get("id"),
            "description": (info.get("description") or "")[:500],
            "tags": info.get("tags", []),
            "size": info.get("size", "unknown"),
        }
    except Exception:
        return {"id": dataset_id, "description": f"The {dataset_id} dataset.", "note": "Limited info (API unavailable)"}

@ToolRegistry.register("configure_experiment")
def configure_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "draft",
        "message": "Experiment configuration prepared. Review and confirm to launch.",
        "config": params,
    }

@ToolRegistry.register("launch_experiment")
def launch_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    job = Job.objects.create(
        name=params.get("name", ""),
        model_name=params["model_name"],
        task_name=params["task_name"],
        attention_type=params.get("attention_type", "full"),
        method=params.get("method", "baseline"),
        epochs=params.get("epochs", 3),
        batch_size=params.get("batch_size", 16),
        learning_rate=params.get("learning_rate", 2e-5),
        max_seq_length=params.get("max_seq_length", 128),
        compression_rank=params.get("compression_rank", 64),
        status="queued",
    )
    from jobscheduler.scheduler import scheduler
    scheduler.add_job(str(job.id), priority=5)
    return {
        "status": "launched",
        "run_id": str(job.id),
        "monitor_url": f"/experiments/{job.id}",
        "message": f"🚀 Experiment launched! ID: {str(job.id)[:8]}. You can monitor it now.",
    }

@ToolRegistry.register("check_experiment_status")
def check_experiment_status(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        job = Job.objects.get(id=params["run_id"])
        return {
            "run_id": str(job.id),
            "status": job.status,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "final_metrics": job.final_metrics,
        }
    except Job.DoesNotExist:
        return {"error": "Run not found"}

@ToolRegistry.register("list_available_tasks")
def list_available_tasks(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"tasks": ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]}

@ToolRegistry.register("list_attention_types")
def list_attention_types(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from framework.attentions.core.registry import ATTENTION_REGISTRY
        return {"attentions": list(ATTENTION_REGISTRY.keys())}
    except ImportError:
        return {"attentions": ["full", "aurora", "nova", "griffin_ssd", "linformer", "svd", "goldfinch"]}

@ToolRegistry.register("read_latex_project")
def read_latex_project(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        project = PaperProject.objects.get(id=params["project_id"])
        files = project.files.all()
        return {
            "project_name": project.name,
            "files": [{"filename": f.filename, "content": f.content[:3000]} for f in files]
        }
    except PaperProject.DoesNotExist:
        return {"error": "Project not found"}

@ToolRegistry.register("edit_latex_file")
def edit_latex_file(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        project = PaperProject.objects.get(id=params["project_id"])
        file = PaperFile.objects.get(project=project, filename=params["filename"])
        file.content = params["content"]
        file.save()
        return {"status": "updated", "filename": file.filename}
    except (PaperProject.DoesNotExist, PaperFile.DoesNotExist):
        return {"error": "Project or file not found"}

@ToolRegistry.register("compile_latex")
def compile_latex(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        project = PaperProject.objects.get(id=params["project_id"])
        project_files = {f.filename: f.content for f in project.files.all()}
        compiler = LatexCompiler(project_files, project.root_file, project.compiler)
        result = compiler.compile()
        return {
            "status": result["status"],
            "log": result.get("log", "")[:2000],
            "error": result.get("error"),
            "errors": result.get("errors", 0),
            "warnings": result.get("warnings", 0),
        }
    except PaperProject.DoesNotExist:
        return {"error": "Project not found"}


# ====================== LangChain Tool Wrappers ======================
# These use the @tool decorator from LangChain to wrap the plain functions,
# making them callable by the LangGraph agent.

from langchain_core.tools import tool  # ensure langchain-core is installed

@tool
def tool_search_datasets(query: str, limit: int = 5) -> dict:
    """Search HuggingFace for datasets matching a query."""
    return search_datasets({"query": query, "limit": limit})

@tool
def tool_search_models(query: str, task: Optional[str] = None, limit: int = 5) -> dict:
    """Search HuggingFace for models matching a query."""
    return search_models({"query": query, "task": task, "limit": limit})

@tool
def tool_get_dataset_info(dataset_id: str) -> dict:
    """Get detailed info about a specific HuggingFace dataset."""
    return get_dataset_info({"dataset_id": dataset_id})

@tool
def tool_configure_experiment(
    model_name: str,
    task_name: str,
    attention_type: str = "full",
    method: str = "baseline",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
    compression_rank: int = 64,
) -> dict:
    """Propose an experiment configuration for review."""
    return configure_experiment({
        "model_name": model_name,
        "task_name": task_name,
        "attention_type": attention_type,
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "compression_rank": compression_rank,
    })

@tool
def tool_launch_experiment(
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
) -> dict:
    """Launch the experiment. Only call after user confirmation."""
    return launch_experiment({
        "model_name": model_name,
        "task_name": task_name,
        "attention_type": attention_type,
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "compression_rank": compression_rank,
        "name": name,
    })

@tool
def tool_check_experiment_status(run_id: str) -> dict:
    """Check the status of a running/completed experiment."""
    return check_experiment_status({"run_id": run_id})

@tool
def tool_list_available_tasks() -> dict:
    """List all available GLUE benchmark tasks."""
    return list_available_tasks({})

@tool
def tool_list_attention_types() -> dict:
    """List all available attention mechanisms."""
    return list_attention_types({})

@tool
def tool_read_latex_project(project_id: str) -> dict:
    """Read a LaTeX project's files."""
    return read_latex_project({"project_id": project_id})

@tool
def tool_edit_latex_file(project_id: str, filename: str, content: str) -> dict:
    """Edit a specific file in a LaTeX project."""
    return edit_latex_file({"project_id": project_id, "filename": filename, "content": content})

@tool
def tool_compile_latex(project_id: str) -> dict:
    """Compile a LaTeX project to PDF."""
    return compile_latex({"project_id": project_id})
import requests as req_lib
from typing import List, Optional

# ====================== Literature Search Tools ======================

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"

@ToolRegistry.register("search_papers")
def search_papers(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search for research papers on Semantic Scholar and arXiv."""
    query = params.get("query", "")
    limit = params.get("limit", 10)
    source = params.get("source", "semantic_scholar")  # semantic_scholar, arxiv
    
    results = []
    
    # Search Semantic Scholar
    if source in ("semantic_scholar", "all"):
        try:
            resp = req_lib.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params={"query": query, "limit": limit, "fields": "title,authors,abstract,year,citationCount,url,externalIds"},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                for paper in data.get("data", []):
                    results.append({
                        "id": paper.get("paperId"),
                        "title": paper.get("title", ""),
                        "authors": [a.get("name", "") for a in paper.get("authors", [])],
                        "abstract": (paper.get("abstract") or "")[:500],
                        "year": paper.get("year"),
                        "citation_count": paper.get("citationCount", 0),
                        "url": paper.get("url", ""),
                        "source": "semantic_scholar",
                        "source_id": paper.get("paperId"),
                        "arxiv_id": paper.get("externalIds", {}).get("ArXiv"),
                    })
        except Exception as e:
            pass  # Fallback below

    if source in ("arxiv", "all"):
        try:
            resp = req_lib.get(f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={limit}", timeout=15)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            for entry in root.findall('atom:entry', ns):
                arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
                results.append({
                    "id": arxiv_id,
                    "title": entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                    "authors": [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
                    "abstract": entry.find('atom:summary', ns).text.strip()[:500],
                    "year": entry.find('atom:published', ns).text[:4] if entry.find('atom:published', ns) is not None else None,
                    "citation_count": 0,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "source": "arxiv",
                    "source_id": arxiv_id,
                })
        except Exception:
            pass

    if not results:
        return {"results": [], "message": "No papers found. Try different keywords."}

    return {"results": results[:limit], "total": len(results)}


@ToolRegistry.register("save_paper")
def save_paper(params: Dict[str, Any]) -> Dict[str, Any]:
    """Save a paper to the user's research library."""
    from .models import ResearchPaper
    user_id = params.get("user_id")
    if not user_id:
        return {"error": "user_id required"}
    
    paper, created = ResearchPaper.objects.get_or_create(
        source=params.get("source", "semantic_scholar"),
        source_id=params.get("source_id", params.get("id", "")),
        defaults={
            "title": params.get("title", ""),
            "authors": params.get("authors", []),
            "abstract": params.get("abstract", ""),
            "url": params.get("url", ""),
            "year": params.get("year"),
            "citation_count": params.get("citation_count", 0),
            "pdf_url": params.get("pdf_url", ""),
            "saved_by_id": user_id,
        }
    )
    return {"status": "saved" if created else "already_exists", "paper_id": str(paper.id)}


@ToolRegistry.register("search_library")
def search_library(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search the user's saved paper library."""
    from .models import ResearchPaper
    user_id = params.get("user_id")
    query = params.get("query", "")
    tag = params.get("tag", "")
    
    papers = ResearchPaper.objects.filter(saved_by_id=user_id)
    if query:
        papers = papers.filter(title__icontains=query) | papers.filter(abstract__icontains=query)
    if tag:
        papers = papers.filter(tags__contains=[tag])
    
    papers = papers.order_by('-created_at')[:params.get("limit", 20)]
    
    return {
        "results": [
            {
                "id": str(p.id),
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract[:300],
                "year": p.year,
                "url": p.url,
                "tags": p.tags,
            }
            for p in papers
        ]
    }


@ToolRegistry.register("get_citation_graph")
def get_citation_graph(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get the citation graph for a set of papers."""
    from .models import ResearchPaper, CitationEdge
    paper_ids = params.get("paper_ids", [])
    user_id = params.get("user_id")
    
    papers = ResearchPaper.objects.filter(id__in=paper_ids, saved_by_id=user_id)
    edges = CitationEdge.objects.filter(
        source_paper__in=papers
    ) | CitationEdge.objects.filter(
        target_paper__in=papers
    )
    
    nodes = [
        {"id": str(p.id), "title": p.title[:80], "year": p.year, "citation_count": p.citation_count}
        for p in papers
    ]
    edge_list = [
        {"from": str(e.source_paper_id), "to": str(e.target_paper_id), "relationship": e.relationship}
        for e in edges
    ]
    
    return {"nodes": nodes, "edges": edge_list}


@ToolRegistry.register("annotate_paper")
def annotate_paper(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add notes or tags to a saved paper."""
    from .models import ResearchPaper
    paper = ResearchPaper.objects.get(id=params["paper_id"])
    if params.get("notes"):
        paper.notes = params["notes"]
    if params.get("tags"):
        paper.tags = list(set(paper.tags + params["tags"]))
    paper.save()
    return {"status": "updated", "paper_id": str(paper.id)}
@ToolRegistry.register("analyze_training")
def analyze_training(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a training run and provide suggestions for improvement."""
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
    
    # Detect overfitting: train loss decreasing but eval loss increasing
    if len(losses) >= 10 and len(eval_losses) >= 3:
        recent_train = losses[-5:]
        recent_eval = eval_losses[-3:]
        if len(recent_train) >= 2 and len(recent_eval) >= 2:
            train_trend = recent_train[-1] - recent_train[0]
            eval_trend = recent_eval[-1] - recent_eval[0]
            if train_trend < -0.01 and eval_trend > 0.01:
                analysis["issues"].append("overfitting")
                analysis["suggestions"].extend([
                    "Increase dropout rate",
                    "Add or increase weight decay",
                    "Use early stopping",
                    "Reduce model complexity (lower rank or fewer layers)",
                ])
                analysis["status"] = "overfitting"
    
    # Detect plateau: loss not changing significantly
    if len(losses) >= 20:
        recent_20 = losses[-20:]
        loss_std = sum((x - sum(recent_20)/len(recent_20))**2 for x in recent_20) / len(recent_20)
        if loss_std < 0.0001 and min(recent_20) > 0.1:
            analysis["issues"].append("plateau")
            analysis["suggestions"].extend([
                "Reduce learning rate",
                "Try a different learning rate scheduler",
                "Increase batch size for more stable gradients",
            ])
            analysis["status"] = "plateau" if analysis["status"] == "healthy" else analysis["status"]
    
    # Detect underfitting: loss still high after many steps
    if len(losses) >= 50 and losses[-1] > 0.5:
        analysis["issues"].append("underfitting")
        analysis["suggestions"].extend([
            "Increase model capacity (higher rank, more layers)",
            "Train for more epochs",
            "Try a more powerful attention mechanism",
        ])
        analysis["status"] = "underfitting" if analysis["status"] == "healthy" else analysis["status"]
    
    # Detect divergence: loss suddenly spikes
    if len(losses) >= 5:
        if losses[-1] > 2 * max(losses[:-5]) and losses[-1] > 1.0:
            analysis["issues"].append("diverging")
            analysis["suggestions"].extend([
                "Reduce learning rate significantly",
                "Check for exploding gradients (add gradient clipping)",
                "Verify data preprocessing",
            ])
            analysis["status"] = "diverging"
    
    # Metrics summary
    analysis["metrics_summary"] = {
        "current_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "current_eval_accuracy": eval_accuracies[-1] if eval_accuracies else None,
        "best_eval_accuracy": max(eval_accuracies) if eval_accuracies else None,
        "total_steps": len(losses),
        "loss_trend": "decreasing" if len(losses) >= 5 and losses[-1] < losses[0] else "increasing" if len(losses) >= 5 else "unknown",
    }
    
    # Save analysis to DB
    from .models import TrainingAnalysis
    TrainingAnalysis.objects.update_or_create(
        run=run,
        defaults={
            "status": analysis["status"],
            "confidence": 0.8,
            "suggestions": analysis["suggestions"],
            "metrics_summary": analysis["metrics_summary"],
        }
    )
    
    return analysis


@ToolRegistry.register("suggest_next_experiment")
def suggest_next_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """Based on previous runs, suggest the next experiment to try."""
    project_id = params.get("project_id")
    metric_name = params.get("metric_name", "eval_accuracy")
    direction = params.get("direction", "maximize")
    
    runs = TrainingRun.objects.filter(project_id=project_id, status="completed").order_by("-completed_at")[:20]
    
    if not runs:
        return {"suggestion": "No completed runs found. Start with a baseline experiment."}
    
    # Find the best run so far
    best_run = None
    best_metric = -1 if direction == "maximize" else float("inf")
    for run in runs:
        val = (run.final_metrics or {}).get(metric_name)
        if val is not None:
            if (direction == "maximize" and val > best_metric) or (direction == "minimize" and val < best_metric):
                best_metric = val
                best_run = run
    
    if not best_run:
        return {"suggestion": "No metrics found. Run a baseline first."}
    
    # Generate suggestions based on best config
    best_config = best_run.config or {}
    suggestions = []
    
    # Suggest varying learning rate
    current_lr = best_config.get("learning_rate", 2e-5)
    suggestions.append({
        "type": "hyperparameter",
        "description": f"Try learning_rate = {current_lr * 0.5:.1e} or {current_lr * 2:.1e}",
        "params": {"learning_rate": current_lr * 0.5},
    })
    
    # Suggest different attention type
    current_attn = best_config.get("attention_type", "full")
    attention_options = ["aurora", "nova", "griffin_ssd", "linformer"]
    for attn in attention_options:
        if attn != current_attn:
            suggestions.append({
                "type": "architecture",
                "description": f"Try attention_type = '{attn}' instead of '{current_attn}'",
                "params": {"attention_type": attn},
            })
            break
    
    # Suggest different method
    current_method = best_config.get("method", "baseline")
    if current_method == "baseline":
        suggestions.append({
            "type": "method",
            "description": "Try method = 'lora' with rank 8",
            "params": {"method": "lora", "lora_r": 8},
        })
    
    return {
        "best_run": str(best_run.id),
        "best_metric": best_metric,
        "metric_name": metric_name,
        "suggestions": suggestions[:3],
    }
# List of all LangChain tools (for agent binding)
ALL_LANGCHAIN_TOOLS = [
    tool_search_datasets,
    tool_search_models,
    tool_get_dataset_info,
    tool_configure_experiment,
    tool_launch_experiment,
    tool_check_experiment_status,
    tool_list_available_tasks,
    tool_list_attention_types,
    tool_read_latex_project,
    tool_edit_latex_file,
    tool_compile_latex,
    search_papers,
    save_paper,
    search_library,
    get_citation_graph,
    annotate_paper
]

@tool
def tool_search_papers(query: str, limit: int = 10, source: str = "semantic_scholar") -> dict:
    """Search for research papers."""
    return search_papers({"query": query, "limit": limit, "source": source})

@tool
def tool_save_paper(title: str, source: str, source_id: str, authors: Optional[List[str]] = None, abstract: str = "", url: str = "", year: Optional[int] = None, citation_count: int = 0, pdf_url: str = "") -> dict:
    """Save a paper to the library."""
    return save_paper({"title": title, "source": source, "source_id": source_id, "authors": authors or [], "abstract": abstract, "url": url, "year": year, "citation_count": citation_count, "pdf_url": pdf_url})

@tool
def tool_search_library(query: str = "", tag: str = "", limit: int = 20) -> dict:
    """Search saved papers."""
    return search_library({"query": query, "tag": tag, "limit": limit})

@tool
def tool_analyze_training(run_id: str) -> dict:
    """Analyze a training run for issues."""
    return analyze_training({"run_id": run_id})

@tool
def tool_suggest_next_experiment(project_id: str, metric_name: str = "eval_accuracy", direction: str = "maximize") -> dict:
    """Suggest next experiment based on previous runs."""
    return suggest_next_experiment({"project_id": project_id, "metric_name": metric_name, "direction": direction})

# Add to ALL_LANGCHAIN_TOOLS
ALL_LANGCHAIN_TOOLS.extend([
    tool_search_papers,
    tool_save_paper,
    tool_search_library,
    tool_analyze_training,
    tool_suggest_next_experiment,
])