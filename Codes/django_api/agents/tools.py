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
]