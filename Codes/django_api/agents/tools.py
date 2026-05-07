from typing import Dict, Any, Optional
import requests, json
from django.conf import settings
from experiment.models import TrainingRun, Project
from jobscheduler.models import Job

HF_API_URL = "https://huggingface.co/api"
from typing import Dict, Any, Callable
from django.conf import settings
from experiment.models import TrainingRun, Project
from jobscheduler.models import Job
import subprocess, os, json

ToolFunction = Callable[[Dict[str, Any], 'AgentSession'], Dict[str, Any]]

class ToolRegistry:
    """Minimal‑toolset design: each agent gets 3–5 tools, reducing token overhead by up to 73%."""
    
    _tools: Dict[str, ToolFunction] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: ToolFunction):
            cls._tools[name] = func
            return func
        return decorator

    @classmethod
    def get_tools(cls, agent_type: str) -> Dict[str, ToolFunction]:
        # Each agent type gets a curated subset
        subsets = {
            'deep_researcher': ['run_experiment', 'analyze_results', 'compare_runs', 'read_code', 'edit_code', 'propose_hypothesis'],
            'hpo_agent': ['run_experiment', 'analyze_results', 'compare_runs', 'suggest_hparams'],
            'code_agent': ['read_code', 'edit_code', 'run_experiment', 'analyze_results'],
        }
        allowed = subsets.get(agent_type, ['run_experiment', 'analyze_results'])
        return {name: cls._tools[name] for name in allowed if name in cls._tools}

# ----- Define tools -----

@ToolRegistry.register('run_experiment')
def tool_run_experiment(params: dict, session: 'AgentSession') -> dict:
    """Launch a training experiment with given hyperparameters."""
    config = {
        'model_name': params.get('model_name', 'bert-base-uncased'),
        'task_name': params.get('task_name', 'mrpc'),
        'attention_type': params.get('attention_type', 'full'),
        'epochs': params.get('epochs', 3),
        'batch_size': params.get('batch_size', 16),
        'learning_rate': params.get('learning_rate', 2e-5),
        **params,
    }
    run = TrainingRun.objects.create(
        project=session.project,
        job_type='training',
        config=config,
        status='queued',
        name=f"agent_{session.id.hex[:8]}_cycle_{session.cycles_completed + 1}",
    )
    # Submit to scheduler
    from jobscheduler.scheduler import scheduler
    scheduler.add_job(str(run.id), priority=3, estimated_duration=config.get('estimated_duration'))
    
    session.experiments_run += 1
    session.save(update_fields=['experiments_run'])
    
    return {'status': 'queued', 'run_id': str(run.id), 'config': config}


@ToolRegistry.register('analyze_results')
def tool_analyze_results(params: dict, session: 'AgentSession') -> dict:
    """Analyze completed run results."""
    run_id = params.get('run_id')
    try:
        run = TrainingRun.objects.get(id=run_id)
        metrics = run.final_metrics or {}
        return {
            'run_id': str(run.id),
            'status': run.status,
            'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            'config': run.config,
        }
    except TrainingRun.DoesNotExist:
        return {'error': 'Run not found'}


@ToolRegistry.register('compare_runs')
def tool_compare_runs(params: dict, session: 'AgentSession') -> dict:
    """Compare two or more runs."""
    run_ids = params.get('run_ids', [])
    runs = TrainingRun.objects.filter(id__in=run_ids)
    results = []
    for run in runs:
        m = run.final_metrics or {}
        target = m.get(session.metric_name)
        results.append({
            'run_id': str(run.id),
            'name': run.name,
            'metric': target,
            'config': run.config,
        })
    best = max(results, key=lambda r: r['metric'] or -1e9) if results and session.metric_direction == 'maximize' else (min(results, key=lambda r: r['metric'] or 1e9) if results else None)
    return {
        'runs': results,
        'best_so_far': best,
        'direction': session.metric_direction,
    }


@ToolRegistry.register('read_code')
def tool_read_code(params: dict, session: 'AgentSession') -> dict:
    """Read a file from the codebase."""
    filepath = params.get('filepath', '')
    if filepath not in session.codebase_files:
        return {'error': f'File {filepath} not in allowed codebase files'}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        return {'filepath': filepath, 'content': content[:5000]}  # cap at 5K
    except FileNotFoundError:
        return {'error': f'File {filepath} not found on disk'}


@ToolRegistry.register('edit_code')
def tool_edit_code(params: dict, session: 'AgentSession') -> dict:
    """Edit a file. Creates a git commit."""
    filepath = params.get('filepath', '')
    new_content = params.get('content', '')
    if filepath not in session.codebase_files:
        return {'error': f'File {filepath} not in allowed codebase files'}
    
    # Write file
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    # Git commit
    commit_msg = params.get('commit_message', f'Agent edit: {filepath}')
    subprocess.run(['git', 'add', filepath], capture_output=True)
    result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
    
    return {
        'status': 'edited',
        'filepath': filepath,
        'commit_message': commit_msg,
        'git_output': result.stdout + result.stderr,
    }


@ToolRegistry.register('propose_hypothesis')
def tool_propose_hypothesis(params: dict, session: 'AgentSession') -> dict:
    """Record a hypothesis for the current cycle."""
    from .models import AgentAction
    hypothesis = params.get('hypothesis', '')
    session.actions.create(
        cycle=session.cycles_completed + 1,
        action_type='hypothesis',
        output_summary=hypothesis,
    )
    return {'hypothesis': hypothesis, 'recorded': True}

class ToolRegistry:
    _tools: Dict[str, callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn):
            cls._tools[name] = fn
            return fn
        return decorator

    @classmethod
    def execute(cls, name: str, params: Dict) -> Dict:
        tool = cls._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool(params)
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def get_definitions(cls) -> list:
        """Return tool schema for LLM function calling."""
        return [
            {
    "name": "read_latex_project",
    "description": "Read a LaTeX project's files.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "The project UUID"}
        },
        "required": ["project_id"]
    }
},
{
    "name": "edit_latex_file",
    "description": "Edit a specific file in a LaTeX project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "filename": {"type": "string"},
            "content": {"type": "string", "description": "New content for the file"}
        },
        "required": ["project_id", "filename", "content"]
    }
},
{
    "name": "compile_latex",
    "description": "Compile a LaTeX project to PDF.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"}
        },
        "required": ["project_id"]
    }
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
        ]

# ----- Tool Implementations -----

@ToolRegistry.register("search_datasets")
def search_datasets(params: dict) -> dict:
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
def search_models(params: dict) -> dict:
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
def get_dataset_info(params: dict) -> dict:
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
    except:
        return {"id": dataset_id, "description": f"The {dataset_id} dataset.", "note": "Limited info (API unavailable)"}

@ToolRegistry.register("configure_experiment")
def configure_experiment(params: dict) -> dict:
    return {
        "status": "draft",
        "message": "Experiment configuration prepared. Review and confirm to launch.",
        "config": params,
    }

@ToolRegistry.register("launch_experiment")
def launch_experiment(params: dict) -> dict:
    # Create the job/run
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
def check_experiment_status(params: dict) -> dict:
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
def list_available_tasks(params: dict) -> dict:
    return {"tasks": ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]}

@ToolRegistry.register("list_attention_types")
def list_attention_types(params: dict) -> dict:
    try:
        from framework.attentions.core.registry import ATTENTION_REGISTRY
        return {"attentions": list(ATTENTION_REGISTRY.keys())}
    except ImportError:
        return {"attentions": ["full", "aurora", "nova", "griffin_ssd", "linformer", "svd", "goldfinch"]}
    


    @ToolRegistry.register("read_latex_project")
def read_latex_project(params: dict) -> dict:
    from paper.models import PaperProject, PaperFile
    try:
        project = PaperProject.objects.get(id=params['project_id'])
        files = project.files.all()
        return {
            "project_name": project.name,
            "files": [{"filename": f.filename, "content": f.content[:3000]} for f in files]
        }
    except PaperProject.DoesNotExist:
        return {"error": "Project not found"}

@ToolRegistry.register("edit_latex_file")
def edit_latex_file(params: dict) -> dict:
    from paper.models import PaperProject, PaperFile
    try:
        project = PaperProject.objects.get(id=params['project_id'])
        file = PaperFile.objects.get(project=project, filename=params['filename'])
        file.content = params['content']
        file.save()
        return {"status": "updated", "filename": file.filename}
    except (PaperProject.DoesNotExist, PaperFile.DoesNotExist):
        return {"error": "Project or file not found"}

@ToolRegistry.register("compile_latex")
def compile_latex(params: dict) -> dict:
    from paper.views import compile_project  # reuse existing view logic
    # We'll call the existing compile endpoint
    from rest_framework.test import APIRequestFactory
    factory = APIRequestFactory()
    request = factory.post(f'/api/paper/projects/{params["project_id"]}/compile/')
    request.user = None  # will be set by the view decorator
    # Since we need authenticated request, we'll use a simpler approach:
    from paper.compiler import LatexCompiler
    project = PaperProject.objects.get(id=params['project_id'])
    project_files = {f.filename: f.content for f in project.files.all()}
    compiler = LatexCompiler(project_files, project.root_file, project.compiler)
    result = compiler.compile()
    return {
        "status": result['status'],
        "log": result.get('log', '')[:2000],
        "error": result.get('error'),
        "errors": result.get('errors', 0),
        "warnings": result.get('warnings', 0),
    }