import os
import sys
import json
import time
import traceback
import asyncio
import shutil
import tempfile
import subprocess
import threading
import re
import psutil
import torch
from pathlib import Path
from datetime import datetime, timedelta
from django.http import FileResponse, Http404, HttpResponse
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.db.models import Count, Avg
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from .models import Experiment, TrainingRun, TrainingSnapshot
from .serializers import ExperimentSerializer, TrainingRunSerializer, TrainingSnapshotSerializer
from .jobs import start_job, stop_job as do_stop, cancel_job
from .runners import run_training, run_evaluation, run_compression
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY
import logging
import logging
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_exempt

logger = logging.getLogger(__name__)

# ---------- Utility ----------
def _update_run_status(run: TrainingRun, status: str):
    run.status = status
    if status == 'running' and not run.started_at:
        run.started_at = timezone.now()
    elif status in ('completed', 'failed', 'stopped'):
        run.completed_at = timezone.now()
    run.save()

# ---------- Config discovery (unchanged) ----------
@api_view(["GET"])
def list_attentions(request):
    return Response({"attentions": list(ATTENTION_REGISTRY.keys())})

@api_view(["GET"])
def list_models(request):
    return Response({
        "models": ["distilbert-base-uncased","bert-base-uncased","roberta-base","albert-base-v2"]
    })

@api_view(["GET"])
def list_tasks(request):
    return Response({
        "tasks": ["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"]
    })

@api_view(["GET"])
def list_compressors(request):
    return Response({"compressors": list(COMPRESSOR_REGISTRY.keys())})

@api_view(["GET"])
def list_methods(request):
    return Response({"methods": ["baseline","lora","rdrop","r3f","fisher_lora","fisher_gelora","llrd"]})

@api_view(["GET"])
def list_presets(request):
    return Response({"presets": ["fast","accurate","extreme_compress"]})

# ---------- Experiments (containers) ----------
@api_view(["GET"])
def list_experiments(request):
    exps = Experiment.objects.all().order_by("-created_at")
    return Response(ExperimentSerializer(exps, many=True).data)

@api_view(["POST"])
def create_experiment(request):
    """Create a new experiment container."""
    data = request.data
    exp = Experiment.objects.create(
        name=data.get('name', f"Experiment_{timezone.now().timestamp()}"),
        description=data.get('description', ''),
        default_config=data.get('default_config', {}),
        tags=data.get('tags', '')
    )
    return Response(ExperimentSerializer(exp).data, status=201)

@api_view(["GET"])
def experiment_detail(request, exp_id):
    exp = get_object_or_404(Experiment, id=exp_id)
    return Response(ExperimentSerializer(exp).data)

# ---------- Training Runs ----------
@api_view(["GET"])
def list_runs(request, exp_id=None):
    """List all runs, optionally filtered by experiment."""
    if exp_id:
        runs = TrainingRun.objects.filter(experiment_id=exp_id).order_by("-created_at")
    else:
        runs = TrainingRun.objects.all().order_by("-created_at")
    return Response(TrainingRunSerializer(runs, many=True).data)

@api_view(["POST"])
def start_train(request):
    """Start a new training run."""
    data = request.data
    exp_id = data.get('experiment_id')
    if not exp_id:
        # If no experiment specified, create a default experiment container
        exp = Experiment.objects.create(
            name=f"Default_{data.get('task_name','train')}",
            default_config=data
        )
    else:
        exp = get_object_or_404(Experiment, id=exp_id)
    run = TrainingRun.objects.create(
        experiment=exp,
        name=data.get('name', f"Run_{data.get('task_name','train')}"),
        job_type='training',
        config=data,
        status='queued'
    )
    # Start background thread
    from threading import Thread
    Thread(target=run_training, args=(run,), daemon=True).start()
    return Response(TrainingRunSerializer(run).data, status=202)

@api_view(["POST"])
def start_evaluate(request):
    data = request.data
    run_id = data.get('run_id')
    run = get_object_or_404(TrainingRun, id=run_id)
    start_job(run, run_evaluation)
    return Response(TrainingRunSerializer(run).data, status=202)

@api_view(["POST"])
def start_compress(request):
    data = request.data
    run_id = data.get('run_id')
    run = get_object_or_404(TrainingRun, id=run_id)
    start_job(run, run_compression)
    return Response(TrainingRunSerializer(run).data, status=202)

@api_view(["GET"])
def job_status(request, job_id):
    """Get status of a training run (job_id is run UUID)."""
    run = get_object_or_404(TrainingRun, id=job_id)
    return Response(TrainingRunSerializer(run).data)

@api_view(["POST"])
def stop_job(request, job_id):
    """Stop a running job (run_id)."""
    run = get_object_or_404(TrainingRun, id=job_id)
    if run.status in ("queued", "running"):
        cancel_job(str(run.id))
        _update_run_status(run, "stopped")
        return Response({"status": "stopped"})
    return Response({"error": "Job not stoppable"}, status=400)

@api_view(["PATCH"])
def update_run_config(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    new_config = request.data
    run.config.update(new_config)
    run.save(update_fields=["config"])
    return Response(TrainingRunSerializer(run).data)

@api_view(["GET"])
def training_history(request, run_id):
    """Return the training history (steps, losses, eval metrics) for a run."""
    run = get_object_or_404(TrainingRun, id=run_id)
    history = run.training_history if isinstance(run.training_history, dict) else {}
    return Response(history)

@api_view(["GET"])
def training_snapshots(request, run_id):
    """Return detailed snapshots (optional, for precise plotting)."""
    run = get_object_or_404(TrainingRun, id=run_id)
    snapshots = run.snapshots.all().order_by('step')
    data = [{'step': s.step, 'train_loss': s.train_loss, 'eval_loss': s.eval_loss,
             'eval_accuracy': s.eval_accuracy, 'learning_rate': s.learning_rate}
            for s in snapshots]
    return Response(data)

# ---------- Artifacts & Downloads ----------
@api_view(["GET"])
def run_artifacts(request, run_id):
    run_dir = Path(settings.MEDIA_ROOT) / str(run_id)
    if not run_dir.exists():
        return Response({"artifacts": []})
    artifacts = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
    return Response({"artifacts": artifacts})

@api_view(["GET"])
def run_artifact_download(request, run_id, artifact_path):
    run_dir = Path(settings.MEDIA_ROOT) / str(run_id)
    file_path = run_dir / artifact_path
    if not file_path.exists() or not file_path.is_file():
        raise Http404("Artifact not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=os.path.basename(artifact_path))

@api_view(["GET"])
def run_model_download(request, run_id):
    run_dir = Path(settings.MEDIA_ROOT) / str(run_id) / "model"
    if not run_dir.exists():
        raise Http404("Model directory not found")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    shutil.make_archive(tmp.name.replace(".zip", ""), "zip", run_dir)
    return FileResponse(open(tmp.name, "rb"), as_attachment=True, filename="model.zip")

# ---------- System monitoring (unchanged) ----------
@api_view(["GET"])
def gpu_info(request):
    info = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
        info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
        info["utilization"] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
    return Response(info)

@api_view(["GET"])
def system_status(request):
    return Response({
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
    })

# ---------- Paper Generation (adapted for runs) ----------
@api_view(["GET"])
def generate_paper(request, run_id):
    """Generate LaTeX paper for a training run, embed loss plot if exists."""
    run = get_object_or_404(TrainingRun, id=run_id)
    template_path = Path(__file__).parent / "templates" / "paper_template.tex"
    if not template_path.exists():
        return Response({"error": "Template not found"}, status=500)
    template = template_path.read_text()

    config = run.config
    metrics = run.final_metrics
    task = config.get("task_name", "mrpc")
    model_name = config.get("model_name", "distilbert-base-uncased")
    attention = config.get("attention_type", "full")
    method = config.get("method", "baseline")
    epochs = config.get("epochs", "N/A")
    batch_size = config.get("batch_size", "N/A")
    lr = config.get("learning_rate", "N/A")
    max_len = config.get("max_seq_length", "N/A")
    rank = config.get("compression_rank", config.get("rank", "N/A"))

    loss = metrics.get("eval_loss", metrics.get("loss", "N/A"))
    if isinstance(loss, float): loss = f"{loss:.4f}"
    accuracy = metrics.get("eval_accuracy", metrics.get("accuracy", "N/A"))
    if isinstance(accuracy, float): accuracy = f"{accuracy:.4f}"
    f1 = metrics.get("eval_f1", metrics.get("f1", "N/A"))
    if isinstance(f1, float): f1 = f"{f1:.4f}"
    matthews = metrics.get("eval_matthews_correlation", metrics.get("matthews_correlation", "N/A"))
    if isinstance(matthews, float): matthews = f"{matthews:.4f}"

    # Loss plot from media directory
    loss_plot_path = Path(settings.MEDIA_ROOT) / str(run_id) / "loss_plot.png"
    if loss_plot_path.exists():
        figure_block = f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{{{loss_plot_path.as_posix()}}}
    \\caption{{Training and evaluation loss curve.}}
    \\label{{fig:loss}}
\\end{{figure}}
"""
    else:
        figure_block = ""

    replacements = {
        "\\TASKNAME": task.upper(),
        "\\ATTENTION": attention,
        "\\MODELNAME": model_name,
        "\\METHOD": method,
        "\\EPOCHS": str(epochs),
        "\\BATCHSIZE": str(batch_size),
        "\\LEARNINGRATE": str(lr),
        "\\MAXSEQLEN": str(max_len),
        "\\RANK": str(rank),
        "\\LOSS": loss,
        "\\ACCURACY": accuracy,
        "\\FONE": f1,
        "\\MATTHEWS": matthews,
        "\\DATE": timezone.now().strftime("%B %d, %Y"),
        "\\LOSSFIGURE": figure_block,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)

    tmp_dir = tempfile.mkdtemp()
    tex_path = os.path.join(tmp_dir, "paper.tex")
    pdf_path = os.path.join(tmp_dir, "paper.pdf")
    with open(tex_path, "w") as f:
        f.write(template)

    pdf_ok = False
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp_dir, tex_path],
            capture_output=True, text=True, timeout=30, check=False
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp_dir, tex_path],
            capture_output=True, text=True, timeout=30, check=False
        )
        if os.path.exists(pdf_path):
            pdf_ok = True
    except Exception as e:
        print(f"PDF compilation error: {e}")

    # Save .tex file in the run's results folder
    try:
        results_base = Path("results")
        if results_base.exists():
            exp_folders = sorted(results_base.glob(f"*{str(run.id)[:8]}*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if exp_folders:
                dest_tex = exp_folders[0] / "paper_source.tex"
                with open(dest_tex, "w") as f:
                    f.write(template)
                print(f"Saved LaTeX source to {dest_tex}")
    except Exception:
        pass

    if pdf_ok:
        response = FileResponse(open(pdf_path, "rb"), as_attachment=True, filename=f"paper_{run_id}.pdf")
        threading.Timer(10, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)).start()
        return response
    else:
        response = HttpResponse(template, content_type="application/x-latex")
        response["Content-Disposition"] = f"attachment; filename=paper_{run_id}.tex"
        return response

# ---------- PDF Preview (public, for iframe) ----------
@api_view(['GET'])
@authentication_classes([])
@permission_classes([AllowAny])
@csrf_exempt
@xframe_options_exempt
def paper_preview_pdf(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    config = run.config
    metrics = run.final_metrics
    task = config.get("task_name", "mrpc")
    model_name = config.get("model_name", "distilbert-base-uncased")
    attention = config.get("attention_type", "full")
    method = config.get("method", "baseline")
    epochs = config.get("epochs", "N/A")
    batch_size = config.get("batch_size", "N/A")
    lr = config.get("learning_rate", "N/A")
    max_len = config.get("max_seq_length", "N/A")
    rank = config.get("compression_rank", config.get("rank", "N/A"))

    loss = metrics.get("eval_loss", metrics.get("loss", "N/A"))
    if isinstance(loss, float): loss = f"{loss:.4f}"
    accuracy = metrics.get("eval_accuracy", metrics.get("accuracy", "N/A"))
    if isinstance(accuracy, float): accuracy = f"{accuracy:.4f}"
    f1 = metrics.get("eval_f1", metrics.get("f1", "N/A"))
    if isinstance(f1, float): f1 = f"{f1:.4f}"
    matthews = metrics.get("eval_matthews_correlation", metrics.get("matthews_correlation", "N/A"))
    if isinstance(matthews, float): matthews = f"{matthews:.4f}"

    latex_content = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage[margin=1in]{{geometry}}
\\title{{Experiment Report: {task.upper()} with {attention} attention}}
\\author{{Efficient Transformer Toolkit}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
\\section*{{Experiment Configuration}}
\\begin{{tabular}}{{ll}}
Model: & {model_name} \\\\
Task: & {task.upper()} \\\\
Attention type: & {attention} \\\\
Training method: & {method} \\\\
Epochs: & {epochs} \\\\
Batch size: & {batch_size} \\\\
Learning rate: & {lr} \\\\
Max sequence length: & {max_len} \\\\
Compression rank: & {rank} \\\\
\\end{{tabular}}
\\section*{{Final Metrics}}
\\begin{{tabular}}{{ll}}
Loss: & {loss} \\\\
Accuracy: & {accuracy} \\\\
F1 score: & {f1} \\\\
Matthews correlation: & {matthews} \\\\
\\end{{tabular}}
\\end{{document}}
"""
    tmp_dir = tempfile.mkdtemp()
    tex_path = os.path.join(tmp_dir, "paper.tex")
    pdf_path = os.path.join(tmp_dir, "paper.pdf")
    with open(tex_path, "w") as f:
        f.write(latex_content)

    pdf_ok = False
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp_dir, tex_path],
                       capture_output=True, text=True, timeout=30, check=False)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp_dir, tex_path],
                       capture_output=True, text=True, timeout=30, check=False)
        if os.path.exists(pdf_path):
            pdf_ok = True
    except Exception:
        pass

    if pdf_ok:
        response = FileResponse(open(pdf_path, "rb"), content_type="application/pdf")
        threading.Timer(10, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)).start()
        return response
    else:
        return Response({"error": "PDF generation failed (pdflatex not installed?)"}, status=500)

# ---------- HTML Report (web page) ----------
def experiment_report(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    config = run.config
    metrics = run.final_metrics
    context = {
        'run': run,
        'config': config,
        'metrics': metrics,
        'task': config.get('task_name', 'mrpc'),
        'model_name': config.get('model_name', 'distilbert-base-uncased'),
        'attention': config.get('attention_type', 'full'),
        'method': config.get('method', 'baseline'),
        'epochs': config.get('epochs', 'N/A'),
        'batch_size': config.get('batch_size', 'N/A'),
        'learning_rate': config.get('learning_rate', 'N/A'),
        'max_seq_length': config.get('max_seq_length', 'N/A'),
        'rank': config.get('compression_rank', config.get('rank', 'N/A')),
        'loss': metrics.get('eval_loss', metrics.get('loss', 'N/A')),
        'accuracy': metrics.get('eval_accuracy', metrics.get('accuracy', 'N/A')),
        'f1': metrics.get('eval_f1', metrics.get('f1', 'N/A')),
        'matthews': metrics.get('eval_matthews_correlation', metrics.get('matthews_correlation', 'N/A')),
    }
    return render(request, 'web_dashboard/experiment_report.html', context)

# ---------- Authentication ----------
@api_view(["POST"])
@permission_classes([AllowAny])
def register_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    if not username or not password:
        return Response({"error": "Username and password required"}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)
    user = User.objects.create_user(username=username, password=password)
    token = Token.objects.create(user=user)
    return Response({"token": token.key, "username": user.username}, status=201)

@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(request, username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "username": user.username})
    return Response({"error": "Invalid credentials"}, status=400)

@api_view(["POST"])
def logout_view(request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Token "):
        key = auth_header[6:]
        Token.objects.filter(key=key).delete()
    return Response({"status": "logged out"})

# ---------- Evaluation for a run ----------
@api_view(["POST"])
def evaluate_job(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    run_evaluation(run, stop_event=None)
    metrics = run.final_metrics
    return Response({"metrics": metrics})

# ---------- Preview (HTML modal) ----------
@api_view(["GET"])
def preview_paper(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    config = run.config
    metrics = run.final_metrics
    task = config.get("task_name", "mrpc")
    model_name = config.get("model_name", "distilbert-base-uncased")
    attention = config.get("attention_type", "full")
    method = config.get("method", "baseline")
    epochs = config.get("epochs", "N/A")
    batch_size = config.get("batch_size", "N/A")
    lr = config.get("learning_rate", "N/A")
    max_len = config.get("max_seq_length", "N/A")
    rank = config.get("compression_rank", config.get("rank", "N/A"))

    loss = metrics.get("eval_loss", metrics.get("loss", "N/A"))
    if isinstance(loss, float): loss = f"{loss:.4f}"
    accuracy = metrics.get("eval_accuracy", metrics.get("accuracy", "N/A"))
    if isinstance(accuracy, float): accuracy = f"{accuracy:.4f}"
    f1 = metrics.get("eval_f1", metrics.get("f1", "N/A"))
    if isinstance(f1, float): f1 = f"{f1:.4f}"
    matthews = metrics.get("eval_matthews_correlation", metrics.get("matthews_correlation", "N/A"))
    if isinstance(matthews, float): matthews = f"{matthews:.4f}"

    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; background: #1e293b; color: #e2e8f0; padding: 20px; border-radius: 16px;">
        <h2 style="color: #60a5fa;">📄 Experiment Report: {task.upper()}</h2>
        <hr style="border-color: #334155;">
        <h3>⚙️ Configuration</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr><td style="padding: 6px 0;"><strong>Model:</strong></td><td>{model_name}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Task:</strong></td><td>{task.upper()}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Attention:</strong></td><td>{attention}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Method:</strong></td><td>{method}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Epochs:</strong></td><td>{epochs}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Batch size:</strong></td><td>{batch_size}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Learning rate:</strong></td><td>{lr}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Max length:</strong></td><td>{max_len}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Compression rank:</strong></td><td>{rank}</td></tr>
        </table>
        <h3>📊 Final Metrics</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr><td style="padding: 6px 0;"><strong>Loss:</strong></td><td>{loss}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Accuracy:</strong></td><td>{accuracy}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>F1 score:</strong></td><td>{f1}</td></tr>
            <tr><td style="padding: 6px 0;"><strong>Matthews corr.:</strong></td><td>{matthews}</td></tr>
        </table>
        <p style="margin-top: 20px; font-size: 0.85rem; color: #94a3b8;">Generated by Efficient Transformer Toolkit</p>
    </div>
    """
    return Response({"html": html})