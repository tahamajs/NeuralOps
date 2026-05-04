import os, shutil, tempfile, json, logging
from django.http import FileResponse, Http404
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Experiment
from .serializers import ExperimentSerializer
from .jobs import start_job, stop_job as do_stop, cancel_job
from .runners import run_training, run_evaluation, run_compression
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY
import psutil, torch, time, math
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from datetime import datetime
import subprocess, re, shutil, tempfile, threading
from pathlib import Path
from django.http import FileResponse, Http404, HttpResponse
from django.conf import settings
import os, json, logging
from rest_framework.authtoken.models import Token   # correct
from .runners import run_training
import json
from datetime import timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Avg
from django.utils import timezone
from django.views.decorators.cache import cache_page

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

from api.models import Experiment


logger = logging.getLogger(__name__)

# ── Config discovery ─────────────────────────────────────────────
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

# ── Start jobs ────────────────────────────────────────────────────
from threading import Thread

from threading import Thread
from .runners import run_training, run_evaluation, run_compression

from threading import Thread
from .runners import run_training

@api_view(["POST"])
def start_train(request):
    data = request.data
    exp = Experiment.objects.create(config=data, name=f"Exp_{data.get('task_name','train')}")
    Thread(target=run_training, args=(exp,), daemon=True).start()
    return Response(ExperimentSerializer(exp).data, status=202)

@api_view(["POST"])
def start_evaluate(request):
    data = request.data
    exp = Experiment.objects.create(config=data)
    start_job(exp, run_evaluation)
    return Response(ExperimentSerializer(exp).data, status=status.HTTP_202_ACCEPTED)

@api_view(["POST"])
def start_compress(request):
    data = request.data
    exp = Experiment.objects.create(config=data)
    start_job(exp, run_compression)
    return Response(ExperimentSerializer(exp).data, status=status.HTTP_202_ACCEPTED)

# ── Job status / stop / config update ─────────────────────────────
@api_view(["GET"])
def job_status(request, job_id):
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
    return Response(ExperimentSerializer(exp).data)

@api_view(["POST"])
def stop_job(request, job_id):
    """Stop a running job (alias for cancel)."""
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
    if exp.status in ("queued", "running"):
        cancel_job(str(job_id))              # sets the event in jobs.py
        exp.status = "stopped"
        exp.save()
        return Response({"status": "stopped"})
    return Response({"error": "Job not cancelable"}, status=400)

@api_view(["PATCH"])
def update_job_config(request, job_id):
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Job not found"}, status=404)
    new_config = request.data
    exp.config.update(new_config)
    exp.save(update_fields=["config"])
    return Response(ExperimentSerializer(exp).data)

# ── List experiments ─────────────────────────────────────────────
@api_view(["GET"])
def list_experiments(request):
    exps = Experiment.objects.all().order_by("-created_at")
    return Response(ExperimentSerializer(exps, many=True).data)

# ── Artifacts ─────────────────────────────────────────────────────
@api_view(["GET"])
def experiment_artifacts(request, job_id):
    exp_dir = settings.MEDIA_ROOT / str(job_id)
    if not exp_dir.exists():
        return Response({"artifacts": []})
    artifacts = [str(p.relative_to(exp_dir)) for p in exp_dir.rglob("*") if p.is_file()]
    return Response({"artifacts": artifacts})

@api_view(["GET"])
def experiment_artifact_download(request, job_id, artifact_path):
    exp_dir = settings.MEDIA_ROOT / str(job_id)
    file_path = exp_dir / artifact_path
    if not file_path.exists() or not file_path.is_file():
        raise Http404("Artifact not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=os.path.basename(artifact_path))

@api_view(["GET"])
def experiment_model_download(request, job_id):
    exp_dir = settings.MEDIA_ROOT / str(job_id) / "model"
    if not exp_dir.exists():
        raise Http404("Model directory not found")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    shutil.make_archive(tmp.name.replace(".zip", ""), "zip", exp_dir)
    return FileResponse(open(tmp.name, "rb"), as_attachment=True, filename="model.zip")

# ── System monitoring ────────────────────────────────────────────
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



import subprocess, re, shutil, tempfile, os
from django.conf import settings as django_settings

@api_view(["GET"])
def generate_paper(request, job_id):
    """Generate a LaTeX paper for an experiment and return the PDF."""
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Experiment not found"}, status=404)

    # ── 1. Load the template ──────────────────────────────────────
    template_path = Path(__file__).parent / "templates" / "paper_template.tex"
    if not template_path.exists():
        return Response({"error": "Template not found"}, status=500)
    template = template_path.read_text()

    # ── 2. Collect experiment data ─────────────────────────────────
    config = exp.config or {}
    metrics = exp.metrics or {}
    job_type = config.get("job_type", "training")

    # Build a friendly name
    experiment_name = exp.name or config.get("model_name", "Experiment")

    # Extract metrics (handle different task types)
    accuracy = metrics.get("eval_accuracy", metrics.get("eval_pearson", metrics.get("eval_matthews_correlation", "N/A")))
    if isinstance(accuracy, float):
        accuracy = f"{accuracy:.4f}"
    loss = metrics.get("eval_loss", "N/A")
    if isinstance(loss, float):
        loss = f"{loss:.4f}"
    f1 = metrics.get("eval_f1", metrics.get("eval_f1_macro", "N/A"))
    if isinstance(f1, float):
        f1 = f"{f1:.4f}"
    matthews = metrics.get("eval_matthews_correlation", "N/A")
    if isinstance(matthews, float):
        matthews = f"{matthews:.4f}"

    # Paths to artifacts (loss plot)
    loss_plot_path = f"/api/experiments/{job_id}/artifacts/loss_plot.png"
    # Use the actual file if it exists, otherwise a placeholder
    real_plot = django_settings.MEDIA_ROOT / str(job_id) / "loss_plot.png"
    if not real_plot.exists():
        # No plot available – remove the figure from the template
        template = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', template, flags=re.DOTALL)
        loss_plot_path = None
    else:
        template = template.replace("LOSS_PLOT_PATH", real_plot.as_posix())

    # ── 3. Fill the template ───────────────────────────────────────
    replacements = {
        "EXPERIMENT_NAME": experiment_name.replace("_", "\\_"),
        "AUTHOR_NAME": "Efficient Transformer Toolkit",
        "DATE": timezone.now().strftime("%B %d, %Y"),
        "ATTENTION_TYPE": str(config.get("attention_type", "N/A")).replace("_", "\\_"),
        "TASK_NAME": str(config.get("task_name", job_type)).upper(),
        "MODEL_NAME": str(config.get("model_name", "N/A")).replace("_", "\\_"),
        "COMPRESSION_RANK": str(config.get("compression_rank", config.get("rank", "N/A"))),
        "EPOCHS": str(config.get("epochs", "N/A")),
        "BATCH_SIZE": str(config.get("batch_size", "N/A")),
        "LEARNING_RATE": str(config.get("learning_rate", "N/A")),
        "MAX_SEQ_LENGTH": str(config.get("max_seq_length", "N/A")),
        "SEED": str(config.get("seed", "N/A")),
        "LOSS": loss,
        "ACCURACY": accuracy,
        "F1": f1,
        "MATTHEWS": matthews,
        "TASK_TYPE": str(config.get("task_type", "classification")),
    }

    for key, value in replacements.items():
        template = template.replace(f"{{{{{{key}}}}}}", str(value))

    # ── 4. Compile or return .tex ──────────────────────────────────
    # Try server-side compilation (requires pdflatex)
    tmp_dir = tempfile.mkdtemp()
    tex_path = os.path.join(tmp_dir, "paper.tex")

    with open(tex_path, "w") as f:
        f.write(template)

    pdf_path = os.path.join(tmp_dir, "paper.pdf")

    try:
        # Run pdflatex twice (for references)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp_dir, tex_path],
            capture_output=True, text=True, timeout=30
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp_dir, tex_path],
            capture_output=True, text=True, timeout=30
        )

        if os.path.exists(pdf_path):
            response = FileResponse(open(pdf_path, "rb"), as_attachment=True, filename=f"paper_{job_id}.pdf")
            # Cleanup after (we can't easily clean while responding, so schedule it)
            import threading
            def cleanup():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            threading.Timer(10, cleanup).start()
            return response

    except Exception as e:
        logger.warning(f"pdflatex failed: {e}")

    # ── 5. Fallback: return the .tex file ──────────────────────────
    tex_content = template.encode("utf-8")
    response = HttpResponse(tex_content, content_type="application/x-latex")
    response["Content-Disposition"] = f"attachment; filename=paper_{job_id}.tex"
    return response

# ================================================================
from django.shortcuts import render, get_object_or_404
from .models import Experiment

def experiment_detail(request, exp_id):
    experiment = get_object_or_404(Experiment, id=exp_id)
    return render(request, 'web_dashboard/experiment_detail.html', {'experiment': experiment})

# ================================================================
#  Authentication (register / login / logout)
# ================================================================
# from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token          # ← correct DRF token

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


from .runners import run_evaluation   # at the top with other imports

@api_view(["POST"])
def evaluate_job(request, job_id):
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Experiment not found"}, status=404)

    # Run evaluation (this updates exp.config['final_metrics'])
    run_evaluation(exp, stop_event=None)

    # Retrieve the metrics stored in experiment config
    metrics = exp.config.get('final_metrics', {})
    return Response({"metrics": metrics})


from django.shortcuts import render

def experiment_report(request, job_id):
    """Render a full HTML report page for the experiment."""
    from .models import Experiment
    exp = get_object_or_404(Experiment, id=job_id)
    config = exp.config or {}
    metrics = config.get('final_metrics', {})

    context = {
        'experiment': exp,
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

from django.utils import timezone

@api_view(["GET"])
def preview_paper(request, job_id):
    """Return HTML preview of the experiment paper for UI modal."""
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Experiment not found"}, status=404)

    config = exp.config or {}
    metrics = config.get('final_metrics', {})
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






from django.views.decorators.clickjacking import xframe_options_exempt
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_exempt
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny
from rest_framework.authentication import BasicAuthentication

@api_view(['GET'])
@authentication_classes([])          # ← no authentication
@permission_classes([AllowAny])      # ← allow any user
@csrf_exempt                         # ← avoid CSRF for iframe
@xframe_options_exempt               # ← removes X-Frame-Options
def paper_preview_pdf(request, job_id):
    """Return PDF inline for preview (no download, no X-Frame-Options)."""
    try:
        exp = Experiment.objects.get(id=job_id)
    except Experiment.DoesNotExist:
        return Response({"error": "Experiment not found"}, status=404)

    config = exp.config or {}
    metrics = config.get('final_metrics', {})
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

    import tempfile, subprocess, os, threading, shutil
    tmp_dir = tempfile.mkdtemp()
    tex_path = os.path.join(tmp_dir, "paper.tex")
    pdf_path = os.path.join(tmp_dir, "paper.pdf")

    with open(tex_path, "w") as f:
        f.write(latex_content)

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

    if pdf_ok:
        response = FileResponse(open(pdf_path, "rb"), content_type="application/pdf")
        threading.Timer(10, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)).start()
        return response
    else:
        return Response({"error": "PDF generation failed (pdflatex not installed?)"}, status=500)