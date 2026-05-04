# jobscheduler/version_utils.py
import sys
import subprocess
from django.conf import settings
from git import Repo
from .models import VersionedConfig

def capture_git_info(job):
    """
    Capture Git commit hash, branch, dirty state, and remote URL.
    """
    try:
        repo = Repo(settings.BASE_DIR, search_parent_directories=True)
        job.git_commit_hash = repo.head.commit.hexsha
        job.git_branch = repo.active_branch.name
        job.git_dirty = repo.is_dirty(untracked_files=True)
        if repo.remotes:
            job.git_repository = repo.remotes.origin.url
        job.save(update_fields=['git_commit_hash', 'git_branch', 'git_dirty', 'git_repository'])
    except Exception as e:
        print(f"Git info capture failed for job {job.id}: {e}")

def capture_environment(job):
    """
    Capture Python version and pip freeze output.
    """
    try:
        job.python_version = sys.version
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                capture_output=True, text=True)
        job.pip_freeze = result.stdout
        job.save(update_fields=['python_version', 'pip_freeze'])
    except Exception as e:
        print(f"Environment capture failed for job {job.id}: {e}")

def capture_data_version(job):
    """
    Capture DVC data version (if DVC is used). 
    This is optional; gracefully handles missing DVC.
    """
    try:
        # Get the commit hash of the latest .dvc file change in Git
        result = subprocess.run(['git', 'log', '-1', '--format=%H', '--', '*.dvc'],
                                capture_output=True, text=True, cwd=settings.BASE_DIR)
        if result.returncode == 0 and result.stdout.strip():
            job.data_git_commit_id = result.stdout.strip()
        # Try to get DVC commit hash from 'dvc status --json'
        dvc_result = subprocess.run(['dvc', 'status', '--json'],
                                    capture_output=True, text=True, cwd=settings.BASE_DIR)
        if dvc_result.returncode == 0:
            import json
            data = json.loads(dvc_result.stdout)
            job.dvc_data_hash = data.get('commit', '')
        job.save(update_fields=['data_git_commit_id', 'dvc_data_hash'])
    except Exception as e:
        print(f"Data version capture failed for job {job.id}: {e}")

def create_config_snapshot(job, change_description="Initial configuration", user=None):
    """
    Create a versioned snapshot of the job's current configuration.
    """
    latest = job.config_versions.order_by('-version_number').first()
    new_version = (latest.version_number + 1) if latest else 1

    config_snapshot = {
        'model_name': job.model_name,
        'task_name': job.task_name,
        'attention_type': job.attention_type,
        'method': job.method,
        'epochs': job.epochs,
        'batch_size': job.batch_size,
        'learning_rate': job.learning_rate,
        'max_seq_length': job.max_seq_length,
        'compression_rank': job.compression_rank,
        'gpu_requirements': job.gpu_requirements,
        'memory_requirements': job.memory_requirements,
        'priority': job.priority,
        'estimated_duration': job.estimated_duration,
        'notes': job.notes,
    }

    version = VersionedConfig.objects.create(
        run=job,
        version_number=new_version,
        config_snapshot=config_snapshot,
        change_description=change_description,
        previous_version=latest,
        changed_by=user
    )
    return version