def create_config_snapshot(job: Job, change_desc: str = None, user=None):
    """Create a versioned snapshot of the current job configuration"""
    latest = job.config_versions.order_by("-version_number").first()
    new_version = (latest.version_number + 1) if latest else 1

    config_snapshot = {
        "model_name": job.model_name,
        "task_name": job.task_name,
        "attention_type": job.attention_type,
        "method": job.method,
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "learning_rate": job.learning_rate,
        "max_seq_length": job.max_seq_length,
        "compression_rank": job.compression_rank,
        "gpu_requirements": job.gpu_requirements,
        "priority": job.priority,
        "notes": job.notes,
    }

    version = VersionedConfig.objects.create(
        run=job,
        version_number=new_version,
        config_snapshot=config_snapshot,
        change_description=change_desc or "Initial configuration",
        previous_version=latest,
        changed_by=user,
    )

    return version


def create_config_snapshot(job: Job, change_desc: str = None, user=None):
    latest = job.config_versions.order_by("-version_number").first()
    new_version = (latest.version_number + 1) if latest else 1
    config_snapshot = {
        "model_name": job.model_name,
        "task_name": job.task_name,
        "attention_type": job.attention_type,
        "method": job.method,
        # ... include all relevant fields
    }
    version = VersionedConfig.objects.create(
        run=job,
        version_number=new_version,
        config_snapshot=config_snapshot,
        change_description=change_desc or "Initial configuration",
        previous_version=latest,
        changed_by=user,
    )
    return version


try:
    from git import Repo
except ImportError:
    Repo = None


def capture_git_info(job: Job):
    if Repo is None:
        return
    try:
        repo = Repo(".", search_parent_directories=True)
        job.git_commit_hash = repo.head.commit.hexsha
        job.git_branch = repo.active_branch.name
        job.git_dirty = repo.is_dirty(untracked_files=True)
        if repo.remotes:
            job.git_repository = repo.remotes.origin.url
        job.save()
    except Exception as e:
        print(f"Git info capture failed: {e}")


def capture_environment(job: Job):
    import subprocess
    import sys

    job.python_version = sys.version
    pip_freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    job.pip_freeze = pip_freeze.stdout
    job.save()
