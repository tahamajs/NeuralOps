import subprocess, json, sys, importlib.metadata
try:
    from git import Repo
except ImportError:
    Repo = None
try:
    import mlflow
except ImportError:
    mlflow = None

def get_git_info():
    if Repo is None:
        return None
    try:
        repo = Repo('.', search_parent_directories=True)
        return {
            'commit_hash': repo.head.commit.hexsha,
            'branch': repo.active_branch.name,
            'dirty': repo.is_dirty(untracked_files=True),
            'repository': repo.remotes.origin.url if repo.remotes else None,
        }
    except Exception as e:
        print(f"Git error: {e}")
        return None

def get_dvc_data_commit():
    try:
        result = subprocess.run(['git', 'log', '-1', '--format=%H', '--', '*.dvc'],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception:
        return None

def log_environment_to_mlflow():
    mlflow.set_tag("python_version", sys.version)
    reqs = "\n".join([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
    mlflow.log_text(reqs, "requirements.txt")