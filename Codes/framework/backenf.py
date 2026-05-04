import subprocess
import mlflow
from git import Repo

def get_git_info():
    """Capture current Git commit hash, branch, and dirty state"""
    try:
        repo = Repo('.', search_parent_directories=True)
        commit_hash = repo.head.commit.hexsha
        branch = repo.active_branch.name
        
        # Check for uncommitted changes
        is_dirty = repo.is_dirty(untracked_files=True)
        
        # Get repository URL
        remote_url = None
        if repo.remotes:
            remote_url = repo.remotes.origin.url
        
        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'dirty': is_dirty,
            'repository': remote_url
        }
    except Exception as e:
        print(f"Could not get Git info: {e}")
        return None

def get_data_version():
    """Capture DVC data version (if using DVC)"""
    try:
        import dvc.api
        # Get the current DVC commit hash
        result = subprocess.run(['dvc', 'status', '--json'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Extract DVC commit hash from .dvc files
            return data.get('commit', None)
    except Exception:
        pass
    return None

# In your training function:
git_info = get_git_info()
if git_info:
    # Log to MLflow
    mlflow.set_tag("git_commit", git_info['commit_hash'])
    mlflow.set_tag("git_branch", git_info['branch'])
    mlflow.set_tag("git_dirty", str(git_info['dirty']))
    mlflow.log_param("git_commit_hash", git_info['commit_hash'])
    
    # Also log warning if dirty
    if git_info['dirty']:
        print("⚠️ WARNING: Uncommitted changes detected. Reproducibility may be compromised.")

# Log environment
import sys, pkg_resources
mlflow.set_tag("python_version", sys.version)
mlflow.log_text("\n".join([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]), "requirements.txt")





# In your training function
from jobscheduler.logger import CustomRunLogger

logger = CustomRunLogger(run_id=str(job.id))
for epoch in range(epochs):
    loss = train_one_epoch()
    logger.log_scalar('train_loss', loss, step=epoch)
    # Log an image, e.g., confusion matrix
    if epoch % 10 == 0:
        fig = plot_confusion_matrix(...)
        logger.log_figure('confusion_matrix', fig, step=epoch)