def get_dvc_data_commit():
    """Get the DVC commit hash that points to the current dataset version"""
    try:
        # Method 1: Check .dvc files in Git
        result = subprocess.run(['git', 'log', '-1', '--format=%H', '--', '*.dvc'], 
                               capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        
        # Method 2: Use dvc api
        import dvc.api
        with dvc.api.open('data/processed', repo='.'):
            pass  # This will raise if not versioned
        return "dvc-versioned"
    except Exception as e:
        print(f"Could not determine DVC data version: {e}")
        return None

# In training script
dvc_commit = get_dvc_data_commit()
if dvc_commit:
    mlflow.set_tag("data_git_commit", dvc_commit)
    mlflow.log_param("data_version", dvc_commit)