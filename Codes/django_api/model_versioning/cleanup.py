# model_versioning/cleanup.py
def cleanup_old_versions(run, keep_best_k=3, keep_version_tags=['v1', 'v2']):
    """Keep only the best K versions and any tagged versions."""
    versions = run.model_versions.all()
    
    # Always keep versions with important tags
    protected = [v for v in versions if any(v.version.startswith(tag) for tag in keep_version_tags)]
    
    # Keep top-k based on validation metric
    sorted_by_metric = sorted(
        [v for v in versions if v not in protected],
        key=lambda v: v.metrics.get('eval_accuracy', 0),
        reverse=True
    )[:keep_best_k]
    
    to_delete = [v for v in versions if v not in protected and v not in sorted_by_metric]
    
    for version in to_delete:
        # Optionally move to cold storage before deletion
        if version.storage_tier == 'hot':
            # Move to cloud archive
            pass
        else:
            shutil.rmtree(version.artifact_path)
            version.delete()