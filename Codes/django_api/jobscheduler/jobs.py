def create_config_snapshot(job: Job, change_desc: str = None, user=None):
    """Create a versioned snapshot of the current job configuration"""
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
        'priority': job.priority,
        'notes': job.notes
    }
    
    version = VersionedConfig.objects.create(
        run=job,
        version_number=new_version,
        config_snapshot=config_snapshot,
        change_description=change_desc or f"Initial configuration",
        previous_version=latest,
        changed_by=user
    )
    
    return version