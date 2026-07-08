@shared_task(bind=True, max_retries=2)
def run_sync_task(self, task_id: str):
    task = SyncTask.objects.get(id=task_id)
    task.status = "running"
    task.save()
    try:
        conn = task.connection
        if conn.provider == "github":
            # Clone/pull repo
            import os
            import subprocess
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                repo = conn.metadata.get("repo", "")
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo}.git", tmpdir],
                    check=True,
                )
                # Copy files to target
                import shutil

                for f in os.listdir(tmpdir):
                    shutil.copy(os.path.join(tmpdir, f), task.target_path)
                task.files_synced = len(os.listdir(tmpdir))
        task.status = "completed"
        task.completed_at = timezone.now()
        task.save()
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.save()
