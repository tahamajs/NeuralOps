import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite_parallel.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_git_block = """        # Auto-commit to git
        with git_lock:
            try:
                import subprocess
                subprocess.run(["git", "add", "-f", str(RESULTS_DIR)], cwd=str(PROJECT_DIR), check=True)
                subprocess.run(["git", "commit", "-m", f"Auto-commit: {attn} on {task}"], cwd=str(PROJECT_DIR), check=False)
                subprocess.run(["git", "push"], cwd=str(PROJECT_DIR), check=False)
            except Exception as e:
                logger.warning(f"Git push failed: {e}")"""

new_git_block = """        # Auto-commit to git after every single finished run
        with git_lock:
            try:
                import subprocess
                msg = f"feat(benchmark): [{idx}/{total_tasks}] {attn} on {task} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Loss: {loss:.4f}" if 'acc' in locals() else f"fix(benchmark): [{idx}/{total_tasks}] {attn} on {task} telemetry update"
                subprocess.run(["git", "add", "-f", str(RESULTS_DIR), str(MASTER_LOG)], cwd=str(PROJECT_DIR), check=True)
                commit_res = subprocess.run(["git", "commit", "-m", msg], cwd=str(PROJECT_DIR), capture_output=True, text=True, check=False)
                if commit_res.returncode == 0:
                    logger.info(f"Committed run result: {msg}")
                # Pull rebase then push to guarantee delivery
                subprocess.run(["git", "pull", "origin", "main", "--rebase"], cwd=str(PROJECT_DIR), capture_output=True, check=False)
                push_res = subprocess.run(["git", "push", "origin", "main"], cwd=str(PROJECT_DIR), capture_output=True, text=True, check=False)
                if push_res.returncode == 0:
                    logger.info(f"Successfully pushed results for {attn} on {task} to GitHub main branch!")
                else:
                    logger.warning(f"Git push warning: {push_res.stderr.strip()}")
            except Exception as e:
                logger.warning(f"Git commit/push error: {e}")"""

content = content.replace(old_git_block, new_git_block)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated git auto-commit logic!")
