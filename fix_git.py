import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

git_logic = """
            except Exception as e:
                elapsed = time.time() - t0
                log_master(f"[{idx}/{total}] ⚠️ NOTICE: attn={attn} | task={task} | Error: {e} | Time: {elapsed:.1f}s")

            # Push results to git for live github pages
            try:
                import subprocess
                subprocess.run(["git", "add", str(RESULTS_DIR.parent)], cwd=str(PROJECT_DIR), check=True)
                subprocess.run(["git", "commit", "-m", f"Auto-commit: {attn} on {task}"], cwd=str(PROJECT_DIR), check=False)
                subprocess.run(["git", "push"], cwd=str(PROJECT_DIR), check=False)
            except Exception as e:
                log_master(f"Git push failed: {e}")
"""

content = re.sub(
    r"except Exception as e:\s*elapsed = time\.time\(\) - t0\s*log_master[^>]*Time:[^\n]*\n",
    git_logic,
    content
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
