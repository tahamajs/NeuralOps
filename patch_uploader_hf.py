import re

with open("auto_upload_hf_results.py", "r") as f:
    code = f.read()

replacement = """
                import subprocess
                res = subprocess.run(["hf", "upload", repo_id, str(wf), "weights.pt"], capture_output=True)
                if res.returncode != 0:
                    raise Exception(res.stderr.decode("utf-8"))
"""
code = code.replace("""
                api.upload_file(
                    path_or_fileobj=str(wf),
                    path_in_repo="weights.pt",
                    repo_id=repo_id,
                    commit_message=f"Upload model weights for {model_name}-{mod_name} (recovery_steps=50)",
                )
""", replacement)

with open("auto_upload_hf_results.py", "w") as f:
    f.write(code)

