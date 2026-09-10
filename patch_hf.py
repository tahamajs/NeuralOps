import re

filepath = "D:/SVD_Project/auto_upload_hf_results.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the success print with a print + delete
old_success = 'print(f"[{datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}] [HF-Sync] SUCCESS: weights.pt -> {repo_id}")'
new_success = """print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] SUCCESS: weights.pt -> {repo_id}")
                try:
                    wf.unlink()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] DELETED local weights.pt to save space.")
                except Exception as del_err:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Could not delete local weights.pt: {del_err}")"""

if "DELETED local weights" not in content:
    content = content.replace(old_success, new_success)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched auto_upload_hf_results.py to delete files after upload.")
