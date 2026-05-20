import os
import tempfile
from django.conf import settings
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

def upload_note_to_hub(note, token=None):
    """
    Upload a note as a HuggingFace Dataset card or a simple Markdown file.
    The repo will be named: {username}/neuralops-note-{note_id}
    """
    token = token or getattr(settings, 'HUGGINGFACE_TOKEN', None)
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN is not set in Django settings.")

    username = getattr(settings, 'HUGGINGFACE_USERNAME', 'unknown')
    repo_id = f"{username}/neuralops-note-{note.id}"

    api = HfApi()
    # Create repo (or use existing)
    create_repo(repo_id, token=token, repo_type="space", exist_ok=True)  # or "dataset"

    # Prepare content
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write note as Markdown
        md_path = os.path.join(tmpdir, "README.md")
        with open(md_path, "w") as f:
            f.write(_note_to_markdown(note))

        # Upload
        upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
    return f"https://huggingface.co/{repo_id}"


def _note_to_markdown(note) -> str:
    """Convert a ResearcherNote to a rich Markdown string."""
    md = f"""---
title: "{note.title}"
author: "{note.author.username}"
type: "{note.note_type}"
tags: {note.tags}
created: "{note.created_at.isoformat()}"
---

{note.content}

---
*Uploaded by NeuralOps Notes*
"""
    return md