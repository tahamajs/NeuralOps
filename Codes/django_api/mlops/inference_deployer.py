"""One‑click model deployment to HF Spaces, SageMaker, etc."""

import os
import tempfile

from django.conf import settings
from huggingface_hub import HfApi, create_repo, upload_folder


def deploy_to_huggingface(model_version, config: dict):
    """Deploy a model to HuggingFace Spaces as a Gradio app."""
    token = settings.HUGGINGFACE_TOKEN
    username = settings.HUGGINGFACE_USERNAME
    repo_id = f"{username}/neuralops-endpoint-{model_version.id.hex[:8]}"

    api = HfApi(token=token)
    create_repo(
        repo_id, token=token, repo_type="space", space_sdk="gradio", exist_ok=True
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create app.py with Gradio interface
        app_py = f"""
import gradio as gr
from transformers import pipeline

pipe = pipeline('text-classification', model='{model_version.artifact_path}')

def predict(text):
    return pipe(text)

gr.Interface(fn=predict, inputs='text', outputs='label').launch()
"""
        with open(os.path.join(tmpdir, "app.py"), "w") as f:
            f.write(app_py)

        requirements = "transformers\ntorch\ngradio"
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write(requirements)

        upload_folder(
            folder_path=tmpdir, repo_id=repo_id, repo_type="space", token=token
        )

    return f"https://huggingface.co/spaces/{repo_id}"
