import os, tempfile, shutil, logging
from huggingface_hub import HfApi, create_repo, upload_folder
from django.conf import settings
from transformers import AutoConfig

logger = logging.getLogger(__name__)

class HuggingFacePublisher:
    def __init__(self, model_version, token=None):
        self.version = model_version
        self.token = token or settings.HUGGINGFACE_TOKEN
        self.api = HfApi()

    def _generate_repo_name(self):
        run = self.version.run
        base = run.model_name.split('/')[-1]
        return f"{settings.HUGGINGFACE_USERNAME}/{base}-{run.task_name}-v{self.version.version_number}"

    def publish(self):
        repo_name = self._generate_repo_name()
        create_repo(repo_name, token=self.token, exist_ok=True)

        # Prepare the model directory (decompressed if needed)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self.version.artifact_path
            if self.version.compression_method == 'gzip':
                from .services import _decompress_gzip
                _decompress_gzip(self.version.compressed_artifact_path, tmpdir)
                source_dir = tmpdir
            elif self.version.compression_method != 'none':
                # Framework compressed – it's already a directory
                source_dir = self.version.compressed_artifact_path

            # Copy all files to tmpdir (if not already there)
            if source_dir != tmpdir:
                for item in os.listdir(source_dir):
                    s = os.path.join(source_dir, item)
                    d = os.path.join(tmpdir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)

            # Generate model card
            self._generate_model_card(tmpdir)

            # Upload
            upload_folder(folder_path=tmpdir, repo_id=repo_name, token=self.token)

        self.version.is_published_to_hub = True
        self.version.hub_repo_url = f"https://huggingface.co/{repo_name}"
        self.version.save(update_fields=['is_published_to_hub', 'hub_repo_url'])
        return self.version.hub_repo_url

    def _generate_model_card(self, dir_path):
        run = self.version.run
        metrics = self.version.metrics
        card = f"""---
language: en
tags:
- {run.task_name}
- {run.attention_type}
- transformer
metrics:
- name: accuracy
  value: {metrics.get('eval_accuracy', 'N/A')}
- name: f1
  value: {metrics.get('eval_f1', 'N/A')}
- name: loss
  value: {metrics.get('eval_loss', 'N/A')}
---

# {run.model_name} – {run.task_name.upper()}

This model was trained using the Transformer Pro Toolkit.

## Training Configuration
- **Base Model**: {run.model_name}
- **Attention Type**: {run.attention_type}
- **Training Method**: {run.method}
- **Epochs**: {run.epochs}
- **Batch Size**: {run.batch_size}
- **Learning Rate**: {run.learning_rate}
- **Max Sequence Length**: {run.max_seq_length}

## Performance
- **Accuracy**: {metrics.get('eval_accuracy', 'N/A')}
- **F1 Score**: {metrics.get('eval_f1', 'N/A')}
- **Loss**: {metrics.get('eval_loss', 'N/A')}

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_name}")

with open(os.path.join(dir_path, "README.md"), "w") as f:
f.write(card)

"""