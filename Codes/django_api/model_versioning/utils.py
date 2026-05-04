# model_versioning/utils.py
import os
import tempfile
import shutil
import logging
from django.conf import settings
from django.utils import timezone
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoConfig, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class HuggingFacePublisher:
    def __init__(self, model_version, token=None, username=None):
        self.version = model_version
        self.token = token or getattr(settings, 'HUGGINGFACE_TOKEN', None)
        self.username = username or getattr(settings, 'HUGGINGFACE_USERNAME', None)
        if not self.token or not self.username:
            raise ValueError("HUGGINGFACE_TOKEN and HUGGINGFACE_USERNAME must be set in settings")
        self.api = HfApi()
        self.repo_name = f"{self.username}/{self._generate_repo_name()}"
    
    def _generate_repo_name(self):
        run = self.version.run
        model_slug = run.model_name.replace('/', '-')
        return f"{model_slug}-{run.task_name}-v{self.version.version_number}"
    
    def prepare_model_for_hub(self, output_dir):
        """Convert saved model to Hugging Face format."""
        from transformers import AutoConfig, AutoModelForSequenceClassification
        # Load model from artifact path
        model_path = os.path.join(self.version.artifact_path, 'model.pth')
        config_path = self.version.artifact_path
        
        # For simplicity, we assume the model was saved using transformers save_pretrained
        # If not, we need to reconstruct. Here we assume the artifact_path contains a full HF model.
        model = AutoModelForSequenceClassification.from_pretrained(self.version.artifact_path)
        
        # Add metadata
        config = AutoConfig.from_pretrained(self.version.artifact_path)
        config.task_specific_params = {
            "task": self.version.run.task_name,
            "attention_type": self.version.run.attention_type,
            "training_method": self.version.run.method
        }
        model.config = config
        
        model.save_pretrained(output_dir)
        self._generate_model_card(output_dir)
        return output_dir
    
    def _generate_model_card(self, output_dir):
        """Create README.md with model details."""
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

# {run.model_name} - {run.task_name.upper()}

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

tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("{self.repo_name}")