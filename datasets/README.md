# 📊 Consolidated Datasets Folder

All project datasets have been consolidated into the central `/datasets` folder at the project root for consistency across the framework and external tools.

## 📁 Folder Structure

- `/datasets/glue/`: Standard GLUE benchmark tasks (CoLA, SST-2, MRPC, etc.).
- `/datasets/coco/`: COCO dataset files and scripts.
- `/datasets/udt-coco/`: Unified Data Tool annotations for COCO.
- `/datasets/vlm-selection/`: Experimental datasets for Vision-Language Models.
- `/datasets/bigbench/`:
  - `data/`: Prompting and ground truth data.
  - `tasks/`: BIG-bench task definitions.
- `/datasets/vgnlp_data/`: Evaluation data for Visual Grounding NLP tasks.
- `/datasets/truthful_qa/`: Multiple choice and generation splits for TruthfulQA.

## 🛠️ Usage in Framework

The `Codes` framework uses `framework.utils.paths` to resolve these directories dynamically:

```python
from framework.utils.paths import get_datasets_dir, get_glue_dir

# Get absolute path to /datasets
ds_path = get_datasets_dir()

# Get absolute path to /datasets/glue
glue_path = get_glue_dir()
```

## 📝 Configuration

- **Django**: The `GLUE_DATA_PATH` in `settings.py` is automatically updated to use the consolidated path.
- **Data Loaders**: All `load_glue_local` calls and `ExperimentConfig` defaults now point to this centralized location.
