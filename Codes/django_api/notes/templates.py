"""
Default note templates for ML research workflows.
These are created automatically when the app starts.
"""
from .models import NoteTemplate

DEFAULT_TEMPLATES = [
    {
        'name': 'Training Run Report',
        'description': 'Standard template for documenting a training run',
        'note_type': 'training_report',
        'content_template': """## 📈 Training Run: {{model_name}} → {{task_name}}

### Configuration
- **Model:** `{{model_name}}`
- **Task:** `{{task_name}}`
- **Attention:** {{attention_type}}
- **Method:** {{method}}
- **Epochs:** {{epochs}}
- **Batch Size:** {{batch_size}}
- **Learning Rate:** {{learning_rate}}

### Results
- **Accuracy:** {{accuracy}}
- **F1 Score:** {{f1_score}}
- **Loss:** {{loss}}

### Observations
*(Add your observations here)*

### Next Steps
- [ ] 
""",
        'variables_schema': {
            'model_name': 'string', 'task_name': 'string',
            'attention_type': 'string', 'method': 'string',
            'epochs': 'number', 'batch_size': 'number',
            'learning_rate': 'number', 'accuracy': 'number',
            'f1_score': 'number', 'loss': 'number',
        },
        'is_default': True,
    },
    {
        'name': 'Compression Experiment',
        'description': 'Document a model compression experiment',
        'note_type': 'compression_report',
        'content_template': """## 🗜️ Compression Experiment: {{model_name}}

### Compression Method
- **Method:** {{compression_method}}
- **Rank/Budget:** {{rank}}
- **Pre-Compression Accuracy:** {{accuracy_before}}
- **Post-Compression Accuracy:** {{accuracy_after}}
- **Accuracy Drop:** {{accuracy_drop}}
- **Size Before:** {{size_before}} MB
- **Size After:** {{size_after}} MB
- **Compression Ratio:** {{compression_ratio}}x

### Analysis
*(Add analysis here)*

### Recommendations
- 
""",
        'variables_schema': {
            'model_name': 'string', 'compression_method': 'string',
            'rank': 'number', 'accuracy_before': 'number',
            'accuracy_after': 'number', 'accuracy_drop': 'number',
            'size_before': 'number', 'size_after': 'number',
            'compression_ratio': 'number',
        },
        'is_default': True,
    },
    {
        'name': 'Literature Review',
        'description': 'Summarize a research paper',
        'note_type': 'literature_review',
        'content_template': """## 📄 Paper Review: {{paper_title}}

**Authors:** {{authors}}  
**Year:** {{year}}  
**Link:** {{paper_url}}  
**Citations:** {{citation_count}}

### Abstract Summary
{{abstract_summary}}

### Key Contributions
- 

### Methodology
- 

### Results
- 

### Relevance to Our Work
- 

### Questions / Follow-ups
- 
""",
        'variables_schema': {
            'paper_title': 'string', 'authors': 'string', 'year': 'string',
            'paper_url': 'string', 'citation_count': 'number',
            'abstract_summary': 'string',
        },
        'is_default': True,
    },
    {
        'name': 'Hypothesis / Idea',
        'description': 'Capture a research hypothesis or idea',
        'note_type': 'hypothesis',
        'content_template': """## 💡 Hypothesis: {{title}}

### Motivation
{{motivation}}

### Hypothesis
**If we** {{action}}, **then** {{expected_result}}, **because** {{reasoning}}.

### Experiment Design
- **Independent Variable:** {{independent_var}}
- **Dependent Variable:** {{dependent_var}}
- **Controls:** {{controls}}

### Success Criteria
- 

### Related Work
- 
""",
        'variables_schema': {
            'title': 'string', 'motivation': 'string', 'action': 'string',
            'expected_result': 'string', 'reasoning': 'string',
            'independent_var': 'string', 'dependent_var': 'string',
            'controls': 'string',
        },
        'is_default': True,
    },
    {
        'name': 'Meeting Notes',
        'description': 'Research meeting minutes',
        'note_type': 'meeting_notes',
        'content_template': """## 📅 Meeting: {{meeting_title}}

**Date:** {{date}}  
**Attendees:** {{attendees}}

### Agenda
1. 

### Discussion Points
- 

### Decisions Made
- 

### Action Items
- [ ] **{{person}}**: {{task}}
""",
        'variables_schema': {
            'meeting_title': 'string', 'date': 'string',
            'attendees': 'string', 'person': 'string', 'task': 'string',
        },
        'is_default': True,
    },
]


def create_default_templates():
    """Create default templates if they don't exist."""
    for template_data in DEFAULT_TEMPLATES:
        NoteTemplate.objects.get_or_create(
            name=template_data['name'],
            is_default=True,
            defaults=template_data,
        )