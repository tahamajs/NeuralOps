from autonomous_research.models import PipelineTemplate
from django.core.management.base import BaseCommand

TEMPLATES = [
    {
        "name": "Standard Research Pipeline",
        "description": "Complete pipeline from literature review to manuscript",
        "category": "general",
        "steps_config": [
            {
                "step_type": "literature_search",
                "title": "Search Literature",
                "description": "Search for relevant papers",
            },
            {
                "step_type": "paper_analysis",
                "title": "Analyze Papers",
                "description": "Extract key findings",
            },
            {
                "step_type": "hypothesis_generation",
                "title": "Generate Hypothesis",
                "description": "Formulate research hypothesis",
            },
            {
                "step_type": "experiment_design",
                "title": "Design Experiment",
                "description": "Design experiments to test hypothesis",
            },
            {
                "step_type": "experiment_execution",
                "title": "Run Experiments",
                "description": "Execute the designed experiments",
            },
            {
                "step_type": "result_analysis",
                "title": "Analyze Results",
                "description": "Analyze experiment results",
            },
            {
                "step_type": "manuscript_section",
                "title": "Draft Manuscript",
                "description": "Write the research manuscript",
            },
        ],
    },
    {
        "name": "Quick Hypothesis Test",
        "description": "Rapid hypothesis generation and testing",
        "category": "quick",
        "steps_config": [
            {
                "step_type": "literature_search",
                "title": "Quick Literature Scan",
                "description": "Scan recent papers",
            },
            {
                "step_type": "hypothesis_generation",
                "title": "Generate Hypothesis",
                "description": "Formulate a testable hypothesis",
            },
            {
                "step_type": "experiment_design",
                "title": "Design Quick Experiment",
                "description": "Design a minimal experiment",
            },
            {
                "step_type": "experiment_execution",
                "title": "Run Experiment",
                "description": "Execute experiment",
            },
            {
                "step_type": "result_analysis",
                "title": "Quick Analysis",
                "description": "Analyze and report results",
            },
        ],
    },
]


class Command(BaseCommand):
    help = "Seed default pipeline templates"

    def handle(self, *args, **options):
        for tpl in TEMPLATES:
            PipelineTemplate.objects.get_or_create(
                name=tpl["name"],
                defaults=tpl,
            )
        self.stdout.write(self.style.SUCCESS("Pipeline templates created"))
