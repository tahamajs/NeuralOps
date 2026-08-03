import json
import uuid

from django.conf import settings
from django.db import models


class ResearchProject(models.Model):
    """A container linking all research assets."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_projects",
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    collaborators = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name="collab_projects"
    )
    notes = models.ManyToManyField(
        "notes.ResearcherNote", blank=True, related_name="projects"
    )
    experiments = models.ManyToManyField(
        "experiment.TrainingRun", blank=True, related_name="projects"
    )
    papers = models.ManyToManyField(
        "paper.PaperProject", blank=True, related_name="research_projects"
    )
    models_marketplace = models.ManyToManyField(
        "model_versioning.ModelVersion", blank=True, related_name="projects"
    )
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)


class ProjectKnowledgeEdge(models.Model):
    """A typed edge between two assets in a project."""

    source_type = models.CharField(
        max_length=50
    )  # 'note', 'experiment', 'paper', 'model'
    source_id = models.UUIDField()
    target_type = models.CharField(max_length=50)
    target_id = models.UUIDField()
    relationship = models.CharField(max_length=50, default="related")
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)


class ProjectAnnotation(models.Model):
    """A comment anchored to any asset in a project."""

    project = models.ForeignKey(
        ResearchProject, on_delete=models.CASCADE, related_name="annotations"
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    asset_type = models.CharField(max_length=50)
    asset_id = models.UUIDField()
    content = models.TextField()
    resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)


class AgentTeam:
    """Orchestrates a team of specialized AI agents."""

    ROLES = {
        "generator": "Generate creative hypotheses and experimental designs.",
        "critic": "Critically evaluate proposals for flaws and suggest improvements.",
        "researcher": "Search for relevant literature and provide evidence.",
        "synthesizer": "Combine findings into coherent summaries and reports.",
        "executor": "Execute experiments by calling platform APIs (launch training, etc.).",
    }

    def __init__(self, team_config: dict, project_id: str):
        self.config = team_config
        self.project_id = project_id
        self.conversation_history = []

    def _get_project_context(self) -> str:
        return f"Project: {self.project_id}"

    def run_cycle(self, task: str):
        results = {}
        for role, model_id in self.config["roles"].items():
            system_prompt = f"You are a {role} agent. {self.ROLES[role]}\nProject context: {self._get_project_context()}"
            results[role] = f"Response for {role} on task: {task}"
        return results
