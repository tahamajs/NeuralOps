from django.apps import AppConfig
from django.db.models.signals import post_save
from django.conf import settings
import json

class AgentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'agents'

    def ready(self):
        # Wire cross‑app signals
        from experiment.models import TrainingRun
        from agents.tools import analyze_training
        from notes.models import ResearcherNote

        def on_run_complete(sender, instance, **kwargs):
            if instance.status in ("completed", "failed"):
                analysis = analyze_training(run_id=str(instance.id))
                user_id = instance.experiment.user_id if instance.experiment else None
                title = f"Training {'Complete' if instance.status == 'completed' else 'Failed'}: {instance.name or instance.id}"
                ResearcherNote.objects.create(
                    author_id=user_id,
                    title=title,
                    content=json.dumps(analysis, indent=2),
                    tags=["auto-generated", "training-report", instance.status],
                    is_public=False,
                    pinned=False,
                    metadata={"run_id": str(instance.id)},
                )

        post_save.connect(on_run_complete, sender=TrainingRun, dispatch_uid="agent_on_run_complete")