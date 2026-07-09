import json

from django.apps import AppConfig
from django.db.models.signals import post_save


class AgentsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agents"

    def ready(self):
        # Wire cross‑app signals
        from agents.tools import analyze_training
        from experiment.models import TrainingRun
        from notes.models import ResearcherNote

        def on_run_complete(sender, instance, **kwargs):
            if instance.status in ("completed", "failed"):
                analysis = analyze_training(run_id=str(instance.id))
                user_id = getattr(instance.experiment, "user_id", None) if instance.experiment else None
                if not user_id:
                    from django.contrib.auth import get_user_model
                    User = get_user_model()
                    first_user = User.objects.first()
                    user_id = first_user.id if first_user else None
                title = f"Training {'Complete' if instance.status == 'completed' else 'Failed'}: {instance.name or instance.id}"
                ResearcherNote.objects.create(
                    author_id=user_id,
                    title=title,
                    content=json.dumps(analysis, indent=2),
                    tags=["auto-generated", "training-report", instance.status],
                    is_public=False,
                    is_pinned=False,
                    metadata={"run_id": str(instance.id)},
                )

        post_save.connect(
            on_run_complete, sender=TrainingRun, dispatch_uid="agent_on_run_complete"
        )
