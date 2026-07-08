"""Metaflow‑inspired DAG executor for ML pipelines."""

import logging

from .models import PipelineRun, PipelineStepRun

logger = logging.getLogger(__name__)


class PipelineExecutor:
    def __init__(self, run: PipelineRun):
        self.run = run
        self.pipeline = run.pipeline
        self.context = run.params.copy()

    def execute(self):
        self.run.status = "running"
        self.run.started_at = timezone.now()
        self.run.save()

        try:
            for step_def in self.pipeline.steps:
                step_run = PipelineStepRun.objects.create(
                    run=self.run,
                    step_name=step_def["name"],
                    status="running",
                    started_at=timezone.now(),
                )
                # Execute step logic based on type
                if step_def["type"] == "training":
                    self._execute_training_step(step_def, step_run)
                elif step_def["type"] == "evaluation":
                    self._execute_eval_step(step_def, step_run)
                elif step_def["type"] == "compression":
                    self._execute_compression_step(step_def, step_run)
                elif step_def["type"] == "deployment":
                    self._execute_deploy_step(step_def, step_run)
                step_run.status = "completed"
                step_run.completed_at = timezone.now()
                step_run.save()

            self.run.status = "completed"
        except Exception:
            self.run.status = "failed"
            logger.exception("Pipeline failed")
        finally:
            self.run.completed_at = timezone.now()
            self.run.save()

    def _execute_training_step(self, step_def, step_run):
        from experiment.models import Experiment, TrainingRun

        config = {**self.context, **step_def.get("config", {})}
        experiment = Experiment.objects.create(name=f"Pipeline: {self.pipeline.name}")
        training_run = TrainingRun.objects.create(
            experiment=experiment, config=config, status="queued"
        )
        # Trigger Celery training task
        from experiment.tasks import run_training_task

        run_training_task.delay(str(training_run.id))
        step_run.output_artifacts = {"run_id": str(training_run.id)}
        self.context["latest_model_run_id"] = str(training_run.id)

    def _execute_eval_step(self, step_def, step_run):
        from experiment.models import TrainingRun

        run_id = self.context.get("latest_model_run_id")
        if run_id:
            run = TrainingRun.objects.get(id=run_id)
            from experiment.tasks import run_evaluation_task

            run_evaluation_task.delay(str(run.id))

    def _execute_compression_step(self, step_def, step_run):
        run_id = self.context.get("latest_model_run_id")
        if run_id:
            # Launch compression via Celery
            pass

    def _execute_deploy_step(self, step_def, step_run):
        model_version_id = self.context.get("model_version_id")
        if model_version_id:
            from .tasks import deploy_inference_endpoint

            deploy_inference_endpoint.delay(
                model_version_id, step_def.get("config", {})
            )
