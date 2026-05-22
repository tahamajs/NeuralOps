"""Celery tasks for running autonomous research pipelines."""
import logging
from celery import shared_task
from .models import ResearchPipeline
from .engine import ResearchEngine

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def run_pipeline_async(self, pipeline_id: str):
    """Run a complete research pipeline asynchronously."""
    try:
        pipeline = ResearchPipeline.objects.get(id=pipeline_id)
    except ResearchPipeline.DoesNotExist:
        return "Pipeline not found"

    try:
        engine = ResearchEngine(pipeline)
        engine.run()
        return f"Pipeline {pipeline.name} completed successfully"
    except Exception as exc:
        logger.exception("Pipeline %s failed", pipeline_id)
        pipeline.status = 'failed'
        pipeline.save(update_fields=['status'])
        raise self.retry(exc=exc)