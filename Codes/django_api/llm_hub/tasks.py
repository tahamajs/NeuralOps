from celery import shared_task

from .evaluation import Evaluator
from .models import EvalRun


@shared_task(bind=True, max_retries=2)
def run_evaluation(self, run_id: str):
    run = EvalRun.objects.get(id=run_id)
    evaluator = Evaluator(run)
    evaluator.execute()
