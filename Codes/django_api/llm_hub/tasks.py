from celery import shared_task
from .models import EvalRun
from .evaluation import Evaluator

@shared_task(bind=True, max_retries=2)
def run_evaluation(self, run_id: str):
    run = EvalRun.objects.get(id=run_id)
    evaluator = Evaluator(run)
    evaluator.execute()