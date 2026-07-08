# jobscheduler/tasks.py
from celery import shared_task

from .predictive import PriorityPredictor


@shared_task
def retrain_priority_model():
    """
    Periodic task to retrain the priority prediction model using historical job data.
    Schedule this task daily (e.g., via Celery Beat) to keep the model updated.
    """
    predictor = PriorityPredictor()
    predictor.train()
    return {"status": "retrained", "model_path": predictor.model_path}
